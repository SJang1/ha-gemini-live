"""MCP (Model Context Protocol) handler for Gemini Live Audio integration.

This module provides integration with MCP servers for extended tool capabilities.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from aiohttp import ClientSession

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPServer:
    """MCP server configuration."""

    name: str
    type: str  # "sse" or "stdio"
    url: str | None = None  # For SSE servers
    token: str | None = None  # For SSE server authentication
    command: str | None = None  # For stdio servers
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False
    _process: Any = None
    _reader_task: Any = None


class MCPServerHandler:
    """Handler for MCP servers."""

    def __init__(self, session: ClientSession) -> None:
        """Initialize the handler."""
        self._session = session
        self._servers: dict[str, MCPServer] = {}
        self._tool_callbacks: dict[str, Callable] = {}

    def add_server(
        self,
        name: str,
        server_type: str,
        url: str | None = None,
        token: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Add an MCP server."""
        self._servers[name] = MCPServer(
            name=name,
            type=server_type,
            url=url,
            token=token,
            command=command,
            args=args or [],
            env=env or {},
        )

    def add_server_from_config(self, config: dict[str, Any]) -> None:
        """Add a server from configuration dictionary."""
        self.add_server(
            name=config.get("name", ""),
            server_type=config.get("type", "sse"),
            url=config.get("url"),
            command=config.get("command"),
            args=config.get("args", []),
            env=config.get("env", {}),
        )

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server."""
        if name in self._servers:
            del self._servers[name]
            return True
        return False

    def get_server(self, name: str) -> MCPServer | None:
        """Get a server by name."""
        return self._servers.get(name)

    def get_all_servers(self) -> list[MCPServer]:
        """Get all servers."""
        return list(self._servers.values())

    def get_sse_servers(self) -> list[MCPServer]:
        """Get all SSE servers."""
        return [s for s in self._servers.values() if s.type == "sse"]

    def get_stdio_servers(self) -> list[MCPServer]:
        """Get all stdio servers."""
        return [s for s in self._servers.values() if s.type == "stdio"]

    def get_server_configs(self) -> list[dict[str, Any]]:
        """Get all server configurations as dictionaries."""
        configs = []
        for server in self._servers.values():
            config = {
                "name": server.name,
                "type": server.type,
                "connected": server.connected,
                "tools_count": len(server.tools),
            }
            if server.url:
                config["url"] = server.url
            if server.command:
                config["command"] = server.command
                config["args"] = server.args
            configs.append(config)
        return configs

    async def connect_all_servers(self) -> None:
        """Connect to all configured servers."""
        for name in self._servers:
            try:
                await self.connect_server(name)
            except Exception as e:
                _LOGGER.error("Error connecting to MCP server %s: %s", name, e)

    async def connect_server(self, name: str) -> bool:
        """Connect to a specific MCP server."""
        server = self._servers.get(name)
        if not server:
            return False

        if server.type == "sse":
            return await self._connect_sse_server(server)
        elif server.type == "stdio":
            return await self._connect_stdio_server(server)

        return False

    async def _connect_sse_server(self, server: MCPServer) -> bool:
        """Connect to an SSE MCP server."""
        if not server.url:
            return False

        try:
            # Fetch available tools from the server
            async with self._session.get(f"{server.url}/tools") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tools = data.get("tools", [])
                    server.tools = [
                        MCPTool(
                            name=t.get("name", ""),
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema", {}),
                            server_name=server.name,
                        )
                        for t in tools
                    ]
                    server.connected = True
                    _LOGGER.info(
                        "Connected to SSE MCP server %s with %d tools",
                        server.name,
                        len(server.tools),
                    )
                    return True
        except Exception as e:
            _LOGGER.error("Error connecting to SSE server %s: %s", server.name, e)

        return False

    async def _connect_stdio_server(self, server: MCPServer) -> bool:
        """Connect to a stdio MCP server."""
        if not server.command:
            return False

        try:
            # Start the process
            env = {**dict(__import__("os").environ), **server.env}
            process = await asyncio.create_subprocess_exec(
                server.command,
                *server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            server._process = process

            # Send initialize message
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "gemini-live-ha",
                        "version": "1.0.0",
                    },
                },
            }
            await self._send_stdio_message(process, init_message)

            # Read response
            response = await self._read_stdio_message(process)
            if not response:
                return False

            # Send initialized notification
            await self._send_stdio_message(
                process,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )

            # Get tools list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            await self._send_stdio_message(process, tools_request)

            tools_response = await self._read_stdio_message(process)
            if tools_response and "result" in tools_response:
                tools = tools_response["result"].get("tools", [])
                server.tools = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                        server_name=server.name,
                    )
                    for t in tools
                ]

            server.connected = True
            _LOGGER.info(
                "Connected to stdio MCP server %s with %d tools",
                server.name,
                len(server.tools),
            )
            return True

        except Exception as e:
            _LOGGER.error("Error connecting to stdio server %s: %s", server.name, e)
            return False

    async def _send_stdio_message(self, process, message: dict) -> None:
        """Send a message to a stdio process."""
        if process.stdin:
            data = json.dumps(message) + "\n"
            process.stdin.write(data.encode())
            await process.stdin.drain()

    async def _read_stdio_message(self, process, timeout: float = 5.0) -> dict | None:
        """Read a message from a stdio process."""
        if not process.stdout:
            return None

        try:
            line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=timeout,
            )
            if line:
                return json.loads(line.decode())
        except asyncio.TimeoutError:
            _LOGGER.warning("Timeout reading from stdio process")
        except json.JSONDecodeError as e:
            _LOGGER.error("Error decoding stdio message: %s", e)

        return None

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from a specific MCP server."""
        server = self._servers.get(name)
        if not server:
            return

        if server._process:
            server._process.terminate()
            await server._process.wait()
            server._process = None

        server.connected = False

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in self._servers:
            await self.disconnect_server(name)

    def get_tools_as_functions(self) -> list[dict[str, Any]]:
        """Get all tools from all servers as function definitions for Gemini."""
        functions = []
        for server in self._servers.values():
            if not server.connected:
                continue
            for tool in server.tools:
                # Sanitize function name (server_name__tool_name)
                func_name = self._make_function_name(server.name, tool.name)
                functions.append({
                    "name": func_name,
                    "description": f"[{server.name}] {tool.description}",
                    "parameters": tool.input_schema,
                })
        return functions

    def _make_function_name(self, server_name: str, tool_name: str) -> str:
        """Create a valid function name from server and tool names."""
        # Replace non-alphanumeric chars with underscores
        safe_server = re.sub(r"[^a-zA-Z0-9]", "_", server_name)
        safe_tool = re.sub(r"[^a-zA-Z0-9]", "_", tool_name)
        return f"{safe_server}__{safe_tool}"

    def parse_function_name(self, func_name: str) -> tuple[str, str] | None:
        """Parse a function name back to server_name and tool_name."""
        if "__" not in func_name:
            return None
        parts = func_name.split("__", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an MCP server."""
        server = self._servers.get(server_name)
        if not server or not server.connected:
            return {"error": f"Server {server_name} not connected"}

        if server.type == "sse":
            return await self._call_sse_tool(server, tool_name, arguments)
        elif server.type == "stdio":
            return await self._call_stdio_tool(server, tool_name, arguments)

        return {"error": "Unknown server type"}

    async def _call_sse_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an SSE server."""
        try:
            async with self._session.post(
                f"{server.url}/tools/{tool_name}",
                json=arguments,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    return {"error": f"Tool call failed: {text}"}
        except Exception as e:
            return {"error": str(e)}

    async def _call_stdio_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on a stdio server."""
        if not server._process:
            return {"error": "Server process not running"}

        try:
            request = {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
            await self._send_stdio_message(server._process, request)
            response = await self._read_stdio_message(server._process, timeout=30.0)

            if response and "result" in response:
                content = response["result"].get("content", [])
                # Extract text content
                for item in content:
                    if item.get("type") == "text":
                        return {"result": item.get("text", "")}
                return response["result"]
            elif response and "error" in response:
                return {"error": response["error"]}

            return {"error": "No response from tool"}

        except Exception as e:
            return {"error": str(e)}


class HomeAssistantMCPTools:
    """Built-in Home Assistant MCP-like tools."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize with Home Assistant instance."""
        self._hass = hass

    def get_builtin_tools(self) -> list[dict[str, Any]]:
        """Get built-in HA tools as function definitions."""
        return [
            {
                "name": "get_entity_state",
                "description": "Get the current state of a Home Assistant entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID (e.g., light.living_room)",
                        },
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "call_service",
                "description": "Call a Home Assistant service to control devices",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The service domain (e.g., light, switch)",
                        },
                        "service": {
                            "type": "string",
                            "description": "The service name (e.g., turn_on, turn_off)",
                        },
                        "target": {
                            "type": "object",
                            "description": "Target entities/areas for the service",
                            "properties": {
                                "entity_id": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "area_id": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional service data",
                        },
                    },
                    "required": ["domain", "service"],
                },
            },
            {
                "name": "get_entities_by_domain",
                "description": "List all entities in a specific domain",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain (e.g., light, switch, sensor)",
                        },
                    },
                    "required": ["domain"],
                },
            },
            {
                "name": "get_area_entities",
                "description": "Get all entities in a specific area",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area_id": {
                            "type": "string",
                            "description": "The area ID or name",
                        },
                    },
                    "required": ["area_id"],
                },
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a built-in tool."""
        if tool_name == "get_entity_state":
            return await self._get_entity_state(arguments)
        elif tool_name == "call_service":
            return await self._call_service(arguments)
        elif tool_name == "get_entities_by_domain":
            return await self._get_entities_by_domain(arguments)
        elif tool_name == "get_area_entities":
            return await self._get_area_entities(arguments)

        return {"error": f"Unknown tool: {tool_name}"}

    async def _get_entity_state(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get entity state."""
        entity_id = arguments.get("entity_id", "")
        state = self._hass.states.get(entity_id)

        if state is None:
            return {"error": f"Entity {entity_id} not found"}

        return {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat(),
        }

    async def _call_service(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a Home Assistant service."""
        domain = arguments.get("domain", "")
        service = arguments.get("service", "")
        target = arguments.get("target", {})
        data = arguments.get("data", {})

        try:
            await self._hass.services.async_call(
                domain,
                service,
                {**data, **target},
                blocking=True,
            )
            return {"success": True, "message": f"Called {domain}.{service}"}
        except Exception as e:
            return {"error": str(e)}

    async def _get_entities_by_domain(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Get entities by domain."""
        domain = arguments.get("domain", "")
        entities = [
            {
                "entity_id": state.entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
            }
            for state in self._hass.states.async_all()
            if state.entity_id.startswith(f"{domain}.")
        ]
        return {"domain": domain, "entities": entities}

    async def _get_area_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get entities in an area."""
        area_id = arguments.get("area_id", "")

        from homeassistant.helpers import area_registry, entity_registry

        area_reg = area_registry.async_get(self._hass)
        entity_reg = entity_registry.async_get(self._hass)

        # Find area
        area = area_reg.async_get_area(area_id)
        if not area:
            # Try by name
            for a in area_reg.async_list_areas():
                if a.name.lower() == area_id.lower():
                    area = a
                    break

        if not area:
            return {"error": f"Area {area_id} not found"}

        # Get entities in area
        entities = []
        for entry in entity_reg.entities.values():
            if entry.area_id == area.id:
                state = self._hass.states.get(entry.entity_id)
                if state:
                    entities.append({
                        "entity_id": entry.entity_id,
                        "state": state.state,
                        "friendly_name": state.attributes.get("friendly_name"),
                    })

        return {"area": area.name, "entities": entities}
