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

from .const import (
    CONF_MCP_SERVER_NAME,
    CONF_MCP_SERVER_TYPE,
    CONF_MCP_SERVER_URL,
    CONF_MCP_SERVER_COMMAND,
    CONF_MCP_SERVER_ARGS,
    CONF_MCP_SERVER_ENV,
)

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

    async def _read_stderr(self) -> None:
        """Read and log stderr from the process to prevent buffer blocking."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                # Log stderr output for debugging
                try:
                    stderr_text = line.decode(errors="replace").strip()
                except Exception:
                    stderr_text = str(line)
                if stderr_text:
                    _LOGGER.debug("MCP server stderr (%s): %s", self.name, stderr_text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _LOGGER.debug("Error reading stderr for %s: %s", self.name, e)


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
        # Normalize args/env which may come from UI as strings
        args_val = config.get(CONF_MCP_SERVER_ARGS, [])
        if isinstance(args_val, str):
            try:
                args_val = json.loads(args_val)
            except Exception:
                args_val = [s for s in args_val.split() if s]

        env_val = config.get(CONF_MCP_SERVER_ENV, {})
        if isinstance(env_val, str):
            try:
                env_val = json.loads(env_val)
            except Exception:
                # Try comma-separated KEY=VAL pairs: key1=val1,key2=val2
                try:
                    pairs = [p for p in env_val.split(",") if p and "=" in p]
                    parsed = {}
                    for p in pairs:
                        k, v = p.split("=", 1)
                        parsed[k.strip()] = v.strip()
                    env_val = parsed
                except Exception:
                    _LOGGER.debug("Could not parse env string for server %s: %s", config.get(CONF_MCP_SERVER_NAME), env_val)
                    env_val = {}

        _LOGGER.debug("Adding MCP server from config: %s (args=%s, env=%s)", config.get(CONF_MCP_SERVER_NAME), args_val, env_val)

        self.add_server(
            name=config.get(CONF_MCP_SERVER_NAME, ""),
            server_type=config.get(CONF_MCP_SERVER_TYPE, "sse"),
            url=config.get(CONF_MCP_SERVER_URL),
            command=config.get(CONF_MCP_SERVER_COMMAND),
            args=args_val,
            env=env_val,
        )

        # Log the resulting stored config for easy verification
        srv = self._servers.get(config.get(CONF_MCP_SERVER_NAME, ""))
        if srv:
            _LOGGER.debug("Registered MCP server %s: command=%s args=%s env=%s", srv.name, srv.command, srv.args, srv.env)

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

    async def connect_server(self, name: str) -> bool:
        """Connect to a server by name. Returns True if connected."""
        server = self._servers.get(name)
        if not server:
            return False

        if server.type == "stdio":
            return await self._connect_stdio_server(server)

        # SSE servers are HTTP-based and don't require a persistent connection here
        # We consider them "connected" for the purpose of tool listing.
        return True


    async def connect_all_servers(self) -> dict[str, bool]:
        """Attempt to connect to all servers and return a map of results."""
        results: dict[str, bool] = {}
        for name in list(self._servers.keys()):
            try:
                results[name] = await self.connect_server(name)
            except Exception:
                _LOGGER.exception("Error connecting to server %s", name)
                results[name] = False
        return results

    def get_server_configs(self) -> list[dict[str, Any]]:
        """Return server configs (useful for listing via service)."""
        configs: list[dict[str, Any]] = []
        for server in self._servers.values():
            configs.append({
                "name": server.name,
                "type": server.type,
                "url": server.url,
                "command": server.command,
                "args": server.args,
                "env": server.env,
                "connected": server.connected,
                "tool_count": len(server.tools),
            })
        return configs

    async def _connect_stdio_server(self, server: MCPServer) -> bool:
        """Connect to a stdio MCP server."""
        if not server.command:
            return False

        import os

        try:
            # Prepare environment
            process_env = os.environ.copy()

            # Auto-add UV environment variables for uv/uvx commands in HASSIO
            if server.command in ("uv", "uvx"):
                uv_defaults = {
                    "UV_TOOL_DIR": "/config/.uv/tools",
                    "UV_CACHE_DIR": "/config/.uv/cache",
                    "TMPDIR": "/config/.uv/tmp",
                }
                for key, default_value in uv_defaults.items():
                    if key not in server.env:
                        server.env[key] = default_value
                        _LOGGER.debug("Auto-set %s=%s for uvx/uv command", key, default_value)

            # Create directories for known env vars
            for key in ("UV_TOOL_DIR", "UV_CACHE_DIR", "TMPDIR", "UV_TOOL_BIN_DIR"):
                value = server.env.get(key)
                if value:
                    try:
                        os.makedirs(value, exist_ok=True)
                        _LOGGER.debug("Created directory for %s: %s", key, value)
                    except Exception as e:
                        _LOGGER.warning("Failed to create directory %s: %s", value, e)

            # Start the process
            cmd = [server.command] + (server.args or [])
            _LOGGER.info("Starting stdio MCP server: %s", " ".join(cmd))

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )
            server._process = process

            # One-time quick stderr snapshot at INFO level to help operators
            # who haven't enabled DEBUG yet. This reads a few stderr lines
            # non-blocking (short timeout) and logs them at INFO so they
            # appear in default logs. This does not touch stdout (protocol)
            # and therefore won't interfere with initialize/tools messages.
            try:
                if process.stderr:
                    for _ in range(5):
                        try:
                            line = await asyncio.wait_for(process.stderr.readline(), timeout=0.05)
                        except asyncio.TimeoutError:
                            break
                        if not line:
                            break
                        try:
                            snapshot = line.decode(errors="replace").strip()
                        except Exception:
                            snapshot = str(line)
                        if snapshot:
                            _LOGGER.info("stdio[%s][stderr-snapshot]: %s", server.name, snapshot)
            except Exception:
                _LOGGER.debug("Error taking stderr snapshot for %s", server.name, exc_info=True)

            # Start per-server stderr reader to avoid blocking (start after snapshot
            # to prevent two coroutines reading stderr concurrently)
            server._reader_task = asyncio.create_task(server._read_stderr())

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

            # Read initialize response.
            # The stdio server may take time to boot; poll for up to 30s.
            init_response = None
            start = asyncio.get_running_loop().time()
            init_deadline = start + 30.0
            while asyncio.get_running_loop().time() < init_deadline:
                resp = await self._read_stdio_message(process, server.name, timeout=1.0)
                if resp:
                    init_response = resp
                    break
            if not init_response:
                _LOGGER.warning("No initialize response from stdio MCP server %s", server.name)
                # continue and attempt tools/list; some servers respond later
            else:
                _LOGGER.debug("Received initialize response from %s: %s", server.name, init_response)

            # Send initialized notification
            await self._send_stdio_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})

            # Request tools list
            tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            await self._send_stdio_message(process, tools_request)

            # Read tools response (also allow extra time for server startup)
            tools_response = None
            start = asyncio.get_running_loop().time()
            tools_deadline = start + 20.0
            while asyncio.get_running_loop().time() < tools_deadline:
                resp = await self._read_stdio_message(process, server.name, timeout=1.0)
                if resp:
                    tools_response = resp
                    break
            if not tools_response:
                _LOGGER.warning("No tools response from stdio MCP server %s", server.name)
            elif "result" in tools_response:
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
            if not server.tools:
                _LOGGER.warning(
                    "Connected to stdio MCP server %s but found %d tools",
                    server.name,
                    len(server.tools),
                )
            else:
                _LOGGER.info(
                    "Connected to stdio MCP server %s with %d tools",
                    server.name,
                    len(server.tools),
                )
            return True

        except Exception as e:
            _LOGGER.exception("Error connecting to stdio server %s: %s", server.name, e)
            return False

    async def _send_stdio_message(self, process, message: dict) -> None:
        """Send a message to a stdio process."""
        if process.stdin:
            data = json.dumps(message) + "\n"
            process.stdin.write(data.encode())
            await process.stdin.drain()

    async def _read_stdio_errors(self, stderr, server_name: str) -> None:
        """Continuously read and log stderr from a stdio process."""
        try:
            while True:
                if not stderr:
                    break
                line = await stderr.readline()
                if not line:
                    break
                try:
                    text = line.decode(errors="replace").rstrip()
                except Exception:
                    text = str(line)
                _LOGGER.debug("stdio[%s][stderr]: %s", server_name, text)
        except Exception:
            _LOGGER.exception("Error reading stdio stderr for %s", server_name)

    async def _read_stdio_message(self, process, server_name: str | None = None, timeout: float = 5.0) -> dict | None:
        """Read a message from a stdio process and log the raw stdout line at debug."""
        if not process.stdout:
            return None

        try:
            line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=timeout,
            )
            if line:
                try:
                    raw = line.decode(errors="replace").rstrip()
                except Exception:
                    raw = str(line)
                _LOGGER.debug("stdio[%s][stdout]: %s", server_name or "stdio", raw)
                return json.loads(raw)
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
        
        try:
            if server._reader_task:
                server._reader_task.cancel()
                await server._reader_task
        except Exception:
            pass

        server._reader_task = None
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
