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
from urllib.parse import urlparse, urlunparse
import socket

from .const import (
    CONF_MCP_SERVER_NAME,
    CONF_MCP_SERVER_TYPE,
    CONF_MCP_SERVER_URL,
    CONF_MCP_SERVER_COMMAND,
    CONF_MCP_SERVER_ARGS,
    CONF_MCP_SERVER_ENV,
    CONF_MCP_SERVER_TOKEN,
    CONF_MCP_SERVER_AUTH_HEADER,
    CONF_MCP_SERVER_HEADERS,
)

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def _strip_surrounding_quotes(val: str) -> str:
    """Remove surrounding single or double quotes from a string."""
    if not isinstance(val, str):
        return val
    # Trim whitespace first
    s = val.strip()
    # Remove any leading or trailing single or double quote characters.
    # Use strip so that unmatched or stray quotes (e.g. trailing ") are removed too.
    return s.strip('"\'')


def _mask_token(token: str | None) -> str:
    """Return a masked representation of a token for safe logging."""
    if not token:
        return "None"
    try:
        t = token.strip()
    except Exception:
        return "****"
    if t.lower().startswith("bearer "):
        return "Bearer ****"
    return "****"


def _normalize_auth_value(val: str | None) -> str | None:
    """Normalize various forms of an authorization/header value into a safe header value.

    Accepts raw token (eyJ...), full header (Bearer ...), or a string like
    'Authorization: Bearer ...' and returns a value suitable for the
    Authorization header (e.g. 'Bearer eyJ...'). Returns None if input is falsy.
    """
    if not val:
        return None
    try:
        s = _strip_surrounding_quotes(val).strip()
    except Exception:
        s = str(val).strip()

    # If the user pasted a full header like 'Authorization: Bearer ...', remove the leading key
    if ":" in s and s.lower().startswith("authorization"):
        try:
            _, rest = s.split(":", 1)
            s = rest.strip()
        except Exception:
            pass

    # If already starts with Bearer, return as-is
    if s.lower().startswith("bearer "):
        return s

    # Otherwise assume it's a raw token and prefix
    if s:
        return f"Bearer {s}"

    return None



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
    headers: dict[str, str] = field(default_factory=dict)
    command: str | None = None  # For stdio servers
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False
    _process: Any = None
    _reader_task: Any = None
    session_id: str | None = None  # For HTTP/Streamable-HTTP servers
    _http_sse_resp: Any = None

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
        headers: dict[str, str] | None = None,
    ) -> None:
        """Add an MCP server."""
        # Do not normalize or modify the user's provided token/header values here.
        self._servers[name] = MCPServer(
            name=name,
            type=server_type,
            url=url,
            token=token,
            command=command,
            args=args or [],
            env=env or {},
            headers=headers or {},
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
        headers_val = config.get(CONF_MCP_SERVER_HEADERS, {})
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
                        parsed[_strip_surrounding_quotes(k.strip())] = _strip_surrounding_quotes(v.strip())
                    env_val = parsed
                except Exception:
                    _LOGGER.debug("Could not parse env string for server %s: %s", config.get(CONF_MCP_SERVER_NAME), env_val)
                    env_val = {}

        # Parse headers similarly to env (accept JSON or comma-separated key=value or key: value)
        if isinstance(headers_val, str):
            try:
                headers_val = json.loads(headers_val)
            except Exception:
                try:
                    pairs = [p for p in headers_val.split(",") if p and ("=" in p or ":" in p)]
                    parsed_h = {}
                    for p in pairs:
                        if ":" in p:
                            k, v = p.split(":", 1)
                        else:
                            k, v = p.split("=", 1)
                        parsed_h[_strip_surrounding_quotes(k.strip())] = _strip_surrounding_quotes(v.strip())
                    headers_val = parsed_h
                except Exception:
                    _LOGGER.debug("Could not parse headers string for server %s: %s", config.get(CONF_MCP_SERVER_NAME), headers_val)
                    headers_val = {}

            # If the user provided an explicit Authorization value in the config
            # (CONF_MCP_SERVER_AUTH_HEADER or legacy token), prefer using that
            # verbatim as an HTTP header instead of normalizing or adding prefixes.
            try:
                auth_raw = config.get(CONF_MCP_SERVER_AUTH_HEADER, config.get(CONF_MCP_SERVER_TOKEN))
                if auth_raw:
                    # Only set Authorization header if not already provided in parsed headers
                    if isinstance(headers_val, dict) and not any(k.lower() == "authorization" for k in headers_val.keys()):
                        headers_val["Authorization"] = _strip_surrounding_quotes(auth_raw)
            except Exception:
                pass

        _LOGGER.debug(
            "Adding MCP server from config: %s (args=%s, env=%s, headers=%s)",
            config.get(CONF_MCP_SERVER_NAME),
            args_val,
            env_val,
            headers_val,
        )

        # Pass the raw auth via headers (if present); leave token alone so
        # we don't attempt to mutate or prefix it later.
        self.add_server(
            name=config.get(CONF_MCP_SERVER_NAME, ""),
            server_type=config.get(CONF_MCP_SERVER_TYPE, "sse"),
            url=config.get(CONF_MCP_SERVER_URL),
            token=config.get(CONF_MCP_SERVER_TOKEN),
            command=config.get(CONF_MCP_SERVER_COMMAND),
            args=args_val,
            env=env_val,
            headers=headers_val,
        )

        # Log the resulting stored config for easy verification
        srv = self._servers.get(config.get(CONF_MCP_SERVER_NAME, ""))
        if srv:
            # Don't log raw token/header values; log masked presence only
            auth_in_headers = any(k.lower() == "authorization" for k in (srv.headers or {}))
            masked = _mask_token(srv.token)
            _LOGGER.debug(
                "Registered MCP server %s: command=%s args=%s env=%s token=%s auth_in_headers=%s",
                srv.name,
                srv.command,
                srv.args,
                srv.env,
                masked,
                auth_in_headers,
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

    def get_http_servers(self) -> list[MCPServer]:
        """Get all HTTP/streamable-HTTP servers."""
        return [s for s in self._servers.values() if s.type == "http"]

    def get_stdio_servers(self) -> list[MCPServer]:
        """Get all stdio servers."""
        return [s for s in self._servers.values() if s.type == "stdio"]

    async def connect_server(self, name: str) -> bool:
        """Connect to a server by name. Returns True if connected."""
        server = self._servers.get(name)
        if not server:
            return False

        _LOGGER.debug("Connecting to MCP server %s (type=%s) cmd=%s url=%s", server.name, server.type, server.command, server.url)

        # If server explicitly requests stdio, prefer that.
        if server.type == "stdio" or (server.command and server.type == ""):
            _LOGGER.debug("Attempting stdio connect for %s", server.name)
            ok = await self._connect_stdio_server(server)
            if ok:
                return True

        # If declared as http/sse, or unknown type with a URL, attempt HTTP discovery
        if server.type in ("http", "sse") or server.url:
            _LOGGER.debug("Attempting HTTP discovery for %s", server.name)
            ok = await self._connect_http_server(server)
            if ok:
                return True

        # As a last resort, if command is specified try stdio again
        if server.command:
            _LOGGER.debug("Fallback: attempting stdio connect for %s (final attempt)", server.name)
            return await self._connect_stdio_server(server)

        _LOGGER.warning("No suitable transport found for server %s (type=%s, url=%s, command=%s)", server.name, server.type, server.url, server.command)
        return False

    async def _connect_http_server(self, server: MCPServer) -> bool:
        """Discover tools from an HTTP/Streamable-HTTP MCP server.

        This will POST to the server's /tools/list endpoint and populate
        server.tools if the server responds with a tools list.
        """
        if not server.url:
            return False

        try:
            async def _attempt_alternate_host_post(orig_url: str, payload: Any, hdrs: dict[str, str], timeout: int = 10):
                """Try posting to alternate hostnames/IPs when the original URL returned 401.

                Returns tuple (status:int, data: Any, used_url: str) or (None, None, None).
                """
                try:
                    parsed = urlparse(orig_url)
                    host = parsed.hostname or ""
                    portsuf = f":{parsed.port}" if parsed.port else ""
                    candidates: list[str] = []
                    # If host is localhost or 127.0.0.1, try both forms and the machine IP
                    if host in ("localhost", "127.0.0.1"):
                        candidates.extend(["127.0.0.1"])
                        try:
                            ip = socket.gethostbyname(socket.gethostname())
                            if ip and ip not in candidates and ip != host:
                                candidates.append(ip)
                        except Exception:
                            pass
                    else:
                        # Try resolving hostname to IP
                        try:
                            ip = socket.gethostbyname(host)
                            if ip and ip != host:
                                candidates.append(ip)
                        except Exception:
                            pass

                    for candidate in candidates:
                        new_netloc = f"{candidate}{portsuf}"
                        new_url = urlunparse((parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
                        try:
                            masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in hdrs.items()}
                            _LOGGER.debug("Retrying POST to alternate host %s headers=%s", new_url, masked)
                        except Exception:
                            pass
                        try:
                            async with self._session.post(new_url, json=payload, headers=hdrs, timeout=timeout) as resp2:
                                st = resp2.status
                                ctype2 = resp2.headers.get("Content-Type", "")
                                text2 = await resp2.text()
                                if st == 200:
                                    try:
                                        return st, await resp2.json(), new_url
                                    except Exception:
                                        return st, text2, new_url
                                _LOGGER.debug("Alternate host POST %s returned %s body=%s", new_url, st, text2[:500])
                        except Exception as e:
                            _LOGGER.debug("Alternate host POST to %s failed: %s", new_url, e)
                except Exception:
                    pass
                return None, None, None

            # Prepare headers and token like SSE calls
            request_headers = {**(server.headers or {})}
            if server.token:
                if not any(h.lower() == "authorization" for h in request_headers):
                    auth_value = server.token
                    if not auth_value.lower().startswith("bearer "):
                        auth_value = f"Bearer {auth_value}"
                    request_headers["Authorization"] = auth_value

            # Log whether Authorization will be included (masked)
            try:
                auth_present = any(h.lower() == "authorization" for h in request_headers)
                _LOGGER.debug(
                    "HTTP init headers for %s: Authorization_present=%s token_mask=%s",
                    server.name,
                    auth_present,
                    _mask_token(server.token),
                )
            except Exception:
                pass

            base = server.url.rstrip("/")
            # Ensure Host header matches the target netloc (helps proxies/auth that rely on Host)
            try:
                parsed_base = urlparse(base)
                base_netloc = parsed_base.netloc
                if base_netloc and not any(h.lower() == "host" for h in request_headers):
                    request_headers["Host"] = base_netloc
            except Exception:
                base_netloc = None

            # Build Accept header per spec: support JSON and event-stream
            request_headers.setdefault("Accept", "application/json, text/event-stream")

            # 1) Try to POST an InitializeRequest to the MCP endpoint (new Streamable HTTP transport)
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "gemini-live-ha", "version": "1.0.0"},
                },
            }

            headers_with_sid = {**request_headers}

            try:
                # Log masked headers that will be sent with initialize POST for diagnostics
                try:
                    masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in request_headers.items()}
                    _LOGGER.debug("HTTP initialize POST headers for %s: %s", server.name, masked)
                except Exception:
                    pass

                # Detect double-Bearer without printing token
                try:
                    auth_val = request_headers.get("Authorization") if isinstance(request_headers, dict) else None
                    double_bearer = False
                    if isinstance(auth_val, str):
                        double_bearer = auth_val.lower().count("bearer") > 1
                    _LOGGER.debug("Initialize POST Authorization double_bearer=%s", double_bearer)
                except Exception:
                    pass

                # Use raw post so we can keep the response open if it's an SSE stream.
                resp = await self._session.post(f"{base}", json=init_message, headers=request_headers, timeout=10)
                # If server accepts initialize, it may return JSON or an SSE stream.
                if 400 <= resp.status < 500:
                    # Old HTTP+SSE server or unsupported: fallback to opening GET to parse endpoint event
                    _LOGGER.debug("Initialize POST returned %d: falling back to GET for %s", resp.status, server.name)
                    init_failed = True
                    # If 401 Unauthorized, try alternate host forms (IP instead of localhost)
                    if resp.status == 401:
                        try:
                            st, data_alt, used_url = await _attempt_alternate_host_post(base, init_message, request_headers, timeout=10)
                            if st == 200 and data_alt is not None:
                                # Use alternate URL as base
                                parsed_used = urlparse(used_url)
                                alt_base = f"{parsed_used.scheme}://{parsed_used.netloc}"
                                server.url = alt_base
                                headers_with_sid = {**request_headers}
                                # if data_alt contains session id, set it
                                if isinstance(data_alt, dict):
                                    sid = data_alt.get("sessionId") or data_alt.get("mcpSessionId") or data_alt.get("result", {}).get("sessionId")
                                    if sid:
                                        server.session_id = sid
                                        headers_with_sid["Mcp-Session-Id"] = server.session_id
                                init_failed = False
                                _LOGGER.info("Initialize succeeded via alternate host %s for %s", used_url, server.name)
                        except Exception:
                            pass
                else:
                    init_failed = False
                    sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                    data = None
                    ctype = resp.headers.get("Content-Type", "")
                    if "application/json" in ctype:
                        try:
                            data = await resp.json()
                        except Exception:
                            data = None
                    # If JSON contains session id, use it
                    if not sid and isinstance(data, dict):
                        sid = data.get("sessionId") or data.get("mcpSessionId") or data.get("result", {}).get("sessionId")

                    if sid:
                        server.session_id = sid
                        headers_with_sid["Mcp-Session-Id"] = server.session_id
                        _LOGGER.debug("Got MCP session id for %s: %s", server.name, server.session_id)

                    # If response is an SSE stream, keep the response open and start reader task.
                    if "text/event-stream" in ctype:
                        server._http_sse_resp = resp
                        server._reader_task = asyncio.create_task(self._read_http_sse(server, resp))
                    else:
                        # Not a streaming response; release immediately to free connection.
                        try:
                            await resp.release()
                        except Exception:
                            pass

            except Exception as e:
                _LOGGER.debug("Initialize POST failed for %s: %s", server.name, e)
                init_failed = True

            # If initialize POST failed with 4xx, attempt GET to detect old HTTP+SSE endpoint
            if init_failed:
                try:
                    # per backwards compatibility, try GET and expect 'endpoint' event
                    # Use the same headers we would use for POST (including Authorization/session id)
                    get_headers = {**headers_with_sid, "Accept": "text/event-stream"}
                    async with self._session.get(base, headers=get_headers, timeout=10) as get_resp:
                        if get_resp.status == 200 and "text/event-stream" in get_resp.headers.get("Content-Type", ""):
                            # parse the first SSE event and look for endpoint in data
                            endpoint = None
                            async for raw in get_resp.content:
                                if not raw:
                                    continue
                                try:
                                    text = raw.decode(errors="replace")
                                except Exception:
                                    text = str(raw)
                                for line in text.splitlines():
                                    if line.startswith("data:"):
                                        payload = line[len("data:"):].strip()
                                        try:
                                            j = json.loads(payload)
                                            # old transport: endpoint event contains POST endpoint
                                            if isinstance(j, dict) and j.get("event") == "endpoint":
                                                endpoint = j.get("data") or j.get("endpoint") or j.get("uri")
                                                break
                                        except Exception:
                                            pass
                                if endpoint:
                                    break
                            try:
                                await get_resp.release()
                            except Exception:
                                pass
                            if endpoint:
                                # use endpoint as base for future POSTs
                                _LOGGER.debug("Detected legacy HTTP+SSE endpoint for %s -> %s", server.name, endpoint)
                                base = endpoint.rstrip("/")
                                server.url = base
                                headers_with_sid = {**request_headers}
                            else:
                                _LOGGER.debug("No endpoint event found in GET for %s", server.name)
                except Exception as e:
                    _LOGGER.debug("GET fallback failed for %s: %s", server.name, e)

            # Send initialized notification (best-effort) to confirm initialization
            try:
                notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                try:
                    masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in headers_with_sid.items()}
                    _LOGGER.debug("Initialize notification POST to %s headers=%s", base, masked)
                except Exception:
                    pass
                await self._session.post(f"{base}", json=notif, headers=headers_with_sid, timeout=5)
            except Exception:
                pass

            # Fetch tools list via POST {base}/tools/list using session header when available
            tools = []
            tools_url = f"{base.rstrip('/')}/tools/list"

            async def _try_fetch_tools(url: str, method: str = "post"):
                try:
                    _LOGGER.debug("Attempting %s %s for tools (server=%s)", method.upper(), url, server.name)
                    try:
                        hdrs_for_log = request_headers if request_headers is not None else {}
                        masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in hdrs_for_log.items()}
                        _LOGGER.debug("Tools fetch headers for %s: %s", server.name, masked)
                    except Exception:
                        pass

                    if method.lower() == "post":
                        # Some MCP HTTP servers expect JSON-RPC payloads for tools/list
                        payload = {}
                        if url.rstrip("/").endswith("/tools/list") or url.rstrip("/") == base.rstrip("/"):
                            payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

                        try:
                            auth_val = request_headers.get("Authorization") if isinstance(request_headers, dict) else None
                            double_bearer = False
                            if isinstance(auth_val, str):
                                double_bearer = auth_val.lower().count("bearer") > 1
                            _LOGGER.debug("Tools POST Authorization double_bearer=%s", double_bearer)
                        except Exception:
                            pass

                        async with self._session.post(url, json=payload, headers=request_headers, timeout=10) as resp:
                            status = resp.status
                            ctype = resp.headers.get("Content-Type", "")
                            text = await resp.text()
                            _LOGGER.debug("Tools fetch resp status=%s content-type=%s body=%s", status, ctype, text[:1000])
                            if resp.status != 200:
                                # If unauthorized, try alternate host for this POST
                                if resp.status == 401:
                                    try:
                                        st2, data2, used2 = await _attempt_alternate_host_post(url, payload, request_headers, timeout=10)
                                        if st2 == 200 and data2 is not None:
                                            # discovered via alternate host
                                            parsed_used2 = urlparse(used2)
                                            alt_base2 = f"{parsed_used2.scheme}://{parsed_used2.netloc}"
                                            server.url = alt_base2
                                            # keep using the same request headers
                                            return data2, st2, None
                                    except Exception:
                                        pass
                                return None, status, text
                            # If the server returned an SSE stream as a whole in the
                            # response body, try to extract JSON from `data:` lines.
                            if "text/event-stream" in ctype:
                                try:
                                    data_lines: list[str] = []
                                    for line in text.splitlines():
                                        if line.startswith("data:"):
                                            data_lines.append(line[len("data:"):].strip())
                                    if data_lines:
                                        combined = "\n".join(data_lines)
                                        try:
                                            parsed = json.loads(combined)
                                            return parsed, status, None
                                        except Exception:
                                            # Fallback: try to locate first `{` and parse
                                            idx = combined.find("{")
                                            if idx != -1:
                                                try:
                                                    parsed = json.loads(combined[idx:])
                                                    return parsed, status, None
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                            try:
                                return await resp.json(), status, None
                            except Exception:
                                return text, status, None
                    else:
                        # Some servers require GET requests to include Accept: text/event-stream
                        get_headers = {**(request_headers or {}), "Accept": "text/event-stream, application/json"}
                        try:
                            masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in get_headers.items()}
                            _LOGGER.debug("Tools fetch GET to %s headers=%s", url, masked)
                        except Exception:
                            pass
                        async with self._session.get(url, headers=get_headers, timeout=10) as resp:
                            status = resp.status
                            ctype = resp.headers.get("Content-Type", "")
                            text = await resp.text()
                            _LOGGER.debug("Tools fetch (GET) resp status=%s content-type=%s body=%s", status, ctype, text[:1000])
                            if resp.status != 200:
                                return None, status, text
                            # If GET returned an SSE payload, try to extract JSON from data: lines
                            if "text/event-stream" in ctype:
                                try:
                                    data_lines: list[str] = []
                                    for line in text.splitlines():
                                        if line.startswith("data:"):
                                            data_lines.append(line[len("data:"):].strip())
                                    if data_lines:
                                        combined = "\n".join(data_lines)
                                        try:
                                            parsed = json.loads(combined)
                                            return parsed, status, None
                                        except Exception:
                                            idx = combined.find("{")
                                            if idx != -1:
                                                try:
                                                    parsed = json.loads(combined[idx:])
                                                    return parsed, status, None
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                            try:
                                return await resp.json(), status, None
                            except Exception:
                                return text, status, None
                except Exception as e:
                    _LOGGER.debug("Exception fetching tools from %s: %s", url, e)
                    return None, None, str(e)

            def _extract_tools_from_obj(data_obj):
                # Accept several shapes: {tools: [...]}, {result: {tools: [...]}}, [ ... ]
                if data_obj is None:
                    return []
                if isinstance(data_obj, list):
                    return data_obj
                if isinstance(data_obj, dict):
                    if "tools" in data_obj and isinstance(data_obj["tools"], list):
                        return data_obj["tools"]
                    if "result" in data_obj and isinstance(data_obj["result"], dict) and "tools" in data_obj["result"]:
                        return data_obj["result"]["tools"]
                    # legacy nesting
                    for key in ("data", "payload"):
                        if key in data_obj and isinstance(data_obj[key], dict) and "tools" in data_obj[key]:
                            return data_obj[key]["tools"]
                return []

            # Try several endpoints/fallbacks to discover tools
            fetch_attempts = [
                (tools_url, "post"),
                (base, "post"),
                (tools_url, "get"),
                (base + "/tools/list", "get"),
            ]

            data = None
            for url, method in fetch_attempts:
                data_obj, status, error_text = await _try_fetch_tools(url, method)
                if data_obj is None:
                    # if we got an error body but 200 wasn't returned, continue
                    continue
                # extract tools
                candidate = _extract_tools_from_obj(data_obj)
                if candidate:
                    tools = candidate
                    break

            if not tools:
                _LOGGER.warning("Failed to discover tools for HTTP MCP server %s (tried %d endpoints)", server.name, len(fetch_attempts))
                return False

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

                # Attempt to open a listening GET SSE stream for server-initiated messages (optional)
            try:
                # Use Last-Event-ID support when we reconnect later; include Authorization/session headers
                get_headers = {**headers_with_sid, "Accept": "text/event-stream"}
                if server.session_id:
                    get_headers["Mcp-Session-Id"] = server.session_id
                    try:
                        masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in get_headers.items()}
                        _LOGGER.debug("Opening SSE listening GET to %s headers=%s", base, masked)
                    except Exception:
                        pass
                resp = await self._session.get(base, headers=get_headers, timeout=10)
                ctype = resp.headers.get("Content-Type", "")
                if resp.status == 200 and "text/event-stream" in ctype:
                    server._http_sse_resp = resp
                    server._reader_task = asyncio.create_task(self._read_http_sse(server, resp))
                    _LOGGER.info("Opened HTTP SSE listening stream for %s", server.name)
                else:
                    try:
                        await resp.release()
                    except Exception:
                        pass
            except Exception:
                pass

            _LOGGER.info("Connected to HTTP MCP server %s with %d tools", server.name, len(server.tools))
            return True

        except Exception as e:
            _LOGGER.exception("Error connecting to HTTP MCP server %s: %s", server.name, e)
            return False


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
                "headers": server.headers,
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

    async def _read_http_sse(self, server: MCPServer, resp) -> None:
        """Read SSE-style events from an open HTTP response.

        This reads lines and emits debug logs; when an SSE event with JSON
        data is received and contains tool results, it will log them.
        """
        try:
            if not resp:
                return

            # Simple SSE parser: collect lines until a blank line, then parse fields
            buffer_lines: list[str] = []
            async for raw in resp.content:
                if not raw:
                    continue
                try:
                    text = raw.decode(errors="replace")
                except Exception:
                    text = str(raw)

                # Split by newlines; accumulate until a blank line separates events
                for line in text.splitlines():
                    if line == "":
                        # End of event
                        if not buffer_lines:
                            continue
                        event_id = None
                        event_type = None
                        data_lines: list[str] = []
                        for buf_line in buffer_lines:
                            if buf_line.startswith("id:"):
                                event_id = buf_line[len("id:"):].strip()
                            elif buf_line.startswith("event:"):
                                event_type = buf_line[len("event:"):].strip()
                            elif buf_line.startswith("data:"):
                                data_lines.append(buf_line[len("data:"):].strip())
                            else:
                                # ignore other SSE fields
                                pass

                        data_text = "\n".join(data_lines).strip()
                        if data_text:
                            _LOGGER.debug("HTTP SSE event [%s] event=%s id=%s data=%s", server.name, event_type or "", event_id or "", data_text)
                            try:
                                j = json.loads(data_text)
                                # If server sends requests/notifications, we may need to handle them.
                                if isinstance(j, dict) and j.get("method"):
                                    # Server-initiated request or notification
                                    _LOGGER.info("HTTP SSE server request/notification from %s: %s", server.name, j)
                                elif isinstance(j, dict) and j.get("result"):
                                    _LOGGER.info("HTTP SSE server response from %s: %s", server.name, j)
                            except Exception:
                                _LOGGER.debug("Non-JSON SSE data from %s: %s", server.name, data_text)

                        buffer_lines = []
                    else:
                        buffer_lines.append(line)

            # end async for

        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("Error reading HTTP SSE for %s", server.name)
        finally:
            try:
                await resp.release()
            except Exception:
                pass

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from a specific MCP server."""
        server = self._servers.get(name)
        if not server:
            return

        # If this is a stdio server, terminate the process
        if server._process:
            server._process.terminate()
            await server._process.wait()
            server._process = None

        # If this is an HTTP/Streamable-HTTP server, optionally send DELETE to terminate session
        if server.type == "http" and server.url and server.session_id:
            try:
                del_headers = {"Mcp-Session-Id": server.session_id}
                try:
                    masked = {k: ("****" if k.lower() == "authorization" else v) for k, v in del_headers.items()}
                    _LOGGER.debug("DELETE session to %s headers=%s", server.url, masked)
                except Exception:
                    pass
                await self._session.delete(server.url, headers=del_headers, timeout=5)
            except Exception:
                pass

        try:
            if server._reader_task:
                server._reader_task.cancel()
                await server._reader_task
        except Exception:
            pass

        # If we have an open aiohttp response for SSE, release it
        try:
            if getattr(server, "_http_sse_resp", None):
                try:
                    await server._http_sse_resp.release()
                except Exception:
                    pass
                server._http_sse_resp = None
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
            _LOGGER.debug("Tool call attempted for disconnected server %s: %s(%s)", server_name, tool_name, arguments)
            return {"error": f"Server {server_name} not connected"}
        _LOGGER.debug("Calling tool %s on server %s (type=%s) args=%s", tool_name, server_name, server.type, arguments)

        # Route based on declared type; if unknown, try transports heuristically.
        if server.type == "sse":
            return await self._call_sse_tool(server, tool_name, arguments)
        if server.type == "stdio":
            return await self._call_stdio_tool(server, tool_name, arguments)
        if server.type == "http":
            return await self._call_http_tool(server, tool_name, arguments)

        # Unknown type: prefer HTTP if URL available, otherwise stdio if command exists
        if server.url:
            _LOGGER.debug("Unknown server type; using HTTP call for %s", server.name)
            return await self._call_http_tool(server, tool_name, arguments)
        if server._process:
            _LOGGER.debug("Unknown server type; using stdio call for %s", server.name)
            return await self._call_stdio_tool(server, tool_name, arguments)

        _LOGGER.error("Cannot call tool %s on server %s: no transport available", tool_name, server_name)
        return {"error": "Unknown server type or no transport available"}

    async def _call_sse_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an SSE server."""
        try:
            # Prepare headers: start from configured headers, then fall back to token if provided
            request_headers = {**(server.headers or {})}
            if server.token:
                # Prefer explicit Authorization header in headers; otherwise use token
                if not any(h.lower() == "authorization" for h in request_headers):
                    auth_value = server.token
                    if not auth_value.lower().startswith("bearer "):
                        auth_value = f"Bearer {auth_value}"
                    request_headers["Authorization"] = auth_value

            # Log masked auth presence
            try:
                auth_present = any(h.lower() == "authorization" for h in request_headers)
                _LOGGER.debug(
                    "SSE tool call headers for %s: Authorization_present=%s token_mask=%s",
                    server.name,
                    auth_present,
                    _mask_token(server.token),
                )
            except Exception:
                pass

            # Build JSON-RPC tools/call request per MCP spec
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 100 + int(asyncio.get_event_loop().time()) % 100000,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }

            primary = f"{server.url.rstrip('/')}/tools/call"
            fallback = f"{server.url.rstrip('/')}"
            tried = []
            for url in (primary, fallback):
                _LOGGER.debug("SSE tool POST attempt %s headers=%s rpc=%s", url, {k: ("****" if k.lower()=="authorization" else v) for k,v in request_headers.items()}, rpc_request)
                try:
                    auth_val = request_headers.get("Authorization") if isinstance(request_headers, dict) else None
                    double_bearer = False
                    if isinstance(auth_val, str):
                        double_bearer = auth_val.lower().count("bearer") > 1
                    _LOGGER.debug("SSE tool POST Authorization double_bearer=%s", double_bearer)
                except Exception:
                    pass
                async with self._session.post(url, json=rpc_request, headers=request_headers, timeout=30) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    text = await resp.text()
                    _LOGGER.debug("SSE tool response from %s status=%s content-type=%s body=%s", url, resp.status, ctype, text[:2000])

                    # If endpoint not found, try next
                    if resp.status in (404, 405):
                        tried.append((url, resp.status, text))
                        _LOGGER.debug("SSE tool endpoint %s returned %s, trying next fallback", url, resp.status)
                        continue

                    # Non-200 error
                    if resp.status != 200:
                        return {"error": f"Tool call failed: {text}", "status": resp.status}

                    # If JSON, return parsed
                    if "application/json" in ctype:
                        try:
                            return await resp.json()
                        except Exception:
                            return {"result": text}

                    # If SSE stream or text, try to extract JSON data blocks
                    if "text/event-stream" in ctype or text.startswith("event:") or text.strip().startswith("{"):
                        data_lines = []
                        for line in text.splitlines():
                            if line.startswith("data:"):
                                data_lines.append(line[len("data:"):].strip())
                        data_text = "\n".join(data_lines).strip()
                        if data_text:
                            try:
                                j = json.loads(data_text)
                                return j
                            except Exception:
                                pass

                        try:
                            j = json.loads(text)
                            return j
                        except Exception:
                            return {"result": text}

                    return {"result": text}
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
            _LOGGER.error("Stdio tool call failed: process for %s not running", server.name)
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
            _LOGGER.debug("Stdio sending request to %s: %s", server.name, request)
            await self._send_stdio_message(server._process, request)
            response = await self._read_stdio_message(server._process, timeout=30.0)

            if response and "result" in response:
                content = response["result"].get("content", [])
                # Extract text content
                for item in content:
                    if item.get("type") == "text":
                        _LOGGER.debug("Stdio tool %s result text: %s", tool_name, item.get("text", "")[:1000])
                        return {"result": item.get("text", "")}
                return response["result"]
            elif response and "error" in response:
                _LOGGER.debug("Stdio tool %s returned error: %s", tool_name, response["error"])
                return {"error": response["error"]}

            return {"error": "No response from tool"}

        except Exception as e:
            return {"error": str(e)}

    async def _call_http_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an HTTP/Streamable-HTTP server."""
        if not server.url:
            _LOGGER.error("HTTP tool call failed: server %s has no URL configured", server.name)
            return {"error": "Server URL not configured"}

        try:
            headers = {**(server.headers or {})}
            # Include Accept as required
            headers.setdefault("Accept", "application/json, text/event-stream")
            if server.session_id:
                headers["Mcp-Session-Id"] = server.session_id
            if server.token and not any(h.lower() == "authorization" for h in headers):
                auth_value = server.token
                if not auth_value.lower().startswith("bearer "):
                    auth_value = f"Bearer {auth_value}"
                headers["Authorization"] = auth_value

            # Log masked auth presence for HTTP tool call
            try:
                auth_present = any(h.lower() == "authorization" for h in headers)
                _LOGGER.debug(
                    "HTTP tool call headers for %s: Authorization_present=%s token_mask=%s",
                    server.name,
                    auth_present,
                    _mask_token(server.token),
                )
            except Exception:
                pass

            # Build JSON-RPC tools/call request per MCP spec
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 200 + int(asyncio.get_event_loop().time()) % 100000,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }

            primary = server.url.rstrip("/") + "/tools/call"
            fallback = server.url.rstrip("/")
            for url in (primary, fallback):
                _LOGGER.debug("HTTP tool POST attempt %s headers=%s rpc=%s", url, {k: ("****" if k.lower()=="authorization" else v) for k,v in headers.items()}, rpc_request)
                try:
                    auth_val = headers.get("Authorization") if isinstance(headers, dict) else None
                    double_bearer = False
                    if isinstance(auth_val, str):
                        double_bearer = auth_val.lower().count("bearer") > 1
                    _LOGGER.debug("HTTP tool POST Authorization double_bearer=%s", double_bearer)
                except Exception:
                    pass
                async with self._session.post(url, json=rpc_request, headers=headers, timeout=30) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    text_body = await resp.text()
                    _LOGGER.debug("HTTP tool response from %s status=%s content-type=%s body=%s", url, resp.status, ctype, text_body[:2000])

                    # If endpoint not found, try next fallback
                    if resp.status in (404, 405):
                        _LOGGER.debug("HTTP tool endpoint %s returned %s, trying fallback", url, resp.status)
                        continue

                    if resp.status == 200:
                        # Try JSON first
                        if "application/json" in ctype:
                            try:
                                return await resp.json()
                            except Exception:
                                return {"result": text_body}
                        if "text/event-stream" in ctype or text_body.startswith("event:"):
                            # parse SSE inline (simple parser)
                            buffer_lines: list[str] = []
                            for line in text_body.splitlines():
                                if line == "":
                                    data_lines = [ln[len("data:"):].strip() for ln in buffer_lines if ln.startswith("data:")]
                                    data_text = "\n".join(data_lines).strip()
                                    buffer_lines = []
                                    if not data_text:
                                        continue
                                    try:
                                        j = json.loads(data_text)
                                        if isinstance(j, dict) and ("result" in j or "error" in j):
                                            return j
                                    except Exception:
                                        pass
                                else:
                                    buffer_lines.append(line)
                            return {"error": "No response in SSE stream"}
                        return {"result": text_body}
                    elif resp.status == 202:
                        return {"status": "accepted"}
                    else:
                        return {"error": f"Tool call failed: {text_body}", "status": resp.status}

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
