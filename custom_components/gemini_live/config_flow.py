"""Config flow for Gemini Live integration."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_API_KEY,
    CONF_ENABLE_AFFECTIVE_DIALOG,
    CONF_ENABLE_GOOGLE_SEARCH,
    CONF_ENABLE_PROACTIVE_AUDIO,
    CONF_INSTRUCTIONS,
    CONF_MCP_SERVER_ARGS,
    CONF_MCP_SERVER_COMMAND,
    CONF_MCP_SERVER_ENABLED,
    CONF_MCP_SERVER_ENV,
    CONF_MCP_SERVER_NAME,
    CONF_MCP_SERVER_TOKEN,
    CONF_MCP_SERVER_TYPE,
    CONF_MCP_SERVER_URL,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_VOICE,
    CONF_ENABLE_PERSONALIZATION,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_VOICE,
    DOMAIN,
    MCP_SERVER_TYPE_SSE,
    MCP_SERVER_TYPE_STDIO,
    MODELS,
    VOICES,
)

_LOGGER = logging.getLogger(__name__)


async def validate_api_key(hass: HomeAssistant, api_key: str) -> bool:
    """Validate the API key by attempting a simple API call."""
    session = async_get_clientsession(hass)
    
    # Test the API key with a simple models list request
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        async with session.get(url) as response:
            return response.status == 200
    except aiohttp.ClientError:
        return False


class GeminiLiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Gemini Live."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._api_key: str | None = None
        self._config: dict[str, Any] = {}
        self._mcp_servers: list[dict[str, Any]] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            api_key = user_input[CONF_API_KEY]

            if await validate_api_key(self.hass, api_key):
                self._api_key = api_key
                return await self.async_step_configure()
            else:
                errors["base"] = "invalid_auth"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )

    async def async_step_configure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure the integration settings."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Save config and, if personalization was enabled, require explicit confirmation.
            self._config = user_input
            if self._config.get(CONF_ENABLE_PERSONALIZATION):
                return await self.async_step_personalization_confirm()
            return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="configure",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_NAME, default="Gemini Live"): str,
                    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_VOICE, default=DEFAULT_VOICE): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=VOICES,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_INSTRUCTIONS, default=DEFAULT_INSTRUCTIONS
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=2.0,
                            step=0.1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_ENABLE_GOOGLE_SEARCH, default=True
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_PERSONALIZATION, default=False
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_AFFECTIVE_DIALOG, default=False
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_PROACTIVE_AUDIO, default=False
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show MCP server management menu."""
        if user_input is not None:
            action = user_input.get("action")
            if action == "add_sse":
                return await self.async_step_mcp_add_sse()
            elif action == "add_stdio":
                return await self.async_step_mcp_add_stdio()
            elif action == "finish":
                return self._create_entry()

        # Build server list description
        server_list = ""
        if self._mcp_servers:
            server_list = "\n".join(
                f"  • {s.get(CONF_MCP_SERVER_NAME, 'Unnamed')} ({s.get(CONF_MCP_SERVER_TYPE, 'sse')})"
                for s in self._mcp_servers
            )
        else:
            server_list = "  No servers configured"

        return self.async_show_form(
            step_id="mcp_menu",
            data_schema=vol.Schema(
                {
                    vol.Required("action", default="finish"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "add_sse", "label": "Add SSE Server"},
                                {"value": "add_stdio", "label": "Add Stdio Server"},
                                {"value": "finish", "label": "Finish Setup"},
                            ],
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
            description_placeholders={
                "server_count": str(len(self._mcp_servers)),
                "server_list": server_list,
            },
        )

    async def async_step_personalization_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show a warning/confirmation when personalization is enabled."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # If user confirms, continue; otherwise disable personalization and continue.
            if user_input.get("confirm_personalization"):
                return await self.async_step_mcp_menu()
            # User did not confirm; disable the option and continue
            self._config[CONF_ENABLE_PERSONALIZATION] = False
            return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="personalization_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required("confirm_personalization", default=False): selector.BooleanSelector()
                }
            ),
            description_placeholders={
                "warning": (
                    "Enabling personalization may send additional conversation data to the service. "
                    "Only enable if you understand and accept potential data handling implications."
                )
            },
            errors=errors,
        )

    async def async_step_mcp_add_sse(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add an SSE MCP server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input.get(CONF_MCP_SERVER_URL, "")
            if url and not url.startswith(("http://", "https://")):
                errors[CONF_MCP_SERVER_URL] = "invalid_url"
            elif url:
                self._mcp_servers.append({
                    CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, f"SSE Server {len(self._mcp_servers) + 1}"),
                    CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_SSE,
                    CONF_MCP_SERVER_URL: url,
                    CONF_MCP_SERVER_TOKEN: user_input.get(CONF_MCP_SERVER_TOKEN, ""),
                    CONF_MCP_SERVER_ENABLED: True,
                })
                return await self.async_step_mcp_menu()
            else:
                return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="mcp_add_sse",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MCP_SERVER_NAME): str,
                    vol.Required(CONF_MCP_SERVER_URL): str,
                    vol.Optional(CONF_MCP_SERVER_TOKEN): str,
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_add_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a Stdio MCP server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            command = user_input.get(CONF_MCP_SERVER_COMMAND, "")
            if command:
                # Parse args from comma-separated string
                args_str = user_input.get(CONF_MCP_SERVER_ARGS, "")
                args = [a.strip() for a in args_str.split(",") if a.strip()] if args_str else []

                # Parse env from key=value pairs
                env_str = user_input.get(CONF_MCP_SERVER_ENV, "")
                env = {}
                if env_str:
                    for pair in env_str.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            env[key.strip()] = value.strip()

                self._mcp_servers.append({
                    CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, f"Stdio Server {len(self._mcp_servers) + 1}"),
                    CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_STDIO,
                    CONF_MCP_SERVER_COMMAND: command,
                    CONF_MCP_SERVER_ARGS: args,
                    CONF_MCP_SERVER_ENV: env,
                    CONF_MCP_SERVER_ENABLED: True,
                })
                return await self.async_step_mcp_menu()
            else:
                return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="mcp_add_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MCP_SERVER_NAME): str,
                    vol.Required(CONF_MCP_SERVER_COMMAND): str,
                    vol.Optional(CONF_MCP_SERVER_ARGS, default=""): str,
                    vol.Optional(CONF_MCP_SERVER_ENV, default=""): str,
                }
            ),
            errors=errors,
            description_placeholders={
                "args_hint": "Comma-separated arguments (e.g., --port,8080)",
                "env_hint": "Comma-separated key=value pairs (e.g., API_KEY=xxx,DEBUG=true)",
            },
        )

    def _create_entry(self) -> FlowResult:
        """Create the config entry."""
        data = {
            CONF_API_KEY: self._api_key,
            CONF_MODEL: self._config.get(CONF_MODEL, DEFAULT_MODEL),
            CONF_VOICE: self._config.get(CONF_VOICE, DEFAULT_VOICE),
            CONF_INSTRUCTIONS: self._config.get(CONF_INSTRUCTIONS, DEFAULT_INSTRUCTIONS),
            CONF_TEMPERATURE: self._config.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            CONF_ENABLE_PERSONALIZATION: self._config.get(CONF_ENABLE_PERSONALIZATION, False),
            CONF_ENABLE_GOOGLE_SEARCH: self._config.get(CONF_ENABLE_GOOGLE_SEARCH, True),
            CONF_ENABLE_AFFECTIVE_DIALOG: self._config.get(CONF_ENABLE_AFFECTIVE_DIALOG, False),
            CONF_ENABLE_PROACTIVE_AUDIO: self._config.get(CONF_ENABLE_PROACTIVE_AUDIO, False),
            CONF_MCP_SERVERS: self._mcp_servers,
        }

        return self.async_create_entry(
            title=self._config.get(CONF_NAME, "Gemini Live"),
            data=data,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> GeminiLiveOptionsFlow:
        """Get the options flow for this handler."""
        return GeminiLiveOptionsFlow()


class GeminiLiveOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Gemini Live."""

    def __init__(self) -> None:
        """Initialize options flow."""
        self._mcp_servers: list[dict[str, Any]] = []
        self._editing_server_index: int | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        # Initialize MCP servers from config entry on first call
        if not self._mcp_servers:
            current_config = {**self.config_entry.data, **self.config_entry.options}
            self._mcp_servers = list(
                current_config.get(CONF_MCP_SERVERS, [])
            )

        if user_input is not None:
            # Save main settings and go to MCP menu
            self._main_config = {
                CONF_MODEL: user_input.get(CONF_MODEL, DEFAULT_MODEL),
                CONF_VOICE: user_input.get(CONF_VOICE, DEFAULT_VOICE),
                CONF_INSTRUCTIONS: user_input.get(CONF_INSTRUCTIONS, DEFAULT_INSTRUCTIONS),
                CONF_TEMPERATURE: user_input.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                CONF_ENABLE_GOOGLE_SEARCH: user_input.get(CONF_ENABLE_GOOGLE_SEARCH, True),
                CONF_ENABLE_PERSONALIZATION: user_input.get(CONF_ENABLE_PERSONALIZATION, False),
                CONF_ENABLE_AFFECTIVE_DIALOG: user_input.get(CONF_ENABLE_AFFECTIVE_DIALOG, False),
                CONF_ENABLE_PROACTIVE_AUDIO: user_input.get(CONF_ENABLE_PROACTIVE_AUDIO, False),
            }
            return await self.async_step_mcp_menu()

        current_config = {**self.config_entry.data, **self.config_entry.options}

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL,
                        default=current_config.get(CONF_MODEL, DEFAULT_MODEL),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_VOICE,
                        default=current_config.get(CONF_VOICE, DEFAULT_VOICE),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=VOICES,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_INSTRUCTIONS,
                        default=current_config.get(CONF_INSTRUCTIONS, DEFAULT_INSTRUCTIONS),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=current_config.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=2.0,
                            step=0.1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_ENABLE_GOOGLE_SEARCH,
                        default=current_config.get(CONF_ENABLE_GOOGLE_SEARCH, True),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_PERSONALIZATION,
                        default=current_config.get(CONF_ENABLE_PERSONALIZATION, False),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_AFFECTIVE_DIALOG,
                        default=current_config.get(CONF_ENABLE_AFFECTIVE_DIALOG, False),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_ENABLE_PROACTIVE_AUDIO,
                        default=current_config.get(CONF_ENABLE_PROACTIVE_AUDIO, False),
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show MCP server management menu."""
        if user_input is not None:
            action = user_input.get("action")
            if action == "add_sse":
                return await self.async_step_mcp_add_sse()
            elif action == "add_stdio":
                return await self.async_step_mcp_add_stdio()
            elif action == "manage":
                return await self.async_step_mcp_manage()
            elif action == "finish":
                return self._save_options()

        # Build options based on current servers
        options = [
            {"value": "add_sse", "label": "Add SSE Server"},
            {"value": "add_stdio", "label": "Add Stdio Server"},
        ]

        if self._mcp_servers:
            options.insert(0, {"value": "manage", "label": "Manage Existing Servers"})

        options.append({"value": "finish", "label": "Save & Finish"})

        # Build server list description
        server_list = ""
        if self._mcp_servers:
            server_list = "\n".join(
                f"  • {s.get(CONF_MCP_SERVER_NAME, 'Unnamed')} ({s.get(CONF_MCP_SERVER_TYPE, 'sse')}) {'✓' if s.get(CONF_MCP_SERVER_ENABLED, True) else '✗'}"
                for s in self._mcp_servers
            )
        else:
            server_list = "  No servers configured"

        return self.async_show_form(
            step_id="mcp_menu",
            data_schema=vol.Schema(
                {
                    vol.Required("action", default="finish"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
            description_placeholders={
                "server_count": str(len(self._mcp_servers)),
                "server_list": server_list,
            },
        )

    async def async_step_mcp_add_sse(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add an SSE MCP server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input.get(CONF_MCP_SERVER_URL, "")
            if url and not url.startswith(("http://", "https://")):
                errors[CONF_MCP_SERVER_URL] = "invalid_url"
            elif url:
                self._mcp_servers.append({
                    CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, f"SSE Server {len(self._mcp_servers) + 1}"),
                    CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_SSE,
                    CONF_MCP_SERVER_URL: url,
                    CONF_MCP_SERVER_TOKEN: user_input.get(CONF_MCP_SERVER_TOKEN, ""),
                    CONF_MCP_SERVER_ENABLED: True,
                })
                return await self.async_step_mcp_menu()
            else:
                return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="mcp_add_sse",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MCP_SERVER_NAME): str,
                    vol.Required(CONF_MCP_SERVER_URL): str,
                    vol.Optional(CONF_MCP_SERVER_TOKEN): str,
                }
            ),
            errors=errors,
        )

    async def async_step_mcp_add_stdio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a Stdio MCP server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            command = user_input.get(CONF_MCP_SERVER_COMMAND, "")
            if command:
                # Parse args from comma-separated string
                args_str = user_input.get(CONF_MCP_SERVER_ARGS, "")
                args = [a.strip() for a in args_str.split(",") if a.strip()] if args_str else []

                # Parse env from key=value pairs
                env_str = user_input.get(CONF_MCP_SERVER_ENV, "")
                env = {}
                if env_str:
                    for pair in env_str.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            env[key.strip()] = value.strip()

                self._mcp_servers.append({
                    CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, f"Stdio Server {len(self._mcp_servers) + 1}"),
                    CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_STDIO,
                    CONF_MCP_SERVER_COMMAND: command,
                    CONF_MCP_SERVER_ARGS: args,
                    CONF_MCP_SERVER_ENV: env,
                    CONF_MCP_SERVER_ENABLED: True,
                })
                return await self.async_step_mcp_menu()
            else:
                return await self.async_step_mcp_menu()

        return self.async_show_form(
            step_id="mcp_add_stdio",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MCP_SERVER_NAME): str,
                    vol.Required(CONF_MCP_SERVER_COMMAND): str,
                    vol.Optional(CONF_MCP_SERVER_ARGS, default=""): str,
                    vol.Optional(CONF_MCP_SERVER_ENV, default=""): str,
                }
            ),
            errors=errors,
            description_placeholders={
                "args_hint": "Comma-separated arguments (e.g., --port,8080)",
                "env_hint": "Comma-separated key=value pairs (e.g., API_KEY=xxx,DEBUG=true)",
            },
        )

    async def async_step_mcp_manage(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage existing MCP servers."""
        if user_input is not None:
            action = user_input.get("action")
            if action == "back":
                return await self.async_step_mcp_menu()
            elif action and action.startswith("edit_"):
                try:
                    self._editing_server_index = int(action.split("_")[1])
                    return await self.async_step_mcp_edit()
                except (ValueError, IndexError):
                    pass
            elif action and action.startswith("delete_"):
                try:
                    idx = int(action.split("_")[1])
                    if 0 <= idx < len(self._mcp_servers):
                        del self._mcp_servers[idx]
                except (ValueError, IndexError):
                    pass
            elif action and action.startswith("toggle_"):
                try:
                    idx = int(action.split("_")[1])
                    if 0 <= idx < len(self._mcp_servers):
                        current = self._mcp_servers[idx].get(CONF_MCP_SERVER_ENABLED, True)
                        self._mcp_servers[idx][CONF_MCP_SERVER_ENABLED] = not current
                except (ValueError, IndexError):
                    pass

        # Build server action options
        options = []
        for i, server in enumerate(self._mcp_servers):
            name = server.get(CONF_MCP_SERVER_NAME, f"Server {i + 1}")
            enabled = server.get(CONF_MCP_SERVER_ENABLED, True)

            options.append({"value": f"edit_{i}", "label": f"✏️ Edit: {name}"})
            options.append({"value": f"toggle_{i}", "label": f"{'🔴 Disable' if enabled else '🟢 Enable'}: {name}"})
            options.append({"value": f"delete_{i}", "label": f"🗑️ Delete: {name}"})

        options.append({"value": "back", "label": "← Back to Menu"})

        return self.async_show_form(
            step_id="mcp_manage",
            data_schema=vol.Schema(
                {
                    vol.Required("action", default="back"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_mcp_edit(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Edit an existing MCP server."""
        errors: dict[str, str] = {}

        if self._editing_server_index is None or self._editing_server_index >= len(self._mcp_servers):
            return await self.async_step_mcp_manage()

        server = self._mcp_servers[self._editing_server_index]
        server_type = server.get(CONF_MCP_SERVER_TYPE, MCP_SERVER_TYPE_SSE)

        if user_input is not None:
            if server_type == MCP_SERVER_TYPE_SSE:
                url = user_input.get(CONF_MCP_SERVER_URL, "")
                if url and not url.startswith(("http://", "https://")):
                    errors[CONF_MCP_SERVER_URL] = "invalid_url"
                else:
                    self._mcp_servers[self._editing_server_index] = {
                        CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, server.get(CONF_MCP_SERVER_NAME)),
                        CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_SSE,
                        CONF_MCP_SERVER_URL: url,
                        CONF_MCP_SERVER_TOKEN: user_input.get(CONF_MCP_SERVER_TOKEN, ""),
                        CONF_MCP_SERVER_ENABLED: server.get(CONF_MCP_SERVER_ENABLED, True),
                    }
                    self._editing_server_index = None
                    return await self.async_step_mcp_manage()
            else:
                # Parse args from comma-separated string
                args_str = user_input.get(CONF_MCP_SERVER_ARGS, "")
                args = [a.strip() for a in args_str.split(",") if a.strip()] if args_str else []

                # Parse env from key=value pairs
                env_str = user_input.get(CONF_MCP_SERVER_ENV, "")
                env = {}
                if env_str:
                    for pair in env_str.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            env[key.strip()] = value.strip()

                self._mcp_servers[self._editing_server_index] = {
                    CONF_MCP_SERVER_NAME: user_input.get(CONF_MCP_SERVER_NAME, server.get(CONF_MCP_SERVER_NAME)),
                    CONF_MCP_SERVER_TYPE: MCP_SERVER_TYPE_STDIO,
                    CONF_MCP_SERVER_COMMAND: user_input.get(CONF_MCP_SERVER_COMMAND, ""),
                    CONF_MCP_SERVER_ARGS: args,
                    CONF_MCP_SERVER_ENV: env,
                    CONF_MCP_SERVER_ENABLED: server.get(CONF_MCP_SERVER_ENABLED, True),
                }
                self._editing_server_index = None
                return await self.async_step_mcp_manage()

        if server_type == MCP_SERVER_TYPE_SSE:
            return self.async_show_form(
                step_id="mcp_edit",
                data_schema=vol.Schema(
                    {
                        vol.Required(CONF_MCP_SERVER_NAME, default=server.get(CONF_MCP_SERVER_NAME, "")): str,
                        vol.Required(CONF_MCP_SERVER_URL, default=server.get(CONF_MCP_SERVER_URL, "")): str,
                        vol.Optional(CONF_MCP_SERVER_TOKEN, default=server.get(CONF_MCP_SERVER_TOKEN, "")): str,
                    }
                ),
                errors=errors,
            )
        else:
            # Format args and env for editing
            args = server.get(CONF_MCP_SERVER_ARGS, [])
            args_str = ",".join(args) if isinstance(args, list) else str(args)

            env = server.get(CONF_MCP_SERVER_ENV, {})
            env_str = ",".join(f"{k}={v}" for k, v in env.items()) if isinstance(env, dict) else str(env)

            return self.async_show_form(
                step_id="mcp_edit",
                data_schema=vol.Schema(
                    {
                        vol.Required(CONF_MCP_SERVER_NAME, default=server.get(CONF_MCP_SERVER_NAME, "")): str,
                        vol.Required(CONF_MCP_SERVER_COMMAND, default=server.get(CONF_MCP_SERVER_COMMAND, "")): str,
                        vol.Optional(CONF_MCP_SERVER_ARGS, default=args_str): str,
                        vol.Optional(CONF_MCP_SERVER_ENV, default=env_str): str,
                    }
                ),
                errors=errors,
                description_placeholders={
                    "args_hint": "Comma-separated arguments",
                    "env_hint": "Comma-separated key=value pairs",
                },
            )

    def _save_options(self) -> FlowResult:
        """Save the options."""
        new_data = {**self.config_entry.data}
        new_data[CONF_MCP_SERVERS] = self._mcp_servers
        
        self.hass.config_entries.async_update_entry(
            self.config_entry,
            data=new_data,
        )
        
        return self.async_create_entry(
            title="",
            data={
                **getattr(self, "_main_config", {}),
            },
        )
