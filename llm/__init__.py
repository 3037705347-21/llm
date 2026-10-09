import datetime
import inspect
import json
import os
import pathlib
import struct
from collections.abc import Callable
from typing import Any

import click

from .embeddings import Collection
from .errors import (
    AliasHistoryError,
    ConversationNotSupported,
    ModelError,
    NeedsKeyException,
)
from .hookspecs import hookimpl
from .models import (
    AsyncConversation,
    AsyncKeyModel,
    AsyncModel,
    AsyncResponse,
    Attachment,
    CancelToolCall,
    Conversation,
    EmbeddingModel,
    EmbeddingModelWithAliases,
    KeyModel,
    Model,
    ModelWithAliases,
    Options,
    PauseChain,
    Prompt,
    Response,
    ServerSideTool,
    Tool,
    Toolbox,
    ToolCall,
    ToolOutput,
    ToolResult,
    Usage,
)
from .parts import (
    Message,
    assistant,
    system,
    tool_message,
    user,
)
from .plugins import load_plugins, pm
from .templates import Template
from .utils import Fragment, schema_dsl

__all__ = [
    "AliasHistoryError",
    "AsyncConversation",
    "AsyncKeyModel",
    "AsyncModel",
    "AsyncResponse",
    "Attachment",
    "CancelToolCall",
    "Collection",
    "Conversation",
    "ConversationNotSupported",
    "Fragment",
    "KeyModel",
    "Message",
    "Model",
    "ModelError",
    "NeedsKeyException",
    "Options",
    "PauseChain",
    "Prompt",
    "Response",
    "ServerSideTool",
    "Template",
    "Tool",
    "ToolCall",
    "ToolOutput",
    "ToolResult",
    "Toolbox",
    "Usage",
    "alias_rotations_path",
    "assistant",
    "get_async_model",
    "get_key",
    "get_model",
    "hookimpl",
    "load_alias_rotations",
    "load_aliases",
    "record_alias_rotation",
    "schema_dsl",
    "system",
    "tool_message",
    "user",
    "user_dir",
]
DEFAULT_MODEL = "gpt-5.6-luna"


def get_plugins(all=False):
    plugins = []
    plugin_to_distinfo = dict(pm.list_plugin_distinfo())
    for plugin in pm.get_plugins():
        if not all and plugin.__name__.startswith("llm.default_plugins."):
            continue
        plugin_info = {
            "name": plugin.__name__,
            "hooks": [h.name for h in pm.get_hookcallers(plugin)],
        }
        distinfo = plugin_to_distinfo.get(plugin)
        if distinfo:
            plugin_info["version"] = distinfo.version
            plugin_info["name"] = (
                getattr(distinfo, "name", None) or distinfo.project_name
            )
        plugins.append(plugin_info)
    return plugins


def get_models_with_aliases() -> list["ModelWithAliases"]:
    model_aliases = []

    # Include aliases from aliases.json
    aliases_path = user_dir() / "aliases.json"
    extra_model_aliases: dict[str, list] = {}
    if aliases_path.exists():
        configured_aliases = json.loads(aliases_path.read_text())
        for alias, model_id in configured_aliases.items():
            extra_model_aliases.setdefault(model_id, []).append(alias)

    def register(model, async_model=None, aliases=None):
        alias_list = list(aliases or [])
        if model.model_id in extra_model_aliases:
            alias_list.extend(extra_model_aliases[model.model_id])
        model_aliases.append(ModelWithAliases(model, async_model, alias_list))

    load_plugins()
    pm.hook.register_models(register=register, model_aliases=model_aliases)

    return model_aliases


def _get_loaders(hook_method) -> dict[str, Callable]:
    load_plugins()
    loaders = {}

    def register(prefix, loader):
        suffix = 0
        prefix_to_try = prefix
        while prefix_to_try in loaders:
            suffix += 1
            prefix_to_try = f"{prefix}_{suffix}"
        loaders[prefix_to_try] = loader

    hook_method(register=register)
    return loaders


def get_template_loaders() -> dict[str, Callable[[str], Template]]:
    """Get template loaders registered by plugins."""
    return _get_loaders(pm.hook.register_template_loaders)


def get_fragment_loaders() -> dict[
    str,
    Callable[[str], Fragment | Attachment | list[Fragment | Attachment]],
]:
    """Get fragment loaders registered by plugins."""
    return _get_loaders(pm.hook.register_fragment_loaders)


def get_tools() -> dict[str, Tool | type[Toolbox]]:
    """Return all tools (llm.Tool and llm.Toolbox) registered by plugins."""
    load_plugins()
    tools: dict[str, Tool | type[Toolbox]] = {}

    # Variable to track current plugin name
    current_plugin_name = None

    def register(
        tool_or_function: Tool | type[Toolbox] | Callable[..., Any],
        name: str | None = None,
    ) -> None:
        tool: Tool | type[Toolbox] | None = None

        # If it's a Toolbox class, set the plugin field on it
        if inspect.isclass(tool_or_function):
            if issubclass(tool_or_function, Toolbox):
                tool = tool_or_function
                if current_plugin_name:
                    tool.plugin = current_plugin_name
                tool.name = name or tool.__name__
            else:
                raise TypeError(
                    f"Toolbox classes must inherit from llm.Toolbox, {tool_or_function.__name__} does not."
                )

        # If it's already a Tool instance, use it directly
        elif isinstance(tool_or_function, Tool):
            tool = tool_or_function
            if name:
                tool.name = name
            if current_plugin_name:
                tool.plugin = current_plugin_name

        # If it's a bare function, wrap it in a Tool
        else:
            tool = Tool.function(tool_or_function, name=name)
            if current_plugin_name:
                tool.plugin = current_plugin_name

        # Get the name for the tool/toolbox
        if tool:
            # For Toolbox classes, use their name attribute or class name
            if inspect.isclass(tool) and issubclass(tool, Toolbox):
                prefix = name or getattr(tool, "name", tool.__name__) or ""
            else:
                prefix = name or tool.name or ""

            suffix = 0
            candidate = prefix

            # Avoid name collisions
            while candidate in tools:
                suffix += 1
                candidate = f"{prefix}_{suffix}"

            tools[candidate] = tool

    # Call each plugin's register_tools hook individually to track current_plugin_name
    for plugin in pm.get_plugins():
        current_plugin_name = pm.get_name(plugin)
        hook_caller = pm.hook.register_tools
        plugin_impls = [
            impl for impl in hook_caller.get_hookimpls() if impl.plugin is plugin
        ]
        for impl in plugin_impls:
            impl.function(register=register)

    return tools


def get_embedding_models_with_aliases() -> list["EmbeddingModelWithAliases"]:
    model_aliases = []

    # Include aliases from aliases.json
    aliases_path = user_dir() / "aliases.json"
    extra_model_aliases: dict[str, list] = {}
    if aliases_path.exists():
        configured_aliases = json.loads(aliases_path.read_text())
        for alias, model_id in configured_aliases.items():
            extra_model_aliases.setdefault(model_id, []).append(alias)

    def register(model, aliases=None):
        alias_list = list(aliases or [])
        if model.model_id in extra_model_aliases:
            alias_list.extend(extra_model_aliases[model.model_id])
        model_aliases.append(EmbeddingModelWithAliases(model, alias_list))

    load_plugins()
    pm.hook.register_embedding_models(register=register)

    return model_aliases


def get_embedding_models():
    models = []

    def register(model, aliases=None):
        models.append(model)

    load_plugins()
    pm.hook.register_embedding_models(register=register)
    return models


def get_embedding_model(name):
    aliases = get_embedding_model_aliases()
    try:
        return aliases[name]
    except KeyError:
        raise UnknownModelError("Unknown model: " + str(name))


def get_embedding_model_aliases() -> dict[str, EmbeddingModel]:
    model_aliases = {}
    for model_with_aliases in get_embedding_models_with_aliases():
        for alias in model_with_aliases.aliases:
            model_aliases[alias] = model_with_aliases.model
        model_aliases[model_with_aliases.model.model_id] = model_with_aliases.model
    return model_aliases


def get_async_model_aliases() -> dict[str, AsyncModel]:
    async_model_aliases = {}
    for model_with_aliases in get_models_with_aliases():
        if model_with_aliases.async_model:
            for alias in model_with_aliases.aliases:
                async_model_aliases[alias] = model_with_aliases.async_model
            async_model_aliases[model_with_aliases.model.model_id] = (
                model_with_aliases.async_model
            )
    return async_model_aliases


def get_model_aliases() -> dict[str, Model]:
    model_aliases = {}
    for model_with_aliases in get_models_with_aliases():
        if model_with_aliases.model:
            for alias in model_with_aliases.aliases:
                model_aliases[alias] = model_with_aliases.model
            model_aliases[model_with_aliases.model.model_id] = model_with_aliases.model
    return model_aliases


class UnknownModelError(KeyError):
    pass


def get_models() -> list[Model]:
    "Get all registered models"
    models_with_aliases = get_models_with_aliases()
    return [mwa.model for mwa in models_with_aliases if mwa.model]


def get_async_models() -> list[AsyncModel]:
    "Get all registered async models"
    models_with_aliases = get_models_with_aliases()
    return [mwa.async_model for mwa in models_with_aliases if mwa.async_model]


def get_async_model(name: str | None = None) -> AsyncModel:
    "Get an async model by name or alias"
    aliases = get_async_model_aliases()
    name = name or get_default_model()
    try:
        return aliases[name]
    except KeyError:
        # Does a sync model exist?
        sync_model = None
        try:
            sync_model = get_model(name, _skip_async=True)
        except UnknownModelError:
            pass
        if sync_model:
            raise UnknownModelError("Unknown async model (sync model exists): " + name)
        else:
            raise UnknownModelError("Unknown model: " + name)


def get_model(name: str | None = None, _skip_async: bool = False) -> Model:
    "Get a model by name or alias"
    aliases = get_model_aliases()
    name = name or get_default_model()
    try:
        return aliases[name]
    except KeyError:
        # Does an async model exist?
        if _skip_async:
            raise UnknownModelError("Unknown model: " + name)
        async_model = None
        try:
            async_model = get_async_model(name)
        except UnknownModelError:
            pass
        if async_model:
            raise UnknownModelError("Unknown model (async model exists): " + name)
        else:
            raise UnknownModelError("Unknown model: " + name)


def get_key(
    explicit_key: str | None = None,
    key_alias: str | None = None,
    env_var: str | None = None,
    *,
    alias: str | None = None,
    env: str | None = None,
    input: str | None = None,
) -> str | None:
    """
    Return an API key based on a hierarchy of potential sources. You should use the keyword arguments,
    the positional arguments are here purely for backwards-compatibility with older code.

    :param input: Input provided by the user. This may be the key, or an alias of a key in keys.json.
    :param alias: The alias used to retrieve the key from the keys.json file.
    :param env: Name of the environment variable to check for the key as a final fallback.
    """
    if alias:
        key_alias = alias
    if env:
        env_var = env
    if input:
        explicit_key = input
    stored_keys = load_keys()
    # If user specified an alias, use the key stored for that alias
    if explicit_key in stored_keys:
        return stored_keys[explicit_key]
    if explicit_key:
        # User specified a key that's not an alias, use that
        return explicit_key
    # Stored key over-rides environment variables over-ride the default key
    if key_alias in stored_keys:
        return stored_keys[key_alias]
    # Finally try environment variable
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    # Couldn't find it
    return None


def load_keys():
    path = user_dir() / "keys.json"
    if path.exists():
        return json.loads(path.read_text())
    else:
        return {}


def user_dir():
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path


def aliases_path():
    "Path to the aliases.json file"
    return user_dir() / "aliases.json"


def alias_rotations_path():
    """
    Path to alias_rotations.json, which records every time an alias was
    switched from one model to another. The backfill uses it to tell
    which concrete model an alias referred to at the time a log row was
    written.
    """
    return user_dir() / "alias_rotations.json"


def _atomic_write_json(path, data):
    "Write JSON via a temp file + rename, so readers never see a torn file."
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=4) + "\n", "utf-8")
    os.replace(temporary, path)


def load_aliases():
    """
    Read the current {alias: model_id} mapping from aliases.json.

    Returns {} if the file does not exist. Raises AliasHistoryError if
    the file exists but cannot be read or parsed, so callers that are
    about to derive data from it never do so from a broken file.
    """
    path = aliases_path()
    if not path.exists():
        return {}
    try:
        current = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as ex:
        raise AliasHistoryError(f"Could not read aliases file {path}: {ex}") from ex
    if not isinstance(current, dict):
        raise AliasHistoryError(
            f"Aliases file {path} must contain a JSON object of alias to model"
        )
    return current


def load_alias_rotations():
    """
    Read alias_rotations.json as {alias: [{old, new, datetime}, ...]}.

    Entries are ordered oldest-first. Returns {} if the file does not
    exist. Raises AliasHistoryError if the file cannot be read or does
    not have the expected shape.
    """
    path = alias_rotations_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as ex:
        raise AliasHistoryError(
            f"Could not read alias rotation history {path}: {ex}"
        ) from ex
    message = (
        f"Alias rotation history {path} must be a JSON object mapping alias"
        " names to lists of {{old, new, datetime}} entries"
    )
    if not isinstance(raw, dict):
        raise AliasHistoryError(message)
    rotations = {}
    for alias, entries in raw.items():
        if not isinstance(entries, list):
            raise AliasHistoryError(message)
        for entry in entries:
            if not isinstance(entry, dict):
                raise AliasHistoryError(message)
            try:
                old = entry["old"]
                new = entry["new"]
                rotated_at = entry["datetime"]
            except (KeyError, TypeError) as ex:
                raise AliasHistoryError(message) from ex
            if not all(isinstance(v, str) for v in (old, new, rotated_at)):
                raise AliasHistoryError(message)
        rotations[alias] = sorted(entries, key=lambda entry: entry["datetime"])
    return rotations


def record_alias_rotation(alias, old_model, new_model, rotated_at=None):
    """
    Append a rotation of alias from old_model to new_model.

    Raises AliasHistoryError if the existing rotation history cannot be
    read: in that case nothing is written, so an unreadable history is
    never overwritten.
    """
    if rotated_at is None:
        rotated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rotations = load_alias_rotations()
    rotations.setdefault(alias, []).append(
        {"old": old_model, "new": new_model, "datetime": rotated_at}
    )
    rotations[alias].sort(key=lambda entry: entry["datetime"])
    _atomic_write_json(alias_rotations_path(), rotations)


def set_alias(alias, model_id_or_alias):
    """
    Set an alias to point to the specified model.

    If the alias previously pointed at a different model, the switch is
    appended to alias_rotations.json before aliases.json is updated, so
    the rotation can never be lost. Returns the model_id the alias
    pointed at before, or None if the alias was new.

    Raises AliasHistoryError if aliases.json or the rotation history
    cannot be read; in that case neither file is modified.
    """
    current = load_aliases()
    previous = current.get(alias)
    # Resolve model_id_or_alias to a model_id
    try:
        model = get_model(model_id_or_alias)
        model_id = model.model_id
    except UnknownModelError:
        # Try to resolve it to an embedding model
        try:
            model = get_embedding_model(model_id_or_alias)
            model_id = model.model_id
        except UnknownModelError:
            # Set the alias to the exact string they provided instead
            model_id = model_id_or_alias
    if previous is not None and previous != model_id:
        # Record the rotation before flipping aliases.json: a rotation
        # without the file flip is harmless (the backfill's pre-rotation
        # window still resolves the old rows), but a file flip without a
        # rotation loses the only evidence of what the old model was.
        record_alias_rotation(alias, previous, model_id)
    current[alias] = model_id
    _atomic_write_json(aliases_path(), current)
    return previous


def remove_alias(alias):
    """
    Remove an alias.
    """
    path = aliases_path()
    if not path.exists():
        raise KeyError("No aliases.json file exists")
    try:
        current = json.loads(path.read_text("utf-8"))
    except json.decoder.JSONDecodeError:
        raise KeyError("aliases.json file is not valid JSON")
    if alias not in current:
        raise KeyError(f"No such alias: {alias}")
    del current[alias]
    _atomic_write_json(path, current)


def encode(values):
    return struct.pack("<" + "f" * len(values), *values)


def decode(binary):
    return struct.unpack("<" + "f" * (len(binary) // 4), binary)


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)


def get_default_model(filename="default_model.txt", default=DEFAULT_MODEL):
    path = user_dir() / filename
    if path.exists():
        return path.read_text().strip()
    else:
        return default


def set_default_model(model, filename="default_model.txt"):
    path = user_dir() / filename
    if model is None and path.exists():
        path.unlink()
    else:
        path.write_text(model)


def get_default_embedding_model():
    return get_default_model("default_embedding_model.txt", None)


def set_default_embedding_model(model):
    set_default_model(model, "default_embedding_model.txt")
