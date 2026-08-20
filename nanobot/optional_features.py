"""Deprecated shim for :mod:`nanobot.runtime.features`.

Moved to ``nanobot/runtime/features.py``. This module exists only so existing
imports keep working during the transition; it will be removed in a later
release.
"""

from nanobot.runtime.features import *  # noqa: F401,F403
from nanobot.runtime.features import (  # noqa: F401
    InstallResult,
    OptionalFeatureError,
    _channel_config_snapshot,
    channel_configured,
    channel_enabled,
    command_text,
    disable_optional_feature,
    enable_optional_feature,
    ensure_enabled_channel_dependencies,
    extra_installed,
    install_args_for_extra,
    install_extra,
    load_pyproject,
    missing_pip,
    optional_dependency_groups,
    optional_dependency_groups_from_metadata,
    optional_features_payload,
    read_config_data,
    requirement_installed,
    run_install_command,
    set_channel_config_enabled,
    with_channel_runtime_status,
    write_config_data,
)
