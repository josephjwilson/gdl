"""
YAML loading + CLI merging for Graphormer configs.

Precedence: dataclass defaults < YAML file < explicit CLI args.

The key insight is using argparse's SUPPRESS default to distinguish
"user passed --lr 1e-3" from "user didn't pass --lr" — only explicitly
set CLI args override YAML values.
"""

import argparse
import yaml
from dataclasses import fields, asdict
from typing import Any, Dict, Optional, Tuple

from config.schema import ModelConfig, TrainConfig, AnalysisConfig


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning empty dict on missing/empty file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _overlay_dict_on_dataclass(dc_instance, overrides: dict):
    """Update a dataclass instance's fields from a dict (in-place via __dict__)."""
    valid_fields = {f.name for f in fields(dc_instance)}
    for k, v in overrides.items():
        if k in valid_fields:
            object.__setattr__(dc_instance, k, v)


def _get_explicit_cli_args(parser: argparse.ArgumentParser, argv=None) -> dict:
    """Re-parse CLI with SUPPRESS defaults to find which args were explicitly set.

    Returns a dict of only the args the user actually typed on the command line.
    """
    # Build a shadow parser with SUPPRESS defaults
    shadow = argparse.ArgumentParser(add_help=False)
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        kwargs = {}
        # Copy essential properties
        if action.type is not None:
            kwargs["type"] = action.type
        if action.nargs is not None:
            kwargs["nargs"] = action.nargs
        if action.choices is not None:
            kwargs["choices"] = action.choices
        kwargs["default"] = argparse.SUPPRESS

        if isinstance(action, argparse._StoreTrueAction):
            kwargs.pop("type", None)
            kwargs.pop("nargs", None)
            shadow.add_argument(*action.option_strings, action="store_true", default=argparse.SUPPRESS)
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs.pop("type", None)
            kwargs.pop("nargs", None)
            shadow.add_argument(*action.option_strings, action="store_false", default=argparse.SUPPRESS)
        else:
            shadow.add_argument(*action.option_strings, **kwargs)

    ns, _ = shadow.parse_known_args(argv)
    return vars(ns)


# ── Convenience alias mappings ──────────────────────────────────────────

# CLI arg name -> (config section, field name)
_TRAIN_CLI_TO_MODEL = {
    "num_layers": "encoder_layers",
    "num_heads": "encoder_attention_heads",
}

_ANALYSIS_CLI_TO_MODEL = {
    "num_layers": "encoder_layers",
    "num_heads": "encoder_attention_heads",
}

# Boolean inversions: CLI flag name -> (config section, field name, invert)
_TRAIN_CLI_INVERSIONS = {
    "no_ffn": ("model", "use_ffn"),  # --no_ffn → model.use_ffn = False
    "no_virtual_distance": ("model", "use_virtual_distance"),  # --no_virtual_distance → model.use_virtual_distance = False
    "no_spd_bias": ("model", "use_spd_bias"),  # --no_spd_bias → model.use_spd_bias = False
}

_ANALYSIS_CLI_INVERSIONS = {
    "no_ffn": ("model", "use_ffn"),
}


def _apply_aliases(explicit: dict, model_cfg: ModelConfig, other_cfg, cli_to_model: dict, inversions: dict):
    """Apply convenience aliases from CLI args to config objects."""
    for cli_name, model_field in cli_to_model.items():
        if cli_name in explicit:
            setattr(model_cfg, model_field, explicit.pop(cli_name))

    for cli_name, (section, field_name) in inversions.items():
        if cli_name in explicit:
            value = not explicit.pop(cli_name)
            if section == "model":
                setattr(model_cfg, field_name, value)
            else:
                setattr(other_cfg, field_name, value)


def load_train_config(
    parser: argparse.ArgumentParser,
    argv=None,
) -> Tuple[ModelConfig, TrainConfig]:
    """Load config for training: YAML + CLI merge.

    Returns (model_cfg, train_cfg).
    """
    # First pass: normal parse to get --config path and all defaults
    args = parser.parse_args(argv)

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Layer 1: YAML (if provided)
    yaml_path = getattr(args, "config", None)
    if yaml_path:
        data = _load_yaml(yaml_path)
        if "model" in data:
            _overlay_dict_on_dataclass(model_cfg, data["model"])
        if "train" in data:
            _overlay_dict_on_dataclass(train_cfg, data["train"])

    # Layer 2: Explicit CLI args (highest priority)
    explicit = _get_explicit_cli_args(parser, argv)
    explicit.pop("config", None)  # Already handled

    # Apply convenience aliases before overlaying
    _apply_aliases(explicit, model_cfg, train_cfg, _TRAIN_CLI_TO_MODEL, _TRAIN_CLI_INVERSIONS)

    # Overlay remaining explicit args on train_cfg
    _overlay_dict_on_dataclass(train_cfg, explicit)

    # Sync model-level fields that also appear as direct CLI flags
    if "fixed_spd_bias" in explicit:
        model_cfg.fixed_spd_bias = explicit["fixed_spd_bias"]
    if "causal_mask" in explicit:
        model_cfg.causal_mask = explicit["causal_mask"]
    if "pre_layernorm" in explicit:
        model_cfg.pre_layernorm = explicit["pre_layernorm"]

    return model_cfg, train_cfg


def load_analysis_config(
    parser: argparse.ArgumentParser,
    argv=None,
) -> Tuple[ModelConfig, AnalysisConfig]:
    """Load config for analysis: YAML + CLI merge.

    Returns (model_cfg, analysis_cfg).
    """
    args = parser.parse_args(argv)

    model_cfg = ModelConfig()
    analysis_cfg = AnalysisConfig()

    # Layer 1: YAML
    yaml_path = getattr(args, "config", None)
    if yaml_path:
        data = _load_yaml(yaml_path)
        if "model" in data:
            _overlay_dict_on_dataclass(model_cfg, data["model"])
        if "analysis" in data:
            _overlay_dict_on_dataclass(analysis_cfg, data["analysis"])

    # Layer 2: Explicit CLI args
    explicit = _get_explicit_cli_args(parser, argv)
    explicit.pop("config", None)

    _apply_aliases(explicit, model_cfg, analysis_cfg, _ANALYSIS_CLI_TO_MODEL, _ANALYSIS_CLI_INVERSIONS)

    # Overlay remaining explicit args on analysis_cfg
    _overlay_dict_on_dataclass(analysis_cfg, explicit)

    # Sync model-level fields
    if "fixed_spd_bias" in explicit:
        model_cfg.fixed_spd_bias = explicit["fixed_spd_bias"]

    return model_cfg, analysis_cfg
