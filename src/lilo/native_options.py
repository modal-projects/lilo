"""Apply typed YAML values through a backend's own argparse schema."""

from __future__ import annotations

import argparse


def apply_defaults(
    parser: argparse.ArgumentParser, options: dict, argv: list[str]
) -> None:
    """Override preset arguments after parser construction, before backend validation.

    Values use argparse destination names. Ordinary scalar/list options and boolean
    flags are supported. Custom argparse actions fail explicitly instead of being
    bypassed by set_defaults.
    """
    by_name = {}
    for action in parser._actions:
        by_name.setdefault(action.dest, []).append(action)
    flags, defaults = {}, {}
    booleans = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse.BooleanOptionalAction,
    )
    for key, value in options.items():
        actions = by_name.get(key, [])
        if not actions or not all(action.option_strings for action in actions):
            raise ValueError(f"unknown backend option: {key}")
        if value is None:
            raise ValueError(f"backend option {key} cannot be null")
        if all(isinstance(action, booleans) for action in actions):
            if not isinstance(value, bool):
                raise ValueError(f"backend option {key} requires a boolean")
        else:
            if len(actions) != 1 or type(actions[0]) is not argparse._StoreAction:
                raise ValueError(
                    f"backend option {key} uses an unsupported argparse action"
                )
            action = actions[0]
            multiple = action.nargs in ("+", "*") or isinstance(action.nargs, int)
            if multiple and not isinstance(value, list):
                raise ValueError(f"backend option {key} requires a list")
            if not multiple and isinstance(value, (dict, list, bool)):
                raise ValueError(f"backend option {key} requires a scalar")
            values = value if multiple else [value]
            if action.nargs == "+" and not values:
                raise ValueError(f"backend option {key} requires a nonempty list")
            if isinstance(action.nargs, int) and len(values) != action.nargs:
                raise ValueError(f"backend option {key} requires {action.nargs} values")
            cast = action.type or (lambda x: x)
            try:
                values = [cast(v) for v in values]
            except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
                raise ValueError(
                    f"invalid value for backend option {key}: {exc}"
                ) from exc
            if action.choices is not None and any(
                v not in action.choices for v in values
            ):
                raise ValueError(f"invalid choice for backend option {key}")
            value = values if multiple else values[0]
        defaults[key] = value
        for action in actions:
            action.required = False
            for flag in action.option_strings:
                flags[flag] = action
    known = {flag for action in parser._actions for flag in action.option_strings}
    kept, i = [], 0
    while i < len(argv):
        token = argv[i]
        action = flags.get(token.split("=", 1)[0])
        if action is None:
            kept.append(token)
            i += 1
            continue
        i += 1
        if "=" in token or action.nargs == 0:
            continue
        remaining = action.nargs if isinstance(action.nargs, int) else 1
        while i < len(argv):
            if argv[i].split("=", 1)[0] in known or argv[i].startswith("--"):
                break
            i += 1
            if action.nargs not in ("+", "*"):
                remaining -= 1
                if remaining == 0:
                    break
    argv[:] = kept
    parser.set_defaults(**defaults)
