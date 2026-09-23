"""Translate config dictionaries into arguments for Miles and SGLang.

Those backends expose argparse parsers. Append configured values after their
preset arguments so argparse itself handles types, choices and required options.
Boolean flags need defaults because a store_true flag cannot express False.
"""

import argparse


def apply_config_overrides(parser, options, argv):
    """Mutate argv/defaults before the backend calls parse_args."""
    boolean_actions = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse.BooleanOptionalAction,
    )
    actions_by_name = {}
    for action in parser._actions:
        actions_by_name.setdefault(action.dest, []).append(action)

    for name, value in options.items():
        actions = actions_by_name.get(name, [])
        if not actions or not all(action.option_strings for action in actions):
            raise ValueError(f"unknown backend option: {name}")
        if value is None:
            raise ValueError(f"backend option {name} cannot be null")
        if all(isinstance(action, boolean_actions) for action in actions):
            if not isinstance(value, bool):
                raise ValueError(f"backend option {name} requires a boolean")
            flags = {flag for action in actions for flag in action.option_strings}
            argv[:] = [arg for arg in argv if arg.split("=", 1)[0] not in flags]
            for action in actions:
                action.required = False
            parser.set_defaults(**{name: value})
            continue
        if len(actions) != 1 or type(actions[0]) is not argparse._StoreAction:
            raise ValueError(
                f"backend option {name} uses an unsupported argparse action"
            )
        action = actions[0]
        multiple = action.nargs in ("+", "*") or isinstance(action.nargs, int)
        flag = action.option_strings[0]
        if multiple:
            if not isinstance(value, list):
                raise ValueError(f"backend option {name} requires a list")
            argv.extend([flag, *(str(item) for item in value)])
        else:
            if isinstance(value, (dict, list, bool)):
                raise ValueError(f"backend option {name} requires a scalar")
            argv.append(f"{flag}={value}")
