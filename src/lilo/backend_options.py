"""Helpers shared by backend-owned deployment configuration readers."""


def native_options(values, protected):
    if not isinstance(values, dict):
        raise ValueError("backend configuration must be a mapping")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"native option must use its underscore name: {key}")
        if key in protected:
            raise ValueError(f"option {key} is managed by Lilo")
        result[key] = value
    return result
