from argparse import Namespace
from dataclasses import asdict, is_dataclass
from math import ceil
from typing import Any, Dict, MutableMapping


def format_county_name(name: str):
    return "-".join(s.lower().capitalize() for s in name.split("-"))

def flatten_dict(
    params: MutableMapping[Any, Any], delimiter: str = "/", parent_key: str = ""
) -> dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}

    """
    result: Dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if is_dataclass(v):
            v = asdict(v)
        elif isinstance(v, Namespace):
            v = vars(v)

        if isinstance(v, MutableMapping):
            result = {
                **result,
                **flatten_dict(v, parent_key=new_key, delimiter=delimiter),
            }
        else:
            result[new_key] = v
    return result
