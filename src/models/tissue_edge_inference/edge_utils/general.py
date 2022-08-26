""" General utility functions used in edge classification """


def inv_dictionary(dict_: dict) -> dict:
    """
    Inverts dictionary mapping. Each value in inverted dictionary is a list.

    Args:
        dict_: dictionary to be inverted

    Returns:
        inv_map: dictionary with inverse mapping from `dict_`
    """
    inv_map = {}
    for k, v in dict_.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    return inv_map
