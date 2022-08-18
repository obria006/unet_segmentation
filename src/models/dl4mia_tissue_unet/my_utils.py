def print_dict(dict_):
    for key, val in dict_.items():
        if isinstance(val, dict):
            print_dict(val)  
        else:
            print(f"\t{key}: {val}")