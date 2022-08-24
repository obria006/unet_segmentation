import yaml

def print_dict(dict_):
    """ prints keys and values of dict_ on new line """
    print(yaml.dump(dict_, sort_keys=False,default_flow_style=False))

def save_yaml(dict_:dict, filepath:str):
    """
    Saves dictionary to filepath
    
    Args:
        dict_: Dictionary to be saved
        filepath: Path to save dictionary to
    """
    # write dictionary to file
    assert filepath.endswith('.yaml'), "filepath must be .yaml"
    with open(filepath, 'w') as f:
        yaml.dump(dict_, f)

def load_yaml(filepath:str):
    """
    Loads dictionary from filepath
    
    Args:
        filepath: Path to load dictionary from
    """
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)