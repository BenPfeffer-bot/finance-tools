import yaml

def load_config(config_path: str = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/config/config.yaml") -> dict:
    """
    Loads the configuration from the given path.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
