import yaml


def parse_config(config_path):
    config_f = open(config_path)
    config = yaml.safe_load(config_f)
    config_f.close()
    return config