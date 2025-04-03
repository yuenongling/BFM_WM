import toml
import argparse

def parse_toml_config(config_file):
    config = toml.load(config_file)
    config['config_file'] = config_file

    return config
