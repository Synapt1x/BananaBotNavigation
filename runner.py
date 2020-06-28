# -*- coding: utf-8 -*-
"""
This code is the main runner CLI for the relevant project code. This will
likely be static for the majority of my projects in order to afford a simple and
reliable CLI for accessing my project code.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import yaml
import argparse

from navigation.navigation_main import NavigationMain


# global constants
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CUR_DIR, 'configs', 'default_config.yaml')


def parse_args():
    """
    Parse provided arguments from the command line.
    """
    arg_parser = argparse.ArgumentParser(
        description: "Argument parsing for accessing andrunning my deep "\
            "reinforcement learning projects")
    args = []

    return args


def load_config(path):
    """
    Load the configuration file that will specify the properties and parameters
    that may change in the general problem environment and/or the underlying RL
    agent/algorithm.
    """
    with open(path, 'r') as config:
        config_data = yaml.safe_load(config)

    return config_data['env_params'], config_data['model_params']


def main(config_file=DEFAULT_CONFIG):
    """
    Main runner for the code CLI.
    """
    env_params, model_params = load_config(config_file)
    navigation_prob = NavigationMain(**env_params, model_params)
    navigation_prob.run_interaction()


if __name__ == '__main__':
    args = parse_args()
    main(**args)
