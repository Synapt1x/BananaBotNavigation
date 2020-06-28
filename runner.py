# -*- coding: utf-8 -*-
"""
This code is the main runner CLI for the relevant project code. This will
likely be static for the majority of my projects in order to afford a simple and
reliable CLI for accessing my project code.

Author: Chris Cadonic

For Deep Reinforcement Learning Nanodegree offered by Udacity.
"""


import os
import argparse


def parse_args():
    """
    Parse provided arguments from the command line.
    """
    arg_parser = argparse.ArgumentParser(
        description: "Argument parsing for accessing andrunning my deep "\
            "reinforcement learning projects")
    args = []

    return args


def main():
    """
    Main runner for the code CLI.
    """
    pass



if __name__ == '__main__':
    args = parse_args()
    main(**args)
