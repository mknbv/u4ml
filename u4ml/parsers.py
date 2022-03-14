""" Defines helper methods for command line arguments parsing. """
from argparse import ArgumentParser
import os


def get_defaults_parser(defaults, base_parser=None):
  """ Adds dictionary of defaults to a parser. """
  if base_parser is None:
    base_parser = ArgumentParser()
  for key, val in defaults.items():
    if isinstance(val, dict):
      base_parser.add_argument(f"--{key}", **val)
    else:
      base_parser.add_argument(f"--{key}", type=type(val), default=val)
  return base_parser


def log_args(args, logdir=None):
  """ Writes `Namespace` of arguments to a text file under logdir directory. """
  if logdir is None:
    logdir = args.logdir
  if not os.path.isdir(logdir):
    os.makedirs(logdir, exist_ok=True)
  with open(os.path.join(logdir, "args.txt"),
            'w', encoding="ascii") as argsfile:
    for key, val in vars(args).items():
      argsfile.write(f"{key}: {val}\n")
  return args
