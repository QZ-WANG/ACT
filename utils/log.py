import os
from os.path import join
import json
import logging


def set_log_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise RuntimeError("Project already exists")


def save_config(args, log_path):
    fn = join(log_path, "config.json")

    with open(fn, 'w') as fp:
        json.dump(args, fp, sort_keys=False, indent=4)
        fp.write("\n")


def set_logger(log_path, timestamp, name):
    logging.basicConfig(
        filename=join(log_path, timestamp+".log"),
        filemode='a',
        format='%(asctime)s,%(msecs)3d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )

    logger = logging.getLogger(name)

    return logger


def set_up_logging(path, timestamp, name=None):
    set_log_dir(path)
    logger = set_logger(path, timestamp, name)
    return logger
