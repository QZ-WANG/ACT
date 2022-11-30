import os
import warnings


def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        warnings.warn("directory %s already exists!" % directory)