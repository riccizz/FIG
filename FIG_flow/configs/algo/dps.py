import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.c = 8.0

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.c = 2.0

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.c = 3.0

    # motion deblur
    config.md = md = ml_collections.ConfigDict()
    md.c = 1.0

    return config