import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.c = 0.03

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.c = 0.04

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.c = 0.04

    # motion deblur
    config.md = md = ml_collections.ConfigDict()
    md.c = 0.025

    return config