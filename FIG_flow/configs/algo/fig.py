import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.k = 1
    sr.c = 20
    sr.w = 0

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.k = 25
    gd.c = 10
    gd.w = 0

    # motion deblur
    config.md = md = ml_collections.ConfigDict()
    md.k = 14
    md.c = 25
    md.w = 0

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.k = 2
    inp.c = 10
    inp.w = 0

    return config