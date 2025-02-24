import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.ngd = 5
    sr.c = 20
    sr.w = 0

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.ngd = 25
    gd.c = 10
    gd.w = 0

    # colorization
    config.co = co = ml_collections.ConfigDict()
    co.ngd = 5
    co.c = 10
    co.w = 0

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.ngd = 1
    inp.c = 10
    inp.w = 0

    # motion deblur
    config.md = md = ml_collections.ConfigDict()
    md.ngd = 3
    md.c = 25
    md.w = 0

    return config