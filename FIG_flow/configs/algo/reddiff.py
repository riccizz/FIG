import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.c = 10.0
    sr.lr = 0.1
    sr.start = 10

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.c = 2.0

    # colorization
    config.co = co = ml_collections.ConfigDict()
    co.c = 5.0

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.c = 3.0

    return config