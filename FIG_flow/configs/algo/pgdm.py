import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()
    sr.start = 10

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()
    gd.start = 10

    # colorization
    config.co = co = ml_collections.ConfigDict()
    co.start = 10

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    inp.start = 10

    # motion deblur
    config.md = md = ml_collections.ConfigDict()
    md.start = 10
    
    return config