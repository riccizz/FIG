import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # super resolution
    config.sr = sr = ml_collections.ConfigDict()

    # gaussian deblur
    config.gd = gd = ml_collections.ConfigDict()

    # inpainting
    config.inp = inp = ml_collections.ConfigDict()
    
    return config