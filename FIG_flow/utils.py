import logging
import os
import torch
import matplotlib.pyplot as plt

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def get_logger(gfile_stream):
  logger = logging.getLogger(name='exp')
  logger.setLevel(logging.INFO)
  
  formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
  stream_handler = logging.StreamHandler(gfile_stream)
  stream_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)
  
  return logger

