import logging
import os
import time

import numpy as np
import torch
import torchvision
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm

import losses
import sde_lib
from datasets import get_dataloader
from forward_operator.measurements import get_operator, GaussianNoise
from metrics import calc_metrics
from models import ddpm, ncsnv2, ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from sampler import get_recon_algo, get_sampler
from utils import *


def main(argv):
  # recon(FLAGS.config, FLAGS.task, FLAGS.recon_folder, FLAGS.sample_algo, FLAGS.recon_algo)
  config = FLAGS.config
  algo_config = FLAGS.algo_config
  data_root = FLAGS.data_root
  recon_root = FLAGS.recon_root
  task = FLAGS.task
  dataset_name = FLAGS.dataset_name
  sample_algo = FLAGS.sample_algo
  recon_algo = FLAGS.recon_algo

  # Create directory
  recon_folder = os.path.join(recon_root, f"results_{dataset_name}")
  os.makedirs(recon_folder, exist_ok=True)
  label_dir = os.path.join(recon_folder, 'label')
  os.makedirs(label_dir, exist_ok=True)
  task_dir = os.path.join(recon_folder, f"{task}")
  os.makedirs(task_dir, exist_ok=True)
  for img_dir in ['measure', 'progress', 'recon', 'exp_log']:
    os.makedirs(os.path.join(task_dir, img_dir), exist_ok=True)
  progress_dir = os.path.join(task_dir, 'progress')
  recon_dir = os.path.join(task_dir, f'recon/{sample_algo}_{recon_algo}{config.sampling.sample_N}')
  os.makedirs(recon_dir, exist_ok=True)
  gfile_stream = open(os.path.join(task_dir, f'exp_log/{sample_algo}_{recon_algo}{config.sampling.sample_N}.txt'), 'w')
  logger = get_logger(gfile_stream)

  # Create data normalizer and its inverse
  scaler = get_data_scaler(config)
  inverse_scaler = get_data_inverse_scaler(config)

  # Create dataloaders
  loader = get_dataloader(dataset_name=dataset_name, root=data_root, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

  # Initialize model
  device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  score_model = mutils.create_model(config, device)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  ckpt_path = os.path.join(f'checkpoint/{dataset_name}.pth')
  state = restore_checkpoint(ckpt_path, state, device=device)
  ema.copy_to(score_model.parameters())
  score_model = score_model.to(device)
  print("model in:", next(score_model.parameters()).device)

  # Setup SDEs
  sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)

  operator, algo_config, y_shape = get_operator(task, algo_config, device)
  sigma_y = 0.05
  noiser = GaussianNoise(sigma=sigma_y)
  
  sampling_shape = (1, config.data.num_channels, config.data.image_size, config.data.image_size)
  recon_fn = get_recon_algo(recon_algo, algo_config, operator)
  sampling_fn = get_sampler(sde, sampling_shape, sigma_y, inverse_scaler, recon_fn, sample_algo, progress_dir, device)
  eps = 1e-3
  
  time_list = []
  for i, x_data in enumerate(loader):
    fname = str(i).zfill(5) + '.png'
    x_data = x_data.to(device)
    z0 = sde.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(device)

    torchvision.utils.save_image(x_data, os.path.join(label_dir, fname), nrow=10, normalize=False)
    y_data = noiser(operator.H(scaler(x_data)))

    if task == "inpainting":
      assert recon_algo != "fig", "use fig+ for inpainting."
      torchvision.utils.save_image(inverse_scaler(operator.H_pinv(y_data).reshape(y_shape)), os.path.join(task_dir, 'measure', fname), nrow=10, normalize=False)
    else:
      assert recon_algo != "fig+", "only use fig+ for inpainting"
      torchvision.utils.save_image(inverse_scaler(y_data.reshape(y_shape)), os.path.join(task_dir, 'measure', fname), nrow=10, normalize=False)

    if recon_algo == "pgdm":
      num_t = recon_fn.config.start / sde.sample_N * (sde.T - eps) + eps
      if torch.prod(torch.tensor(y_data.shape)) == torch.prod(torch.tensor(z0.shape)):
        z0 = num_t * y_data.reshape(z0.shape) + (1-num_t) * z0
      else:
        z0 = num_t * recon_fn.operator.H_pinv(y_data).reshape(z0.shape) + (1-num_t) * z0
      start_t = recon_fn.config.start
    else:
      start_t = 0

    start_time = time.time()
    if recon_algo == "pseudo":
      samples = inverse_scaler(operator.H_pinv(y_data).reshape(x_data.shape))
    else:
      samples = sampling_fn(score_model, z0, y_data, start_t, eps)
    end_time = time.time()

    time_list.append(end_time - start_time)

    torchvision.utils.save_image(samples.clamp_(0.0, 1.0), os.path.join(recon_dir, fname), nrow=10, normalize=False)

  time_list = np.array(time_list)
  print(f"average inference time: {np.mean(time_list):.4f}±{np.std(time_list):.4f}")
  calc_metrics(label_dir, recon_dir, device)



if __name__ == "__main__":
  seed = 45
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  FLAGS = flags.FLAGS
  config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
  config_flags.DEFINE_config_file("algo_config", None, "Training configuration.", lock_config=True)
  flags.DEFINE_string("data_root", None, "The data root")
  flags.DEFINE_string("recon_root", None, "The recon results root")
  flags.DEFINE_string("task", None, "The inverse problem task")
  flags.DEFINE_enum("dataset_name", None, ["celeba", "lsun_bedroom", "afhq_cat"], "Dataset")
  flags.DEFINE_enum("sample_algo", None, ["ode", "sde"], "Running mode")
  flags.DEFINE_enum("recon_algo", None, ["uncond", "pseudo", "dps", "pgdm", "dmps", "fig", "fig+"], "Running mode")
  flags.mark_flags_as_required(["config", "task", "dataset_name", "sample_algo", "recon_algo"])
  
  app.run(main)
