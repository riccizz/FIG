import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision
from torch.fft import fft2, ifft2
from tqdm import tqdm

from models import utils as mutils


__RECON_ALGO__ = {}

def register_recon_algo(name: str):
    def wrapper(cls):
        if __RECON_ALGO__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __RECON_ALGO__[name] = cls
        return cls
    return wrapper

def get_recon_algo(name: str, config, operator, **kwargs):
    if __RECON_ALGO__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __RECON_ALGO__[name](name=name, config=config, operator=operator, **kwargs)

class ReconAlgo(ABC):
  def __init__(self, name, config, operator):
    self.name = name
    self.config = config
    self.operator = operator

  @abstractmethod
  def update(self, **kwargs):
    pass

@register_recon_algo(name='pseudo')
class Pseudo(ReconAlgo):
  def update(self, **kwargs):
    pass
  
@register_recon_algo(name='uncond')
class Uncond(ReconAlgo):
  def update(self, x, x_next, y, v, t, **kwargs):
    x0hat = (1. - t) * v + x
    norm = torch.linalg.norm(y - self.operator.H(x0hat))
    return x_next, norm

@register_recon_algo(name='dps')
class DPS(ReconAlgo):
  def update(self, x, x_next, i, y, v, t, N, **kwargs):
    x0hat = (1. - t) * v + x
    norm = torch.linalg.norm(y - self.operator.H(x0hat))
    if i > 1:
      grad = torch.autograd.grad(outputs=norm, inputs=x, retain_graph=True)[0]
      x_next = x_next - self.config.c * grad
    return x_next, norm

@register_recon_algo(name='pgdm')
class PGDM(ReconAlgo):  
  def update(self, x, x_next, y, v, i, t, N, sigma_y, **kwargs):
    n = x.shape[0]
    x0hat = (1. - t) * v + x
    if self.operator.is_fft:
      norm = torch.linalg.norm(y - self.operator.H(x0hat))
      if i > 1:
        rt_square = (1-t)**2 / (t**2 + (1-t)**2)
        FB, FBC, F2B, _ = self.operator.pre_calculated
        mat = ifft2(FBC / (sigma_y**2 + rt_square * F2B) * fft2(y - ifft2(FB * fft2(x0hat)))).real
        mat_x = (mat.detach() * x0hat).sum()
        grad = torch.autograd.grad(outputs=mat_x, inputs=x, retain_graph=True)[0].detach()
        x_next = x_next + (1-t) / t * grad / N
    else:
      norm = torch.linalg.norm(y.reshape(n, -1) - self.operator.H(x0hat))
      if i > 1:
        sigulars = self.operator.singulars()
        rt_square = (1-t)**2 / (t**2 + (1-t)**2)
        S = 1 / (rt_square * sigulars ** 2 + sigma_y ** 2)
        temp = self.operator.Ut(y - self.operator.H(x0hat))
        mat = self.operator.Ht(self.operator.U(S * temp))
        mat_x = (mat.detach() * x0hat.reshape(n, -1)).sum()
        grad = torch.autograd.grad(outputs=mat_x, inputs=x, retain_graph=True)[0].detach()
        x_next = x_next + (1-t) / t * grad / N
    return x_next, norm

@register_recon_algo(name='dmps')
class DMPS(ReconAlgo):
  def update(self, x, x_next, y, v, i, t, sigma_y, **kwargs):
    n = x.shape[0]
    x0hat = (1. - t) * v + x
    if self.operator.is_fft:
      norm = torch.linalg.norm(y - self.operator.H(x0hat))
      if i > 1:
        scale_S = ((1-t) / t) ** 2
        FB, FBC, F2B, _ = self.operator.pre_calculated
        grad = ifft2(FBC / (sigma_y**2 + scale_S * F2B) * fft2(y - ifft2(FB * fft2(x_next)) / t)).real / t
        x_next = x_next + self.config.c * (1 - t) / t * grad
    else:
      norm = torch.linalg.norm(y.reshape(n, -1) - self.operator.H(x0hat))
      if i > 1:
        sigulars = self.operator.singulars()
        scale_S = ((1-t) / t) ** 2
        S = 1 / (scale_S * sigulars ** 2 + sigma_y ** 2)
        temp = self.operator.Ut(y - self.operator.H(x_next) / t)
        grad = self.operator.Ht(self.operator.U(S * temp)).reshape(x.shape) / t
        x_next = x_next + self.config.c * (1 - t) / t * grad
    return x_next, norm

@register_recon_algo(name='fig')
class FIG(ReconAlgo):
  def update(self, x, x_next, y, v, i, t, next_t, N, noise_y, **kwargs):
    yt = next_t * y + self.config.w * (1-t) * self.operator.H(torch.randn_like(x_next))  # rescaled version of stochastic interpolant 
    if i > 1 and i != N-1:
      for _ in range(self.config.k):
        norm = torch.linalg.norm(yt - self.operator.H(x_next))
        grad = torch.autograd.grad(outputs=norm, inputs=x_next, retain_graph=True)[0]
        x_next = x_next - self.config.c * (1 - t) / t * grad
    norm = torch.linalg.norm(yt - self.operator.H(x_next))
    return x_next, norm

@register_recon_algo(name='fig+')
class FIGplus(ReconAlgo):
  def update(self, x, x_next, y, v, i, t, next_t, N, noise_y, **kwargs):
    x0hat = (1. - t) * v + x
    xt = next_t * x0hat + (1-next_t) * torch.randn_like(x0hat)
    masked = xt - self.operator.H_pinv(self.operator.H(xt)).reshape(xt.shape)
    mix_coef = 0.95
    yt = next_t * y + self.config.w * (1-t) * self.operator.H(torch.randn_like(x_next))  # rescaled version of stochastic interpolant 
    if i > 1 and i != N-1:
      for _ in range(self.config.k):
        norm = torch.linalg.norm(yt - self.operator.H(x_next))
        grad = torch.autograd.grad(outputs=norm, inputs=x_next, retain_graph=True)[0]
        x_next = x_next - self.config.c * (1 - t) / t * grad
      x_next = (1-mix_coef) * x_next + mix_coef * masked + mix_coef * self.operator.H_pinv(self.operator.H(x_next)).reshape(x_next.shape)
    norm = torch.linalg.norm(yt - self.operator.H(x_next))
    return x_next, norm


def get_sampler(sde, shape, sigma_y, inverse_scaler, recon_fn, sample_algo, progress_dir, device='cuda'):
  def euler_sampler(model, z0, y, start_t, eps=1e-3, test=True):
    model_fn = mutils.get_model_fn(model, train=False)
    
    # Initial sample
    x = z0.detach().clone()
    noise_y = recon_fn.operator.H(x)

    ### Uniform
    dt = 1./sde.sample_N
    
    pbar = tqdm(range(start_t, sde.sample_N))
    for i in pbar:
      num_t = i / sde.sample_N * (sde.T - eps) + eps
      next_t = (i+1) / sde.sample_N * (sde.T - eps) + eps
      t = torch.ones(shape[0], device=device) * num_t

      x = x.requires_grad_()
      v = model_fn(x, t*999)

      if sample_algo == "ode":
        x_next = x + v * dt
      elif sample_algo == "sde":
        x_next = x + ((1. + num_t/2) * v - x / 2) * dt + np.sqrt((1. - num_t) * dt) * torch.randn_like(x)

      x_next, norm = recon_fn.update(x=x, x_next=x_next, y=y, v=v, i=i, t=num_t, next_t=next_t, N=sde.sample_N, sigma_y=sigma_y, noise_y=noise_y)
      
      x = x_next.detach()
      
      if test and (i % 10 == 0 or i > sde.sample_N - 9 or i < 10):
        torchvision.utils.save_image(inverse_scaler(x), os.path.join(progress_dir, f'xt_{i}.png'), nrow=10, normalize=False)

      pbar.set_postfix({'distance': norm.item()}, refresh=False)
    
    x = inverse_scaler(x)
    
    return x
  return euler_sampler

