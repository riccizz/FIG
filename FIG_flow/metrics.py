import glob
import logging
import os
import PIL
import sys

import numpy as np
import matplotlib.pyplot as plt
import lpips
import torch
from absl import flags
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm

def calc_metrics(label_dir, recon_dir, device):
    logger = logging.getLogger('exp')
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    files = glob.glob(recon_dir + '/*')

    psnr_list = []
    lpips_list = []
    ssim_list = []

    for idx in tqdm(range(len(files))):
        fname = str(idx).zfill(5) + '.png'

        label = plt.imread(os.path.join(label_dir, fname))[:, :, :3]
        exp_recon = plt.imread(os.path.join(recon_dir, fname))[:, :, :3]

        psnr_exp = peak_signal_noise_ratio(label, exp_recon)
        psnr_list.append(psnr_exp)

        exp_recon = torch.from_numpy(exp_recon).permute(2, 0, 1).to(device)
        label = torch.from_numpy(label).permute(2, 0, 1).to(device)
        ssim_exp =  ssim(label.reshape(1, *label.shape), exp_recon.reshape(1, *exp_recon.shape))
        ssim_list.append(ssim_exp.item())

        exp_recon = exp_recon.view(1, 3, 256, 256) * 2. - 1.
        label = label.view(1, 3, 256, 256) * 2. - 1.

        ytgd_d = loss_fn_vgg(exp_recon, label)
        lpips_list.append(ytgd_d.item())
        
    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    lpips_avg = sum(lpips_list) / len(lpips_list)
    
    logger.info(f'PSNR: {psnr_avg:.3f}±{np.std(psnr_list):.3f}')
    logger.info(f'SSIM: {ssim_avg:.3f}±{np.std(ssim_list):.3f}')
    logger.info(f'LPIPS: {lpips_avg:.3f}±{np.std(lpips_list):.3f}')

    # logger.info(f'LPIPS list: {lpips_list}')

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("label_dir", None, "The label root")
    flags.DEFINE_string("recon_dir", None, "The recon results root")
    flags.mark_flags_as_required(["label_dir", "recon_dir"])
    FLAGS(sys.argv)
    
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    calc_metrics(FLAGS.label_dir, FLAGS.recon_dir, device)
