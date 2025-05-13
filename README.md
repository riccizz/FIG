<div align="center">

# FIG: Flow with Interpolant Guidance for Linear Inverse Problems

[[OpenReview]](https://openreview.net/forum?id=fs2Z2z3GRx)
[[Project]](https://riccizz.github.io/FIG/)

</div>

## Description

Diffusion and flow matching models have been recently used to solve various linear inverse problems such as image restoration. Using a pre-trained diffusion or flow-matching model as a prior, most existing methods modify the reverse-time sampling process by incorporating the likelihood information from the measurement. However, they struggle in challenging scenarios, e.g., in case of high measurement noise or severe ill-posedness. In this paper, we propose Flow with Interpolant Guidance (FIG), an algorithm where the reverse-time sampling is efficiently guided with measurement interpolants through theoretically justified schemes. Experimentally, we demonstrate that FIG efficiently produce highly competitive results on a variety of linear image reconstruction tasks on natural image datasets. We improve upon state-of-the-art baseline algorithms, especially for challenging tasks. Code will be released.

<!-- <table align="center">
  <tr>
    <td align="center"><img src="assets/true_traj.gif" width="250"/></td>
    <td align="center"><img src="assets/rf_traj.gif" width="250"/></td>
    <td align="center"><img src="assets/hrf_traj.gif" width="250"/></td>
  </tr>
  <tr>
    <td align="center">Linear Interpolation</td>
    <td align="center">Rectified Flow</td>
    <td align="center">Hierarchical Rectified Flow (ours)</td>
  </tr>
</table> -->


## How to run

```bash
# clone project
git clone https://github.com/riccizz/FIG.git
cd FIG

# build environment
conda create -n fig python=3.10
conda activate fig
pip install -r requirements.txt

```

We use pretrained checkpoints from the following sources:

- [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) for training-free `FIG-flow`
- [DAPS](https://github.com/zhangbingliang2019/DAPS) and [DDNM](https://github.com/wyhuai/DDNM) for `FIG-diffusion`


## How to cite

If you find this code useful in your research, please cite the following papers:

```bibtex
@inproceedings{
    yan2025fig,
    title={{FIG}: Flow with Interpolant Guidance for Linear Inverse Problems},
    author={Yici Yan and Yichi Zhang and Xiangming Meng and Zhizhen Zhao},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=fs2Z2z3GRx}
}
```


## References

This repo is developed based on [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) and [DAPS](https://github.com/zhangbingliang2019/DAPS). Please also consider citing them if you use this repo. 

```bibtex
@article{liu2022flow,
    title={Flow straight and fast: Learning to generate and transfer data with rectified flow},
    author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
    journal={arXiv preprint arXiv:2209.03003},
    year={2022}
}

@misc{zhang2024improvingdiffusioninverseproblem,
      title={Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing}, 
      author={Bingliang Zhang and Wenda Chu and Julius Berner and Chenlin Meng and Anima Anandkumar and Yang Song},
      year={2024},
      eprint={2407.01521},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01521}, 
}
```