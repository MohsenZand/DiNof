# [Diffusion Models with Deterministic Normalizing Flow Priors](https://arxiv.org/)

Our code is built upon the code from [Song et al.](https://github.com/yang-song/score_sde_pytorch). 
It is tested under Ubuntu 20.04, CUDA 11.6, with NVIDIA A100 GPUs. Python 3.11.1 version is used for development. 

## Datasets
CIFAR-10 dataset is downloaded automatically.
To download CelebA-HQ-256 dataset, you can follow the instructions given in [Vahdat and Kautz](https://github.com/NVlabs/NVAE). 

Due to the size limit, we can not include the FID statistics files, which are required to compute the evaluation metrics. For each dataset, however, you can use the following python scripts located in the `fids` folder to compute FID statistics on the CIFAR-10 and CelebA-HQ-256 datasets: `fid_score.py` and `precompute_fid_statistics.py`

## Training and evaluation
All models can be found in the `models` folder. 

To train a specific model, please run `main.py` and modify the flags to define the required paths and directories. Other flags can be changed if needed. 
For instance, set `mode` as `train` or `eval` for training or evaluation, respectively. 
Also, choose one of the 7 provided config files and its corresponding SDE (ve, vp or subvp).  
All the config files, training and sampling files are self-explanatory. 

## Citation
Please cite our paper if you use code from this repository:
```
@article{zand2023diffusion,
  title={Diffusion Models with Deterministic Normalizing Flow Priors},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  journal={arXiv preprint arXiv:2308.16801},
  year={2023}
}
```