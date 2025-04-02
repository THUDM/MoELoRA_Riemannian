# A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models
Source code of paper: A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models.

<p align="center">
📃 <a href="https://arxiv.org/abs/2502.15828" target="_blank">[Paper]</a> 
</p>

## Install
1. Clone this repository
   ```bash
   git clone https://github.com/THUDM/MoELoRA_Riemannian.git
   ```
2. Install dependencies
   ```bash
   conda create -n moelora_riemannian python=3.10 -y
   conda activate moelora_riemannian
   pip install -r requirements.txt
   ```
## Prepare ScienceQA Data (for example)

1. Prepare the datasets by this script:
   ```bash
   python prepare_data.py \
     --save_path "./dataset/scienceqa" 
   ```

2. Organize your datasets in the following structure:
   ```
   MoELoRA_Riemannian/dataset/
   ├── scienceqa/
   │   ├── science_qa.hf
   │   ├── scienceqa_train.json
   │   ├── scienceqa_test.json
   │   └── ...
   └── ...
   ```

## How to Run
```bash
# CUDA_VISIBLE_DEVICES=[GPU ID] python -m torch.distributed.launch --nproc_per_node 1 [TRAINING_SCRIPT] [DATASET] [OPTIMIZER] [METHOD]

# train MoE-LoRA with per-expert classic Riemannian preconditioners (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd riemannian

# train MoE-LoRA with per-expert classic Riemannian preconditioners (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw riemannian

# train MoE-LoRA with our method (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd ourmethod

# train MoE-LoRA with our method (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw ourmethod
```
## Citation

If you find our survey and repository helpful, please kindly cite our paper:

```
@article{sun2025stronger,
  title={A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models},
  author={Sun, Mengyang and Wang, Yihao and Feng, Tao and Zhang, Dan and Zhu, Yifan and Tang, Jie},
  journal={arXiv preprint arXiv:2502.15828},
  year={2025}
}
```
