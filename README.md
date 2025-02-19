# A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models
Source code of paper: A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models.

## How to Run
```
# CUDA_VISIBLE_DEVICES=[GPU ID] python -m torch.distributed.launch --nproc_per_node 1 [TRAINING_SCRIPT] [DATASET] [OPTIMIZER] [METHOD]

# run MoE with classic Riemannian preconditioners (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd riemannian

# run MoE with classic Riemannian preconditioners (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw riemannian

# run MoE our method (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd ourmethod

# run MoE our method (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw ourmethod
```
