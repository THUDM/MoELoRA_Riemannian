# A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models
Source code of paper: A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models.

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
## Prepare ScienceQA Data

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

# run MoE with classic Riemannian preconditioners (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd riemannian

# run MoE with classic Riemannian preconditioners (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw riemannian

# run MoE our method (the SGD optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA sgd ourmethod

# run MoE our method (the AdamW optimizer)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_llama.py ScienceQA adamw ourmethod
```
