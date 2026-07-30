# DoC

## Description

An unofficial implementation of the paper [Predicting with Confidence on Unseen Distributions](https://arxiv.org/abs/2107.03315).

## Usage

In the environment, we just need PyTorch with a suitable CUDA version.

### Step 1: Clone this repository and change directory to repository root
```bash
git clone https://github.com/ayanban011/DoC.git 
cd DoC
```

### Step 2: Full experiment on PACS, all target domains, all methods:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset pacs --data_root /data/pacs
```

### Step 3: Skip training (use existing checkpoint):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset pacs --data_root /data/pacs \
        --checkpoint results/pacs/art_painting/erm_best.pt --skip_train
```

### Step 4: Single target domain, lightweight backbone:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset pacs --data_root /data/pacs \
        --target_domain photo --backbone resnet18 --epochs 20
```

### Step 5: DomainNet (large — reduce batch size):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset domain_net --data_root /data/domain_net \
        --backbone resnet50 --batch_size 32 --epochs 30
```
