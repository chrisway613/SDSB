# Simplified Diffusion Schrödinger Bridge

An unofficial implementation of the paper [Simplified Diffusion Schrödinger Bridge](https://arxiv.org/abs/2403.14623).

## Installation

1. Clone the repo
   
   ```bash
   git clone https://github.com/checkcrab/SDSB.git
   cd SDSB
   ```

2. Setup conda environment
   
   ```bash
   conda create -n sdsb python=3.10 -y
   conda activate sdsb

   # install torch first, here is an example for cuda 11.8
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # install required packages
   pip install -r requirements.txt
   ```

3. Prepare dataset

   Download the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [AFHQ](https://github.com/clovaai/stargan-v2) datasets into the folder `dataset`.

4. Download checkpoints

   We provide pretrained checkpoints [AFHQ256](https://github.com/tzco/storage/releases/download/sdsb/afhq256.pth), [AFHQ512](https://github.com/tzco/storage/releases/download/sdsb/afhq512.pth), [CelebA](https://github.com/tzco/storage/releases/download/sdsb/celeba.pth), and [2D experiment on pinwheel-checkerboard](https://github.com/tzco/storage/releases/download/sdsb/sdsb-pinwheel-checkerboard8.pth) for inference.

   We also provide Flow Matching models [AFHQ256 cat to dog](https://github.com/tzco/storage/releases/download/sdsb/afhq256_fm_cat2dog.pth), [AFHQ256 dog to cat](https://github.com/tzco/storage/releases/download/sdsb/afhq256_fm_dog2cat.pth), [AFHQ512 cat to dog](https://github.com/tzco/storage/releases/download/sdsb/afhq512_fm_cat2dog.pth), and [AFHQ512 dog to cat](https://github.com/tzco/storage/releases/download/sdsb/afhq512_fm_dog2cat.pth) for initialization.

   Download them into the folder `ckpt`, or you can also download with [`bash script/download_checkpoint.sh`](./script/download_checkpoint.sh).

## Inference

Here we provide some example scripts for sampling from pre-trained models.

**AFHQ 512**

```bash
python inference.py --network adm --prior afhq-dog-512 \
   --dataset afhq-cat-512 --simplify --reparam term \
   --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --ckpt ./ckpt/afhq512.pth --num_sample 128 \
   --batch_size 16
```

`--prior` sets the prior distribution ($p_{\text{prior}}$); `--dataset` is the data distribution ($p_{\text{data}}$); `--simplify` is a flag to use *Simplified DSB*; `--reparam` chooses the way for reparameterization, `term`
 means *Terminal Reparameterization*, `flow` means *Flow Reparameterization*, default is `None`; `--gamma_type` controls the way to add noise to construct $p_{\text{ref}}$; `--ckpt` points to the path of pre-trained model.

Or you could run `python inference.py -h` to see the full argument list.

**AFHQ 256**

```bash
python inference.py --network adm --prior afhq-dog-256 \
   --dataset afhq-cat-256 --simplify --reparam term \
   --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --ckpt ./ckpt/afhq256.pth
```

**CelebA 64**

```bash
python inference.py --network uvit-b --prior pixel-standard \
   --dataset celeba-64 --simplify --reparam term \
   --gamma_type linear_1e-5_1e-4 --exp_name trdsb-celeba \
   --ckpt ./ckpt/celeba.pth
```

**2D experiments**

```bash
python inference_2d.py --prior dsb-pinwheel --dataset checkerboard:8 \
   --exp2d --simplify --gamma_type linear_1e-4_1e-3 \
   --exp_name sdsb-pinwheel-checkerboard8 --ckpt ./ckpt/sdsb-pinwheel-checkerboard8.pth
```

## Training

**2D experiments**

```bash
# Simplified DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name sdsb-pinwheel-checkerboard --noiser flow --simplify
```

**AFHQ512**

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 --backward_ckpt ./ckpt/afhq512_fm_dog2cat.pth --forward_ckpt ./ckpt/afhq512_fm_cat2dog.pth --skip_epochs 1
```

For more training settings, please refer to [`training_command.md`](./training_command.md).
