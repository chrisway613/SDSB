# Training command

**2D experiments**

```bash
# Original DSB
torchrun --standalone train.py \
   --exp2d \                              # to train on 2D datasets
   --method dsb \                         # use DSB method
   --noiser dsb \                         # use DSB p_ref
   --prior dsb-pinwheel \                 # prior distribution (p_prior)
   --dataset checkerboard:8 \             # data distribution (p_data)
   --training_timesteps 16 \              # timesteps for training
   --inference_timesteps 16 \             # timesteps for inference
   --gamma_type linear_1e-4_1e-3 \        # gamma schedule, control the way to add noise
   --repeat_per_epoch 8 \                 # multiplier of iterations per epoch
   --epochs 41 \                          # total training epochs
   --exp_name dsb-pinwheel-checkerboard   # name of experiment

# Simplified DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name sdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify                             # use Simplified DSB

# Terminal Reparameterized DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name trdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify \                           # use Simplified DSB
   --reparam term                         # use Terminal Reparameterization

# Flow Reparameterized DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name frdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify \                           # use Simplified DSB
   --reparam flow                         # use Flow Reparameterization
```

**AFHQ 256**

To train from scratch, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256
```

To train with Flow Matching model as the initialization of backward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --backward_ckpt ./ckpt/afhq256_fm_dog2cat.pth --skip_epochs 1        # skip the first epoch (backward), start training from the second epoch (forward)
```

To train with the same Flow Matching model as the initialization of backward model and forward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --backward_ckpt ./ckpt/afhq256_fm_dog2cat.pth --forward_ckpt ./ckpt/afhq256_fm_dog2cat.pth --skip_epochs 1
```

To train with two seperate Flow Matching models as the initialization of backward model and forward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --backward_ckpt ./ckpt/afhq256_fm_dog2cat.pth --forward_ckpt ./ckpt/afhq256_fm_cat2dog.pth --skip_epochs 1
```

**AFHQ 512**

To train from scratch, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512
```

To train with Flow Matching model as the initialization of backward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --backward_ckpt ./ckpt/afhq512_fm_dog2cat.pth --skip_epochs 1        # skip the first epoch (backward), start training from the second epoch (forward)
```

To train with the same Flow Matching model as the initialization of backward model and forward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --backward_ckpt ./ckpt/afhq512_fm_dog2cat.pth --forward_ckpt ./ckpt/afhq512_fm_dog2cat.pth --skip_epochs 1
```

To train with two seperate Flow Matching models as the initialization of backward model and forward model, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --backward_ckpt ./ckpt/afhq512_fm_dog2cat.pth --forward_ckpt ./ckpt/afhq512_fm_cat2dog.pth --skip_epochs 1
```

If you want to accelerate the training with multiple nodes, e.g., 4, you could replace `torchrun --standalone --nproc_per_node=8` with `torchrun --nnodes=4 --nproc_per_node=8 --max_restarts=3 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=$NODE_RANK`, where `MASTER_ADDR`, `MASTER_PORT`, and `NODE_RANK` are distributed training related environment variables.
