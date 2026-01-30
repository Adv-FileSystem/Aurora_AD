# InformerAD v2.0 (PAD-only AUPRC)

## Environment
- PyTorch + CUDA
- torchrun (DDP optional)

## Train + Eval (single GPU)
python InformerAD_2y.py \
  --mode single \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --test_label data/test_label.csv \
  --L_enc 192 \
  --L_pred 60 \
  --horizons 1 60 \
  --prior_beta 0.5 \
  --epochs 40

## Train + Eval (DDP, 2 GPU)
torchrun --nproc_per_node=2 InformerAD_2y.py \
  --ddp \
  --mode single \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --test_label data/test_label.csv

