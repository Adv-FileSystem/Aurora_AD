# InformerAD v2.0

## Train + Eval (single GPU)
torchrun --nproc_per_node=2 InformerAD_2y_ver2_PAD.py \
  --ddp --mode single \
  --train_csv /home/keti/RANSynCoders/data/train_filled.csv \
  --test_csv /home/keti/RANSynCoders/data/test.csv \
  --test_label /home/keti/RANSynCoders/data/test_label.csv \
  --L_enc 192 \
  --L_pred 60 \
  --horizons 1 60 \
  --prior_beta 0.30 \
  --amp --amp_dtype bf16 \
  --epochs 40 \
  --batch_size 64 \
  --lr 2e-4 \
  --wd 1e-4 \
  --grad_clip 1.0 \
  --hop 2 \
  --ckpt_dir checkpoints_multi_PAD \
  --residual_norm_mode ema

## Test 
torchrun --nproc_per_node=2 InformerAD_2y_ver2_PAD.py \
  --ddp --eval_only \
  --train_csv /home/keti/RANSynCoders/data/train_filled.csv \
  --test_csv /home/keti/RANSynCoders/data/test.csv \
  --test_label /home/keti/RANSynCoders/data/test_label.csv \
  --ckpt_path checkpoints_multi_PAD/H1_60_b30_informer_mt_best.pt \
  --amp --amp_dtype bf16

