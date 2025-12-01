# -*- coding: utf-8 -*-
"""
viz_test_horizon_allfeat.py

- 목적: 학습된 체크포인트(InformerMT + RevIN)를 불러와
       "테스트셋의 특정 윈도우에 대해, 지정한 horizon(기본 30초)에서의 예측 vs 실제"
       를 모든 feature에 대해 한 장의 그래프로 저장
- 특징:
  * horizon(초) 지정 가능 (--horizon, 기본 30)
  * 특정 윈도우 인덱스 지정 (--win_idx, 기본 0)
  * 모든 피처를 x축으로 두고, 해당 horizon 시점의 '실측 vs 예측'을 2개 선으로 비교
  * 기존 학습 스크립트 모듈(예: InformerAD_GPU_DDP_4.py 또는 InformerAD_MT_ForecastAnom_AD+Quantile.py)을 import

사용 예:
python viz_test_horizon_allfeat.py \
  --train_csv ./RANSynCoders/data/train.csv \
  --test_csv  ./RANSynCoders/data/test.csv  \
  --ckpt      ./checkpoints_mt_for4/informer_mt_best.pt \
  --module    InformerAD_MT_ForecastAnom_AD+Quantile \
  --win_idx   3 \
  --horizon   30 \
  --outdir    ./viz_out
"""
import os, sys, math, argparse
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) 학습 모듈 import
# -----------------------------
def _try_import_training_module(modname: str):
    try:
        mod = __import__(modname, fromlist=['*'])
        return mod
    except Exception as e:
        print(f"[WARN] import {modname} 실패: {e}")
        return None

_CANDIDATE_MODULES = [
    "InformerAD_GPU_DDP_4",
    "InformerAD_MT_ForecastAnom_AD+Quantile",
]

# -----------------------------
# 2) 로컬 유틸
# -----------------------------
def load_csv_numeric(path, index_col=0):
    df = pd.read_csv(path, index_col=index_col)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols)==0: raise RuntimeError(f"{path}: 숫자형 컬럼 없음")
    df = df[num_cols].copy()
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df = df.interpolate(method="linear",limit_direction="both",axis=0).ffill().bfill().fillna(0.0)
    df = df.clip(lower=-1e9, upper=1e9)
    return df

def make_forecast_windows(X: np.ndarray, L:int, K:int, hop:int=1):
    T,D=X.shape; past=[]; fut=[]; idxs=[]
    for s in range(0, T-(L+K)+1, hop):
        e_p=s+L; e_f=s+L+K
        past.append(X[s:e_p]); fut.append(X[e_p:e_f]); idxs.append((s,e_p,e_f))
    if not past: return np.empty((0,L,D)), np.empty((0,K,D)), idxs
    return np.stack(past,0), np.stack(fut,0), idxs

def build_time_seconds(seconds: np.ndarray):
    s=seconds.astype(np.float64)
    som=np.mod(s,60.0)/60.0*2*np.pi; soh=np.mod(s,3600.0)/3600.0*2*np.pi
    sod=np.mod(s,86400.0)/86400.0*2*np.pi; sow=np.mod(s,604800.0)/604800.0*2*np.pi
    feats=np.stack([np.sin(som),np.cos(som),np.sin(soh),np.cos(soh),
                    np.sin(sod),np.cos(sod),np.sin(sow),np.cos(sow)],1).astype(np.float32)
    return feats

class WindowsToInformerDatasetWithTime(Dataset):
    def __init__(self,past,future,idxs,time_all,label_len:int):
        self.past=past.astype(np.float32); self.future=future.astype(np.float32)
        self.idxs=idxs; self.time_all=time_all.astype(np.float32) if time_all is not None else None
        self.label_len=int(label_len)
    def __len__(self): return self.past.shape[0]
    def __getitem__(self,i):
        s,e_p,e_f=self.idxs[i]; x_enc=self.past[i]; K=self.future.shape[1]
        x_dec_hist=x_enc[-self.label_len:]; zeros=np.zeros((K,x_enc.shape[1]),np.float32)
        x_dec=np.concatenate([x_dec_hist,zeros],0); y=self.future[i]
        if self.time_all is None: t_enc=None; t_dec=None
        else:
            t_enc=self.time_all[s:e_p]; t_dec=self.time_all[e_p-self.label_len:e_f]
        return (torch.from_numpy(x_enc),
                (torch.from_numpy(t_enc) if t_enc is not None else None),
                torch.from_numpy(x_dec),
                (torch.from_numpy(t_dec) if t_dec is not None else None),
                torch.from_numpy(y),
                torch.tensor([s,e_p,e_f], dtype=torch.long))

def denorm_if_needed(x, revin, stats, use_revin=True):
    if use_revin and (revin is not None) and (stats is not None):
        z,_ = revin(x, mode='denorm', stats=stats)
        return z
    return x

# -----------------------------
# 3) 메인
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv",  required=True)
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--module",    default=None, help="학습 스크립트 모듈명(확장자 .py 제외)")
    parser.add_argument("--outdir",    default="./viz_out")
    parser.add_argument("--win_idx",   type=int, default=0, help="테스트 윈도우 인덱스(0부터)")
    parser.add_argument("--horizon",   type=int, default=30, help="초 단위 horizon (예: 30 → 30초 후)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 학습 모듈 import
    modname = args.module
    mod = None
    tried = []
    if modname:
        tried.append(modname)
        mod = _try_import_training_module(modname)
    if mod is None:
        for cand in _CANDIDATE_MODULES:
            tried.append(cand)
            mod = _try_import_training_module(cand)
            if mod is not None:
                modname = cand
                break
    if mod is None:
        raise RuntimeError(f"학습 모듈 import 실패. 시도한 모듈: {tried}\n"
                           f"→ --module 로 정확한 파일명(.py 제외)을 지정해줘.")

    # 필요한 심볼들
    required = ["InformerMT","RevIN","TrainCfg","decode_autoregressive"]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"모듈 '{modname}'에 '{name}'가 필요합니다.")

    InformerMT = getattr(mod, "InformerMT")
    RevIN      = getattr(mod, "RevIN")
    TrainCfg   = getattr(mod, "TrainCfg")
    decode_autoregressive = getattr(mod, "decode_autoregressive")

    # 데이터 로딩
    x_train_df = load_csv_numeric(args.train_csv)
    x_test_df  = load_csv_numeric(args.test_csv)

    # cfg & 윈도우 구성(학습과 동일)
    cfg = TrainCfg()
    L, K, HOP = cfg.L_enc, cfg.L_pred, cfg.hop
    if args.horizon < 1 or args.horizon > K:
        raise ValueError(f"--horizon(초) {args.horizon}가 L_pred={K} 범위를 벗어났습니다. (1~{K})")

    t_train = build_time_seconds(x_train_df.index.to_numpy(dtype=float))
    t_test  = build_time_seconds(x_test_df.index.to_numpy(dtype=float))

    te_p, te_f, te_idx = make_forecast_windows(x_test_df.values.astype(np.float32), L, K, HOP)
    if len(te_p) == 0:
        raise RuntimeError("테스트 윈도우가 없습니다. (데이터 길이 < L_enc + L_pred)")
    if args.win_idx < 0 or args.win_idx >= len(te_p):
        raise IndexError(f"--win_idx {args.win_idx} 가 테스트 윈도우 개수 {len(te_p)} 범위를 벗어났습니다.")

    ds_te = WindowsToInformerDatasetWithTime(te_p, te_f, te_idx, t_test, cfg.L_label)
    # 필요한 윈도우만 직접 꺼내자
    x_enc, t_enc, x_dec, t_dec, y_true, idx_triplet = ds_te[args.win_idx]

    FEAT_DIM = x_train_df.shape[1]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 모델 & RevIN
    model = InformerMT(
        d_in=FEAT_DIM, d_out=FEAT_DIM,
        d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers, d_layers=cfg.d_layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout, use_distill=cfg.use_distill,
        d_time=8, use_checkpoint=False, horizons_idx=cfg.horizons, quantiles=cfg.quantiles
    ).to(device)
    revin = RevIN(FEAT_DIM, affine=cfg.use_revin_affine).to(device) if getattr(cfg, "use_revin", True) else None

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state", ckpt)  # 저장 포맷 방어
    model.load_state_dict(state, strict=False)
    if revin is not None and ("revin_state" in ckpt) and (ckpt["revin_state"] is not None):
        try:
            revin.load_state_dict(ckpt["revin_state"], strict=False)
        except Exception as e:
            print(f"[WARN] RevIN state load 실패: {e}")

    model.eval()
    if revin is not None: revin.eval()

    # 배치 차원 추가 + 디바이스 이동
    x_enc = x_enc.unsqueeze(0).to(device)
    x_dec = x_dec.unsqueeze(0).to(device)
    t_enc = (t_enc.unsqueeze(0).to(device)) if t_enc is not None else None
    t_dec = (t_dec.unsqueeze(0).to(device)) if t_dec is not None else None
    y_true = y_true.unsqueeze(0).to(device)

    # RevIN 정규화
    if getattr(cfg, "use_revin", True) and (revin is not None):
        x_enc_n, revin_stats = revin(x_enc, mode="norm")   # stats
        x_dec_n,_ = revin(x_dec, mode="norm")
        y_n,_     = revin(y_true, mode="norm")
    else:
        x_enc_n, x_dec_n, y_n = x_enc, x_dec, y_true
        revin_stats = None

    # 중앙 예측(q50) AR rollout 후 denorm
    with torch.no_grad():
        pred_n = decode_autoregressive(
            model, revin, cfg,
            x_enc_n, t_enc,
            x_dec_n[:, :cfg.L_label, :], t_dec,
            teacher_future_n=None, teacher_prob=0.0, train_mode=False
        )
    pred = denorm_if_needed(pred_n, revin, revin_stats, getattr(cfg, "use_revin", True))

    # horizon 시점(초) → 인덱스 변환: 1초 후 → index 0, 30초 후 → index 29
    h = int(args.horizon)
    step_idx = h - 1

    # 해당 step에서 모든 피처 비교
    y_true_vec = y_true[0, step_idx, :].detach().cpu().numpy()  # (D,)
    y_pred_vec = pred[0, step_idx, :].detach().cpu().numpy()    # (D,)

    D = y_true_vec.shape[0]
    xs = np.arange(D)

    # 플롯
    s, e_p, e_f = idx_triplet.tolist()
    title = (f"Test window #{args.win_idx} | idx=({int(s)}→{int(e_f)}) | "
             f"horizon={h}s | features={D}")

    plt.figure(figsize=(max(10, D*0.25), 5))
    # 두 개 라인: True, Pred
    plt.plot(xs, y_true_vec, label="True")
    plt.plot(xs, y_pred_vec, linestyle="--", label="Pred (q50)")
    plt.xlabel("Feature index")
    plt.ylabel(f"Value @ +{h}s")
    plt.title(title)
    plt.legend(loc="best")
    plt.xticks(xs, xs, rotation=0)
    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"h{h}s_win{args.win_idx:05d}_allfeat.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # 값도 CSV로 같이 저장(분석 편의)
    out_csv = os.path.join(args.outdir, f"h{h}s_win{args.win_idx:05d}_allfeat.csv")
    pd.DataFrame({"feature_idx": xs, "true": y_true_vec, "pred": y_pred_vec}).to_csv(out_csv, index=False)

    print(f"[DONE] 그래프: {out_png}")
    print(f"[DONE] 값 CSV: {out_csv}")

if __name__ == "__main__":
    main()

