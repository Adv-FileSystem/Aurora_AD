#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, copy, random, time, json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support
)

# =========================================================
# DDP utils
# =========================================================
def dist_is_init() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def env_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1

def env_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0

def ddp_enabled(arg_ddp: bool) -> bool:
    return bool(arg_ddp) and torch.cuda.is_available() and env_world_size() > 1

def ddp_setup(arg_ddp: bool) -> int:
    """
    torchrun 환경에서만 init. LOCAL_RANK 반환.
    """
    if ddp_enabled(arg_ddp) and not dist_is_init():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return int(os.environ.get("LOCAL_RANK", "0"))

def ddp_rank0() -> bool:
    ws = env_world_size()
    if ws > 1:
        return env_rank() == 0
    return True

def ddp_barrier():
    if dist_is_init():
        dist.barrier()

def ddp_cleanup():
    if dist_is_init():
        dist.destroy_process_group()

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

# =========================================================
# DDP gather helper (length mismatch safe)
# =========================================================
def _ddp_allgather_int(value: int, device) -> List[int]:
    if not dist_is_init():
        return [int(value)]
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [int(x.item()) for x in out]

def _ddp_pad_allgather_sum_1d(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    assert x.dim() == 1, f"expected 1D tensor, got {tuple(x.shape)}"
    if not dist_is_init():
        return x, int(x.numel())
    lens = _ddp_allgather_int(int(x.numel()), x.device)
    min_len = min(lens)
    max_len = max(lens)
    if int(x.numel()) < max_len:
        pad = torch.zeros(max_len - int(x.numel()), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x
    gathered = [torch.zeros_like(x_pad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x_pad)
    s = torch.stack(gathered, 0).sum(0)
    if min_len < max_len:
        s = s[:min_len]
    return s, int(min_len)

# =========================================================
# misc
# =========================================================
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cuda_amp_enabled(device: str) -> bool:
    return ("cuda" in str(device).lower()) and torch.cuda.is_available()

# =========================================================
# IO
# =========================================================
def load_csv_numeric(path: str, index_col: Optional[int] = 0):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=index_col)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise RuntimeError(f"{path} 숫자형 컬럼 없음")
    df = df[num_cols].copy()

    bad0 = (~np.isfinite(df.values)).sum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate(method="linear", limit_direction="both", axis=0).ffill().bfill().fillna(0.0)
    bad1 = (~np.isfinite(df.values)).sum()

    if bad0 > 0 and ddp_rank0():
        print(f"[clean] {path}: non-finite {bad0}개 → {bad1}개(잔여)")

    df = df.clip(lower=-1e9, upper=1e9)
    return df

# =========================================================
# RevIN
# =========================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode: str = "norm", stats=None):
        if mode == "norm":
            m = x.mean(1, keepdim=True)
            v = x.var(1, keepdim=True, unbiased=False)
            z = (x - m) / torch.sqrt(v + self.eps)
            if self.affine:
                z = z * self.gamma + self.beta
            return z, (m, v)
        elif mode == "denorm":
            m, v = stats
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-12)
            return x * torch.sqrt(v + self.eps) + m, None
        else:
            raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# window maker
# =========================================================
def make_forecast_windows(X: np.ndarray, L: int, K: int, hop: int = 1):
    T, D = X.shape
    past, fut, idxs = [], [], []
    for s in range(0, T - (L + K) + 1, hop):
        e_p = s + L
        e_f = s + L + K
        past.append(X[s:e_p])
        fut.append(X[e_p:e_f])
        idxs.append((s, e_p, e_f))
    if not past:
        return np.empty((0, L, D)), np.empty((0, K, D)), idxs
    return np.stack(past, 0), np.stack(fut, 0), idxs

# =========================================================
# Positional/Embedding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, L, D = x.shape
        dev = x.device
        dt = x.dtype
        pos = torch.arange(L, device=dev, dtype=dt).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=dev, dtype=dt) * (-math.log(10000.0) / D))
        pe = torch.zeros(L, D, device=dev, dtype=dt)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

class DataEmbedding(nn.Module):
    def __init__(self, d_in, d_model, d_time=0, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.use_time = d_time > 0
        self.time_proj = nn.Linear(d_time, d_model) if self.use_time else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_time=None):
        h = self.proj(x)
        h = self.pos(h)
        if self.use_time and x_time is not None:
            h = h + self.time_proj(x_time)
        return self.drop(h)

# =========================================================
# Informer core
# =========================================================
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5.0, sample_mult=2.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ad = nn.Dropout(dropout)
        self.od = nn.Dropout(dropout)
        self.factor = factor
        self.sample_mult = sample_mult

    def _split(self, x):
        return x.view(x.size(0), x.size(1), self.n_heads, self.dh).transpose(1, 2)

    def _merge(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.n_heads * self.dh)

    def forward(self, x, attn_mask=None, return_attn=False):
        B, L, _ = x.shape
        Q = self._split(self.q(x))
        K = self._split(self.k(x))
        V = self._split(self.v(x))
        scale = 1.0 / math.sqrt(self.dh)

        u = max(1, int(self.factor * math.log(L + 1)))
        u = min(u, L)
        m = max(1, int(self.sample_mult * math.log(L + 1)))
        m = min(m, L)

        idx = torch.randint(0, L, (m,), device=x.device)
        S = torch.matmul(Q, K[:, :, idx, :].transpose(-2, -1)) * scale
        sparsity = S.max(-1).values - S.mean(-1)
        top = sparsity.topk(k=u, dim=-1).indices

        Qs = torch.gather(Q, 2, top.unsqueeze(-1).expand(-1, -1, -1, self.dh))
        Sc = torch.matmul(Qs, K.transpose(-2, -1)) * scale

        if attn_mask is not None:
            M = attn_mask.unsqueeze(0).unsqueeze(0)
            Ms = torch.gather(M.expand(B, self.n_heads, -1, -1), 2, top.unsqueeze(-1).expand(-1, -1, -1, L))
            Sc = Sc.masked_fill(Ms, float("-inf"))

        Sc = Sc.to(torch.float32)
        Sc = Sc - Sc.max(-1, keepdim=True).values
        A = torch.softmax(Sc, -1).to(V.dtype)
        A = self.ad(A)

        out_sel = torch.matmul(A, V)
        ctx = V.mean(2, keepdim=True)
        out = ctx.expand(B, self.n_heads, L, self.dh).clone()
        out.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dh), out_sel)

        out = self.o(self.od(self._merge(out)))
        return (out, (A, top, L)) if return_attn else (out, None)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1, factor=5.0, sample_mult=2.0):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, dropout, factor, sample_mult)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.d = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn=False):
        h, enc = self.attn(x, attn_mask, return_attn)
        x = self.n1(x + self.d(h))
        h = self.ff(x)
        x = self.n2(x + self.d(h))
        return x, enc

class ConvDistill(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.c = nn.Conv1d(d_model, d_model, 3, 2, 1)
        self.a = nn.GELU()
        self.d = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.d(self.a(self.c(x)))
        return x.transpose(1, 2)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.self_mha = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.cross_mha = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.d = nn.Dropout(dropout)

    def forward(self, x, mem, self_mask=None, return_attn=False):
        h, _ = self.self_mha(x, x, x, attn_mask=self_mask, need_weights=False)
        x = self.n1(x + self.d(h))
        h, attn = self.cross_mha(x, mem, mem, need_weights=True, average_attn_weights=False)
        x = self.n2(x + self.d(h))
        h = self.ff(x)
        x = self.n3(x + self.d(h))
        if return_attn:
            assoc = attn.mean(1) if (attn.dim() == 4 and attn.size(1) > 1) else attn
            return x, assoc
        return x, None

def causal_mask(L: int):
    return torch.triu(torch.ones(L, L, dtype=torch.bool), 1)

# =========================================================
# Quantile helper
# =========================================================
def qname(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"

# =========================================================
# Informer + Heads (reg + cls, quantile)
# =========================================================
class InformerMT(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=3,
        d_ff=1024,
        dropout=0.1,
        use_distill=True,
        probsparse_factor=5.0,
        sample_mult=2.0,
        d_time=0,
        use_checkpoint=True,
        horizons_idx: Tuple[int, ...] = (1, 10, 30, 60),
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        prior_mix_k: int = 3,
        L_label: int = 0,
    ):
        super().__init__()
        self.L_label = int(L_label)

        self.enc_emb = DataEmbedding(d_in, d_model, d_time, dropout)
        self.dec_emb = DataEmbedding(d_in, d_model, d_time, dropout)

        self.e_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, probsparse_factor, sample_mult)
            for _ in range(e_layers)
        ])
        self.use_distill = use_distill and (e_layers > 1)
        self.dists = nn.ModuleList([ConvDistill(d_model, dropout) for _ in range(e_layers - 1)])

        self.d_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)])

        self.q_list = tuple(float(q) for q in quantiles)
        self.q_names = tuple(qname(q) for q in self.q_list)
        self.proj_delta_q = nn.ModuleDict({name: nn.Linear(d_model, d_out) for name in self.q_names})

        self.use_checkpoint = bool(use_checkpoint)
        self.horizons_idx = tuple(sorted(set(int(k) for k in horizons_idx if int(k) >= 1)))

        self.cls_heads = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            ) for k in self.horizons_idx
        })

        self.prior_mix_k = int(prior_mix_k)
        self.prior_alpha_logits = nn.ParameterDict({
            str(k): nn.Parameter(torch.zeros(self.prior_mix_k)) for k in self.horizons_idx
        })

    def forward(self, x_enc, t_enc, x_dec, t_dec, return_assoc=False, return_enc_attn=False):
        if self.use_checkpoint and self.training:
            return_enc_attn = False

        enc = self.enc_emb(x_enc, t_enc)
        enc_attn_list = []

        for i, layer in enumerate(self.e_layers):
            ret = return_enc_attn
            if self.use_checkpoint and self.training:
                enc = torch.utils.checkpoint.checkpoint(
                    lambda e, lyr=layer: lyr(e, return_attn=False)[0],
                    enc, use_reentrant=False
                )
            else:
                enc, enc_attn = layer(enc, return_attn=ret)
                if ret:
                    enc_attn_list.append(enc_attn)

            if self.use_distill and i < len(self.dists):
                enc = self.dists[i](enc)

        dec = self.dec_emb(x_dec, t_dec)
        Ld = dec.size(1)
        mask = causal_mask(Ld).to(dec.device)

        assoc_last = None
        for j, layer in enumerate(self.d_layers):
            want_assoc = return_assoc and (j == len(self.d_layers) - 1)

            if want_assoc:
                dec, assoc_last = layer(dec, enc, self_mask=mask, return_attn=True)
                continue

            if self.use_checkpoint and self.training:
                def _run(d, lyr=layer, m=mask):
                    out, _ = lyr(d, enc, self_mask=m, return_attn=False)
                    return out
                dec = torch.utils.checkpoint.checkpoint(_run, dec, use_reentrant=False)
            else:
                dec, _ = layer(dec, enc, self_mask=mask, return_attn=False)

        y_reg_q = {name: head(dec) for name, head in self.proj_delta_q.items()}

        y_cls = {}
        for k in self.horizons_idx:
            pos = self.L_label + (k - 1)
            if pos < Ld:
                token = dec[:, pos, :]
                y_cls[str(k)] = self.cls_heads[str(k)](token).squeeze(-1)

        return y_reg_q, y_cls, assoc_last, enc_attn_list, dec

# =========================================================
# Train config
# =========================================================
@dataclass
class TrainCfg:
    L_enc: int = 96
    L_label: int = 24
    L_pred: int = 60
    hop: int = 2

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1024
    e_layers: int = 3
    d_layers: int = 3
    dropout: float = 0.10

    batch_size: int = 64
    epochs: int = 40
    patience: int = 0

    lr: float = 2e-4
    wd: float = 1e-4
    grad_clip: float = 1.0

    use_distill: bool = True
    use_revin: bool = True
    use_revin_affine: bool = False

    use_amp: bool = True
    amp_dtype: str = "bf16"  # bf16 or fp16

    autoreg: bool = True
    tf_start: float = 0.7
    tf_end: float = 0.2
    tf_warm_epochs: int = 10

    horizons: Tuple[int, ...] = (1, 10)
    lambda_cls: float = 0.2

    use_ad: bool = True
    lambda_ad: float = 0.1
    alpha_ad_logit: float = 1.0

    predict_delta: bool = True
    delta_tanh_clip: Optional[float] = 3.0

    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)

    d_time_seconds: int = 8

    use_checkpoint: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir: str = "checkpoints_multi"
    ckpt_name: str = "informer_mt_best.pt"

    prior_sigmas: Tuple[float, ...] = (8.0, 32.0, 96.0)
    prior_beta: float = 0.5

    cls_from_forecast_error: bool = True
    cls_err_pos_rate: float = 0.27
    cls_err_mode: str = "l1"  # l1 or l2

    use_point_adjust: bool = True

    use_residual_boost: bool = True
    residual_boost_alpha: float = 1.0
    residual_horizon_weight_mode: str = "inv"  # inv|exp|uniform
    residual_feature_robust: bool = True
    residual_eps: float = 1e-6

    use_event_metrics: bool = True

    debug_ad: bool = True
    debug_every: int = 200

# =========================================================
# helpers (loss terms)
# =========================================================
def _linear_sched(p0, p1, t, T):
    if T <= 1:
        return p1
    t = max(0, min(t, T - 1))
    return float(p0 + (p1 - p0) * (t / (T - 1)))

def _denorm_if_needed(t, revin, stats, use_revin):
    if use_revin and (revin is not None) and (stats is not None):
        z, _ = revin(t, mode="denorm", stats=stats)
        return z
    return t

def variance_ratio_loss(pred, target, eps=1e-8):
    vp = torch.var(pred, dim=(0, 1), unbiased=False)
    vt = torch.var(target, dim=(0, 1), unbiased=False) + eps
    ratio = vp / vt
    return torch.mean((ratio - 1.0) ** 2)

def neg_corr_loss(pred, target, eps=1e-8):
    p = pred - pred.mean(dim=(0, 1), keepdim=True)
    t = target - target.mean(dim=(0, 1), keepdim=True)
    num = (p * t).mean(dim=(0, 1))
    den = torch.sqrt((p * p).mean(dim=(0, 1)) * (t * t).mean(dim=(0, 1)) + eps)
    corr = num / (den + eps)
    return 1.0 - corr.mean()

def pinball_loss(y_pred, y_true, q: float):
    e = y_true - y_pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))

def _norm_dist(p, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    return p / (p.sum(dim=-1, keepdim=True) + eps)

def _js_divergence(p, q, dim=-1, eps=1e-8):
    p = _norm_dist(p, eps=eps)
    q = _norm_dist(q, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=dim)
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=dim)
    return 0.5 * (kl_pm + kl_qm)

# =========================================================
# Residual score helpers
# =========================================================
def _make_horizon_weights(k: int, mode: str, device):
    mode = (mode or "inv").lower()
    if k <= 0:
        return torch.empty((0,), device=device)
    idx = torch.arange(1, k + 1, device=device, dtype=torch.float32)
    if mode == "inv":
        w = 1.0 / idx
    elif mode == "exp":
        tau = max(1.0, k / 4.0)
        w = torch.exp(-(idx - 1.0) / tau)
    else:
        w = torch.ones_like(idx)
    w = w / (w.sum() + 1e-12)
    return w

def _robust_scale_abs_err(abs_err_btD: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, T, D = abs_err_btD.shape
    x = abs_err_btD.reshape(B * T, D)
    med = x.median(dim=0).values
    mad = (x - med).abs().median(dim=0).values.clamp_min(eps)
    z = (abs_err_btD - med.view(1, 1, D)) / mad.view(1, 1, D)
    return z.clamp_min(0.0)

def _residual_horizon_score(yhat_btD: torch.Tensor, y_btD: torch.Tensor, k: int, cfg) -> torch.Tensor:
    abs_err = (yhat_btD[:, :k, :] - y_btD[:, :k, :]).abs()
    if bool(getattr(cfg, "residual_feature_robust", True)):
        scaled = _robust_scale_abs_err(abs_err, eps=float(getattr(cfg, "residual_eps", 1e-6)))
    else:
        scaled = abs_err
    step_score = scaled.sum(dim=2)  # (B,k)
    w = _make_horizon_weights(k, getattr(cfg, "residual_horizon_weight_mode", "inv"), device=yhat_btD.device)
    return (step_score * w.view(1, k)).sum(dim=1)

def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean()
    sd = x.std(unbiased=False).clamp_min(eps)
    return (x - mu) / sd

# =========================================================
# Event metrics
# =========================================================
def _binary_segments(y: np.ndarray):
    segs = []
    n = len(y); i = 0
    while i < n:
        if y[i] == 1:
            j = i + 1
            while j < n and y[j] == 1:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs

def _event_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    true_segs = _binary_segments(y_true)
    pred_segs = _binary_segments(y_pred)
    if len(true_segs) == 0 and len(pred_segs) == 0:
        return 1.0, 1.0, 1.0
    if len(true_segs) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_segs) == 0:
        return 1.0, 0.0, 0.0
    hit_true = sum(1 for (a, b) in true_segs if y_pred[a:b].any())
    R = hit_true / len(true_segs)
    hit_pred = sum(1 for (a, b) in pred_segs if y_true[a:b].any())
    P = hit_pred / len(pred_segs)
    F1 = 0.0 if (P + R) == 0 else (2 * P * R / (P + R))
    return float(P), float(R), float(F1)

# =========================================================
# results.csv appender
# =========================================================
def append_results_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_row = pd.DataFrame([row])

    if not os.path.exists(path):
        new_row.to_csv(path, index=False)
        return

    try:
        old = pd.read_csv(path)
    except Exception:
        bk = path + f".broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.replace(path, bk)
        except Exception:
            pass
        new_row.to_csv(path, index=False)
        return

    all_cols = sorted(set(old.columns).union(set(new_row.columns)))
    old2 = old.reindex(columns=all_cols)
    new2 = new_row.reindex(columns=all_cols)
    out = pd.concat([old2, new2], ignore_index=True)
    out.to_csv(path, index=False)

# =========================================================
# Gaussian mixture basis (cache)
# =========================================================
_gauss_basis_cache = {}

def gaussian_mixture_basis(Lenc: int, sigmas: Tuple[float, ...], device, dtype=torch.float32):
    key = (int(Lenc), tuple(float(s) for s in sigmas), str(device), str(dtype))
    if key in _gauss_basis_cache:
        return _gauss_basis_cache[key]

    idx = torch.arange(Lenc, device=device, dtype=torch.float32)
    center = float(Lenc - 1)
    dist2 = (center - idx) ** 2
    bases = []
    for s in sigmas:
        s = max(float(s), 1e-3)
        g = torch.exp(-dist2 / (2.0 * (s ** 2)))
        g = g / (g.sum() + 1e-8)
        bases.append(g)
    B = torch.stack(bases, dim=0).to(dtype=dtype)
    _gauss_basis_cache[key] = B
    return B

def build_hybrid_prior_for_horizon(
    model: InformerMT,
    cfg: TrainCfg,
    assoc_last: torch.Tensor,   # (B, Ldec, Lenc)
    pos: int,
    horizon_k: int
):
    B, Ldec, Lenc = assoc_last.shape
    device = assoc_last.device
    dtype = assoc_last.dtype

    prior_series = _norm_dist(assoc_last[:, pos, :])

    bases = gaussian_mixture_basis(Lenc, cfg.prior_sigmas, device=device, dtype=torch.float32)
    alpha_logits = model.prior_alpha_logits[str(horizon_k)]
    alpha = torch.softmax(alpha_logits.to(torch.float32), dim=-1)
    prior_explicit_1 = (alpha.unsqueeze(0) @ bases).squeeze(0)  # (Lenc,)
    prior_explicit = prior_explicit_1.unsqueeze(0).expand(B, -1).to(dtype=dtype)
    prior_explicit = _norm_dist(prior_explicit)

    beta = float(cfg.prior_beta)
    prior_hybrid = beta * prior_explicit + (1.0 - beta) * prior_series
    prior_hybrid = _norm_dist(prior_hybrid)
    return prior_hybrid, prior_series, prior_explicit

# =========================================================
# Dataset
# =========================================================
class WindowsToInformerDatasetWithTime(Dataset):
    def __init__(self, past, future, idxs, time_all, label_len):
        self.past = past.astype(np.float32)
        self.future = future.astype(np.float32)
        self.idxs = idxs
        self.time_all = time_all.astype(np.float32) if time_all is not None else None
        self.label_len = int(label_len)

    def __len__(self):
        return self.past.shape[0]

    def __getitem__(self, i):
        s, e_p, e_f = self.idxs[i]
        x_enc = self.past[i]
        K = self.future.shape[1]

        x_dec_hist = x_enc[-self.label_len:]
        zeros = np.zeros((K, x_enc.shape[1]), np.float32)
        x_dec = np.concatenate([x_dec_hist, zeros], 0)
        y = self.future[i]

        if self.time_all is None:
            t_enc = None
            t_dec = None
        else:
            t_enc = self.time_all[s:e_p]
            t_dec = self.time_all[e_p - self.label_len:e_f]

        return (
            torch.from_numpy(x_enc),
            (torch.from_numpy(t_enc) if t_enc is not None else None),
            torch.from_numpy(x_dec),
            (torch.from_numpy(t_dec) if t_dec is not None else None),
            torch.from_numpy(y),
            torch.tensor([s, e_p, e_f], dtype=torch.long)
        )

# =========================================================
# AR decoding (Δ 누적, quantile median 사용)
# =========================================================
def decode_autoregressive(
    model, revin, cfg,
    x_enc_n, t_enc,
    x_dec_seed_n, t_dec,
    teacher_future_n=None,
    teacher_prob=0.5,
    train_mode=True
):
    model.train(mode=train_mode)
    B, Llabel, D = x_dec_seed_n.shape
    K = cfg.L_pred
    cur = x_dec_seed_n.clone()
    preds = []
    q_mid = qname(0.5)

    for k in range(K):
        zeros_k = torch.zeros(B, K - k, D, device=cur.device, dtype=cur.dtype)
        dec_in = torch.cat([cur, zeros_k], 1)
        y_reg_q, _, _, _, _ = model(x_enc_n, t_enc, dec_in, t_dec)
        delta = y_reg_q[q_mid][:, Llabel + k, :].unsqueeze(1)

        if cfg.delta_tanh_clip is not None:
            delta = torch.tanh(delta) * cfg.delta_tanh_clip

        step_n = (cur[:, -1:, :] + delta) if cfg.predict_delta else delta

        if train_mode and (teacher_future_n is not None):
            use_t = (torch.rand((), device=cur.device) < teacher_prob).item()
            step_n = teacher_future_n[:, k:k + 1, :] if use_t else step_n

        cur = torch.cat([cur, step_n], 1)
        preds.append(step_n)

    return torch.cat(preds, 1)

# =========================================================
# Train/Eval loops
# =========================================================
def train_one_epoch(model, revin, loader, opt, scaler, cfg: TrainCfg, epoch: int):
    model.train()
    use_amp = (cfg.use_amp and cuda_amp_enabled(cfg.device))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype, torch.bfloat16)

    sum_loss = 0.0
    N = 0
    q_keys = [qname(q) for q in cfg.quantiles]

    cls_pos_rate = float(getattr(cfg, "cls_err_pos_rate", 0.27))
    cls_pos_rate = min(max(cls_pos_rate, 1e-3), 0.999)

    for it, b in enumerate(loader):
        x_enc, t_enc, x_dec, t_dec, y, idxs = b
        x_enc = x_enc.to(cfg.device, non_blocking=True)
        x_dec = x_dec.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        t_enc = t_enc.to(cfg.device, non_blocking=True) if t_enc is not None else None
        t_dec = t_dec.to(cfg.device, non_blocking=True) if t_dec is not None else None

        if cfg.use_revin and (revin is not None):
            x_enc_n, stats = revin(x_enc, mode="norm")
            x_dec_n, _ = revin(x_dec, mode="norm")
            y_n, _ = revin(y, mode="norm")
        else:
            x_enc_n, x_dec_n, y_n, stats = x_enc, x_dec, y, None

        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            tfp = _linear_sched(cfg.tf_start, cfg.tf_end, epoch, cfg.tf_warm_epochs) if cfg.autoreg else 0.0

            pred_n = decode_autoregressive(
                model, revin, cfg,
                x_enc_n, t_enc,
                x_dec_n[:, :cfg.L_label, :], t_dec,
                teacher_future_n=y_n if cfg.autoreg else None,
                teacher_prob=tfp,
                train_mode=True
            )
            y_pred = _denorm_if_needed(pred_n, revin, stats, cfg.use_revin)

            dec_in_full = torch.cat([x_dec_n[:, :cfg.L_label, :], torch.zeros_like(y_n)], dim=1)
            y_reg_q, y_cls_logits, assoc_last, enc_attn_list, _ = model(
                x_enc_n, t_enc, dec_in_full, t_dec,
                return_assoc=cfg.use_ad,
                return_enc_attn=False
            )

            # Quantile regression loss
            loss_q = 0.0
            for qk in q_keys:
                delta_list = []
                for k in range(cfg.L_pred):
                    pos = cfg.L_label + k
                    dlt = y_reg_q[qk][:, pos:pos + 1, :]
                    if cfg.delta_tanh_clip is not None:
                        dlt = torch.tanh(dlt) * cfg.delta_tanh_clip
                    delta_list.append(dlt)
                delta_seq = torch.cat(delta_list, dim=1)

                hist_last = x_dec_n[:, cfg.L_label - 1:cfg.L_label, :]
                seq_n = torch.cumsum(delta_seq, dim=1) + hist_last
                y_hat = _denorm_if_needed(seq_n, revin, stats, cfg.use_revin)

                q = int(qk[1:]) / 100.0
                y_hat_f = torch.nan_to_num(y_hat.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                y_f = torch.nan_to_num(y.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                loss_q = loss_q + pinball_loss(y_hat_f, y_f, q)
            loss_q = loss_q / max(1, len(q_keys))

            # aux regression
            y_pred_f = torch.nan_to_num(y_pred.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            y_f = torch.nan_to_num(y.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            loss_reg_aux = 0.1 * variance_ratio_loss(y_pred_f, y_f) + 0.1 * neg_corr_loss(y_pred_f, y_f)

            # cls target from forecast error
            B, K, D = y.shape
            y_cls_targets = {}
            for kk in cfg.horizons:
                kk = int(kk)
                if kk <= K:
                    idx_k = kk - 1
                    if getattr(cfg, "cls_err_mode", "l1").lower() == "l2":
                        err = (y_pred[:, idx_k, :] - y[:, idx_k, :]) ** 2
                    else:
                        err = (y_pred[:, idx_k, :] - y[:, idx_k, :]).abs()
                    score = err.mean(dim=1)  # (B,)
                    thr = torch.quantile(score.detach(), 1.0 - cls_pos_rate)
                    y_cls_targets[str(kk)] = (score.detach() >= thr).float()

            # AD loss + boost (js_list 정리 유지)
            loss_ad = torch.tensor(0.0, device=cfg.device)
            ad_boost = {str(int(k)): torch.zeros(B, device=cfg.device) for k in cfg.horizons}
            js_list = []

            if cfg.use_ad and (assoc_last is not None):
                mdl = unwrap_model(model)
                for kk in cfg.horizons:
                    kk = int(kk)
                    pos = cfg.L_label + (kk - 1)
                    if pos < assoc_last.size(1):
                        prior_hybrid, _, _ = build_hybrid_prior_for_horizon(
                            mdl, cfg, assoc_last, pos=pos, horizon_k=kk
                        )
                        js_k = _js_divergence(assoc_last[:, pos, :], prior_hybrid, dim=1)  # (B,)

                        js_mean = js_k.mean()
                        js_std = js_k.std(unbiased=False).clamp_min(1e-6)
                        z = (js_k - js_mean) / js_std
                        ad_boost[str(kk)] = z
                        js_list.append(js_k.mean())

                        if getattr(cfg, "debug_ad", False) and ddp_rank0() and (it % int(getattr(cfg, "debug_every", 200)) == 0):
                            print(f"[TRAIN][DEBUG] k={kk}: js_mean={float(js_mean):.6f} js_std={float(js_std):.6f} (iter={it})")

                if len(js_list) > 0:
                    loss_ad = torch.stack(js_list, dim=0).mean()

            # residual boost
            if bool(getattr(cfg, "use_residual_boost", True)) and float(getattr(cfg, "residual_boost_alpha", 0.0)) != 0.0:
                yhat = y_reg_q.get(qname(0.5), None)
                if yhat is not None:
                    for kk in cfg.horizons:
                        kk = int(kk)
                        kkey = str(kk)
                        rs = _residual_horizon_score(yhat, y, kk, cfg)
                        rz = _zscore(rs, eps=float(getattr(cfg, "residual_eps", 1e-6)))
                        if kkey in y_cls_logits:
                            y_cls_logits[kkey] = y_cls_logits[kkey] + float(cfg.residual_boost_alpha) * rz

            # BCE loss
            loss_cls = torch.tensor(0.0, device=cfg.device)
            cnt = 0
            for kk in cfg.horizons:
                kk = int(kk)
                kkey = str(kk)
                if (kkey in y_cls_logits) and (kkey in y_cls_targets):
                    logits = y_cls_logits[kkey]
                    target = y_cls_targets[kkey]

                    if cfg.use_ad:
                        logits = logits + float(cfg.alpha_ad_logit) * ad_boost[kkey]

                    npos = target.sum().clamp_min(1.0)
                    nneg = (target.numel() - target.sum()).clamp_min(1.0)
                    pos_weight = (nneg / npos).detach()

                    loss_cls = loss_cls + nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, target)
                    cnt += 1
            if cnt > 0:
                loss_cls = loss_cls / cnt

            loss = loss_q + loss_reg_aux + float(cfg.lambda_cls) * loss_cls + (float(cfg.lambda_ad) * loss_ad if cfg.use_ad else 0.0)

        if not torch.isfinite(loss):
            if ddp_rank0():
                print("[ERR] Non-finite loss detected.")
                print("  loss_q:", float(loss_q.detach().cpu()))
                print("  loss_reg_aux:", float(loss_reg_aux.detach().cpu()))
                print("  loss_cls:", float(loss_cls.detach().cpu()))
                print("  loss_ad:", float(loss_ad.detach().cpu()))
            raise RuntimeError("Non-finite loss")

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(opt)
        scaler.update()

        bs = x_enc.size(0)
        sum_loss += (loss.item() * bs)
        N += bs

    if dist_is_init():
        t = torch.tensor([sum_loss, N], device=cfg.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss, N = t.tolist()

    return sum_loss / max(1, N)

@torch.no_grad()
def evaluate_reg_pair_horizons(model, revin, loader, cfg: TrainCfg, steps: Tuple[int, int]):
    model.eval()
    use_amp = (cfg.use_amp and cuda_amp_enabled(cfg.device))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype, torch.bfloat16)

    Y_all, P_all = [], []
    for it, b in enumerate(loader):
        x_enc, t_enc, x_dec, t_dec, y, idxs = b
        x_enc = x_enc.to(cfg.device, non_blocking=True)
        x_dec = x_dec.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        t_enc = t_enc.to(cfg.device, non_blocking=True) if t_enc is not None else None
        t_dec = t_dec.to(cfg.device, non_blocking=True) if t_dec is not None else None

        if cfg.use_revin and (revin is not None):
            x_enc_n, stats = revin(x_enc, mode="norm")
            x_dec_n, _ = revin(x_dec, mode="norm")
        else:
            x_enc_n, x_dec_n, stats = x_enc, x_dec, None

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred_n = decode_autoregressive(
                model, revin, cfg,
                x_enc_n, t_enc,
                x_dec_n[:, :cfg.L_label, :], t_dec,
                teacher_future_n=None, teacher_prob=0.0, train_mode=False
            )
            pred = _denorm_if_needed(pred_n, revin, stats, cfg.use_revin)

        Y_all.append(y.detach().cpu().numpy())
        P_all.append(pred.detach().cpu().numpy())

    if len(Y_all) == 0:
        return {
            steps[0]: {"MAE": np.nan, "RMSE": np.nan, "sMAPE%": np.nan, "MedianAE": np.nan},
            steps[1]: {"MAE": np.nan, "RMSE": np.nan, "sMAPE%": np.nan, "MedianAE": np.nan},
        }

    Y = np.concatenate(Y_all, 0)
    P = np.concatenate(P_all, 0)

    out = {}
    for k in steps:
        if k < 1 or k > cfg.L_pred:
            out[k] = {"MAE": np.nan, "RMSE": np.nan, "sMAPE%": np.nan, "MedianAE": np.nan}
            continue
        idx = k - 1
        yk = Y[:, idx, :]
        pk = P[:, idx, :]
        mae = float(np.mean(np.abs(pk - yk)))
        rmse = float(np.sqrt(np.mean((pk - yk) ** 2)))
        smape = float(np.mean(np.abs(pk - yk) / np.maximum((np.abs(pk) + np.abs(yk)) / 2, 1e-8)) * 100.0)
        medae = float(np.median(np.abs(pk - yk)))
        out[k] = {"MAE": mae, "RMSE": rmse, "sMAPE%": smape, "MedianAE": medae}
    return out

@torch.no_grad()
def predict_cls_prob_on_test(model, revin, cfg: TrainCfg, loader_test, y_test_full: np.ndarray,
                            debug: bool = True, splat: bool = False):
    """
    DDP-safe + coverage-safe
    - assoc_last만으로 hybrid prior(JS) 계산해서 AD boost 적용
    - splat=False: 해당 t에만 기록. score = sigmoid(mean_logit) 방식으로 분산 회복
    - 출력/반환: AUROC 제거, AUPRC만 유지 (라벨: PAD-AUPRC)
    """
    model.eval()
    T_test = int(len(y_test_full))
    y_true_full = y_test_full.astype(np.int64)

    prob_sum_local = {int(k): np.zeros(T_test, dtype=np.float64) for k in cfg.horizons}
    cnt_local = {int(k): np.zeros(T_test, dtype=np.int64) for k in cfg.horizons}

    mdl = unwrap_model(model)

    if debug and ddp_rank0():
        print(f"[TEST] base rate (positive ratio) = {float(np.mean(y_true_full)):.6f} (T={T_test})")
        print("[TEST] splat=False | score=sigmoid(mean_logit)")

    for it, b in enumerate(loader_test):
        x_enc, t_enc, x_dec, t_dec, y, idxs = b
        x_enc = x_enc.to(cfg.device, non_blocking=True)
        x_dec = x_dec.to(cfg.device, non_blocking=True)
        t_enc = t_enc.to(cfg.device, non_blocking=True) if t_enc is not None else None
        t_dec = t_dec.to(cfg.device, non_blocking=True) if t_dec is not None else None
        idxs_np = idxs.numpy()

        if cfg.use_revin and (revin is not None):
            x_enc_n, _ = revin(x_enc, mode="norm")
            x_dec_n, _ = revin(x_dec, mode="norm")
        else:
            x_enc_n, x_dec_n = x_enc, x_dec

        y_reg_q, y_cls_logits, assoc_last, enc_attn_list, _ = model(
            x_enc_n, t_enc, x_dec_n, t_dec,
            return_assoc=cfg.use_ad,
            return_enc_attn=False
        )

        B = x_enc.size(0)

        z_map = None
        if cfg.use_ad and (assoc_last is not None):
            z_map = {str(int(k)): torch.zeros(B, device=cfg.device) for k in cfg.horizons}
            for k in cfg.horizons:
                k = int(k); kk = str(k)
                pos = cfg.L_label + (k - 1)
                if pos < assoc_last.size(1):
                    prior_hybrid, _, _ = build_hybrid_prior_for_horizon(mdl, cfg, assoc_last, pos=pos, horizon_k=k)
                    js_k = _js_divergence(assoc_last[:, pos, :], prior_hybrid, dim=1)
                    js_mean = js_k.mean()
                    js_std = js_k.std(unbiased=False).clamp_min(1e-6)
                    z_map[kk] = (js_k - js_mean) / js_std

        for b_idx in range(B):
            s, e_p, e_f = idxs_np[b_idx].tolist()

            for k in cfg.horizons:
                k = int(k); kk = str(k)
                if kk not in y_cls_logits:
                    continue

                logits = y_cls_logits[kk][b_idx]
                if (z_map is not None) and (kk in z_map):
                    logits = logits + float(cfg.alpha_ad_logit) * z_map[kk][b_idx]
                p = float(torch.sigmoid(logits).item())

                t = e_p + (k - 1)
                if 0 <= t < T_test:
                    prob_sum_local[k][t] += p
                    cnt_local[k][t] += 1

    # DDP gather to rank0
    if dist_is_init():
        rank = dist.get_rank()
        results = {}
        for k in cfg.horizons:
            k = int(k)
            ps = torch.from_numpy(prob_sum_local[k]).to(cfg.device)
            ct = torch.from_numpy(cnt_local[k]).to(cfg.device)

            ps_sum_t, min_len_ps = _ddp_pad_allgather_sum_1d(ps)
            ct_sum_t, min_len_ct = _ddp_pad_allgather_sum_1d(ct)
            min_len = min(min_len_ps, min_len_ct)
            ps_sum_t = ps_sum_t[:min_len]
            ct_sum_t = ct_sum_t[:min_len]

            if rank == 0:
                results[k] = (ps_sum_t.cpu().numpy(), ct_sum_t.cpu().numpy())

        ddp_barrier()
        if rank != 0:
            return None

        prob_sum = {k: results[k][0] for k in results.keys()}
        cnt_map = {k: results[k][1] for k in results.keys()}
    else:
        prob_sum = prob_sum_local
        cnt_map = cnt_local

    out = {}
    base_rate = float(np.mean(y_true_full))

    for k in cfg.horizons:
        k = int(k)
        cnt = cnt_map[k]
        valid = (cnt > 0)
        n_valid = int(valid.sum())
        cover = float(n_valid / max(1, T_test))

        prob = np.full(T_test, np.nan, dtype=np.float64)
        prob[valid] = prob_sum[k][valid] / np.maximum(cnt[valid], 1)

        yv = y_true_full[valid]
        pv = prob[valid]

        if debug and ddp_rank0():
            print(f"[TEST][COVER] k={k}: valid={n_valid}/{T_test} cover={cover*100:.2f}% "
                  f"prob_mean={float(np.nanmean(pv)):.6f} prob_std={float(np.nanstd(pv)):.6f}")

        if pv.size < 10 or np.unique(yv).size < 2:
            out[k] = dict(P=float("nan"), R=float("nan"), F1=float("nan"),
                          AUPRC=float("nan"),
                          thr=float("nan"), n_valid=n_valid, cover=cover)
            continue

        thr = float(np.quantile(pv, 1.0 - base_rate))
        pred = (pv >= thr).astype(np.int64)

        # point-adjust
        pred_adj = pred.copy()
        if bool(getattr(cfg, "use_point_adjust", False)):
            n = len(yv); i = 0
            while i < n:
                if yv[i] == 1:
                    j = i + 1
                    while j < n and yv[j] == 1:
                        j += 1
                    if pred[i:j].any():
                        pred_adj[i:j] = 1
                    i = j
                else:
                    i += 1

        pred_use = pred_adj if bool(getattr(cfg, "use_point_adjust", False)) else pred

        P_raw, R_raw, F1_raw, _ = precision_recall_fscore_support(yv, pred, average="binary", zero_division=0)
        P, R, F1, _ = precision_recall_fscore_support(yv, pred_use, average="binary", zero_division=0)

        # AUPRC only (no AUROC)
        AUPRC = average_precision_score(yv, pv)

        out[k] = dict(P=float(P), R=float(R), F1=float(F1),
                      AUPRC=float(AUPRC),
                      thr=float(thr), n_valid=n_valid, cover=cover)

        if bool(getattr(cfg, "use_event_metrics", True)):
            evP, evR, evF1 = _event_metrics(yv, pred_use)
            out[k]["E_P"] = float(evP)
            out[k]["E_R"] = float(evR)
            out[k]["E_F1"] = float(evF1)

        if debug and ddp_rank0():
            if bool(getattr(cfg, "use_point_adjust", False)):
                print(f"[TEST][CLASS] k={k} thr={thr:.6f} "
                      f"ADJ(F1={F1:.4f},P={P:.4f},R={R:.4f})")
            else:
                print(f"[TEST][CLASS] k={k} thr={thr:.6f} "
                      f"F1={F1:.4f} P={P:.4f} R={R:.4f}")

    return out

# =========================================================
# Save/Load
# =========================================================
def save_checkpoint(path, model, revin, cfg: TrainCfg, epoch: int, extra: dict = None):
    if not ddp_rank0():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mdl = unwrap_model(model)
    payload = {
        "model_state": mdl.state_dict(),
        "revin_state": (revin.state_dict() if revin is not None else None),
        "cfg": cfg.__dict__,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

# =========================================================
# Time feature builder (seconds)
# =========================================================
def build_time_seconds(seconds: np.ndarray):
    s = seconds.astype(np.float64)
    som = np.mod(s, 60.0) / 60.0 * 2 * np.pi
    soh = np.mod(s, 3600.0) / 3600.0 * 2 * np.pi
    sod = np.mod(s, 86400.0) / 86400.0 * 2 * np.pi
    sow = np.mod(s, 604800.0) / 604800.0 * 2 * np.pi
    feats = np.stack([
        np.sin(som), np.cos(som),
        np.sin(soh), np.cos(soh),
        np.sin(sod), np.cos(sod),
        np.sin(sow), np.cos(sow)
    ], 1).astype(np.float32)
    return feats

# =========================================================
# scenario runner
# =========================================================
def _choose_sigmas(L_enc: int) -> Tuple[float, ...]:
    if L_enc <= 96:
        return (8.0, 32.0, 64.0)
    if L_enc <= 144:
        return (8.0, 32.0, 96.0)
    return (8.0, 32.0, 96.0)

def run_scenario(
    tag: str,
    horizons: Tuple[int, ...],
    L_enc: int,
    L_pred: int,
    base_cfg: TrainCfg,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_test: np.ndarray,
    t_train: np.ndarray,
    t_test: np.ndarray,
    prior_beta: float,
    results_csv_path: str,
    num_workers: int = 2
):
    cfg = copy.deepcopy(base_cfg)
    cfg.prior_beta = float(prior_beta)
    cfg.horizons = tuple(int(x) for x in horizons)
    cfg.L_enc = int(L_enc)
    cfg.L_pred = int(L_pred)
    cfg.L_label = cfg.L_enc // 4

    assert cfg.L_pred >= max(cfg.horizons), f"L_pred({cfg.L_pred}) must cover max horizon {max(cfg.horizons)}"
    cfg.prior_sigmas = _choose_sigmas(cfg.L_enc)

    if ddp_rank0():
        print(f"\n=== [{tag}] horizons={cfg.horizons} | L_enc={cfg.L_enc} | L_label={cfg.L_label} | L_pred={cfg.L_pred} | epochs={cfg.epochs} ===")
        print(f"[{tag}] prior_sigmas={cfg.prior_sigmas} prior_beta={cfg.prior_beta}")

    L = cfg.L_enc
    K = cfg.L_pred
    HOP = cfg.hop
    nT = len(x_train_df)

    min_need = L + K + 1
    if nT < min_need:
        raise RuntimeError(f"train too short: len={nT} need>={min_need}")

    cut = int(nT * 0.8)
    cut = max(cut, min_need)
    cut = min(cut, nT - 1)

    tr_df = x_train_df.iloc[:cut].copy()
    va_df = x_train_df.iloc[cut:].copy()

    tr_p, tr_f, tr_idx = make_forecast_windows(tr_df.values.astype(np.float32), L, K, HOP)
    va_p, va_f, va_idx = make_forecast_windows(va_df.values.astype(np.float32), L, K, HOP)
    te_p, te_f, te_idx = make_forecast_windows(x_test_df.values.astype(np.float32), L, K, HOP)

    if tr_p.shape[0] < 1 or te_p.shape[0] < 1:
        raise RuntimeError(f"not enough windows: train={tr_p.shape[0]} test={te_p.shape[0]} (L={L},K={K},hop={HOP})")

    t_tr = t_train[:cut]
    t_va = t_train[cut:]
    t_te = t_test

    ds_tr = WindowsToInformerDatasetWithTime(tr_p, tr_f, tr_idx, t_tr, cfg.L_label)
    ds_va = WindowsToInformerDatasetWithTime(va_p, va_f, va_idx, t_va, cfg.L_label)
    ds_te = WindowsToInformerDatasetWithTime(te_p, te_f, te_idx, t_te, cfg.L_label)

    sm_tr = DistributedSampler(ds_tr, shuffle=True) if dist_is_init() else None
    sm_va = DistributedSampler(ds_va, shuffle=False) if dist_is_init() else None

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=(sm_tr is None), sampler=sm_tr,
                       drop_last=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, sampler=sm_va,
                       drop_last=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, sampler=None,
                       drop_last=False, num_workers=num_workers, pin_memory=True)

    FEAT_DIM = x_train_df.shape[1]
    model = InformerMT(
        d_in=FEAT_DIM, d_out=FEAT_DIM,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        e_layers=cfg.e_layers, d_layers=cfg.d_layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout,
        use_distill=cfg.use_distill,
        d_time=cfg.d_time_seconds,
        use_checkpoint=cfg.use_checkpoint,
        horizons_idx=cfg.horizons,
        quantiles=cfg.quantiles,
        prior_mix_k=len(cfg.prior_sigmas),
        L_label=cfg.L_label
    ).to(cfg.device)

    mdl = unwrap_model(model)
    hs = sorted(int(h) for h in cfg.horizons)
    w_short = torch.tensor([0.70, 0.25, 0.05], dtype=torch.float32, device=cfg.device)
    w_mid = torch.tensor([0.20, 0.60, 0.20], dtype=torch.float32, device=cfg.device)
    w_long = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float32, device=cfg.device)

    for h in hs:
        if len(cfg.prior_sigmas) != 3:
            init = torch.ones(len(cfg.prior_sigmas), device=cfg.device) / max(1, len(cfg.prior_sigmas))
        else:
            if h == hs[0]:
                init = w_short
            elif h == hs[-1]:
                init = w_long
            else:
                init = w_mid
        init = init / init.sum()
        with torch.no_grad():
            mdl.prior_alpha_logits[str(h)].copy_(torch.log(init + 1e-8))

    revin = RevIN(FEAT_DIM, affine=cfg.use_revin_affine).to(cfg.device) if cfg.use_revin else None

    if dist_is_init():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True
        )

    if ddp_rank0():
        print(f"[{tag}] Model params={count_params(unwrap_model(model))/1e6:.2f}M | device={cfg.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and cuda_amp_enabled(cfg.device)))

    for ep in range(1, cfg.epochs + 1):
        if isinstance(dl_tr.sampler, DistributedSampler):
            dl_tr.sampler.set_epoch(ep)

        tr_loss = train_one_epoch(model, revin, dl_tr, opt, scaler, cfg, epoch=ep - 1)
        met_va = evaluate_reg_pair_horizons(model, revin, dl_va, cfg, steps=(1, max(cfg.horizons)))

        if ddp_rank0():
            a1 = met_va[1]
            ah = met_va[max(cfg.horizons)]
            print(f"[{tag}][{ep:03d}] train {tr_loss:.4f} | "
                  f"1s MAE {a1['MAE']:.4f} RMSE {a1['RMSE']:.4f} sMAPE {a1['sMAPE%']:.2f}% || "
                  f"{max(cfg.horizons)}s MAE {ah['MAE']:.4f} RMSE {ah['RMSE']:.4f} sMAPE {ah['sMAPE%']:.2f}%")

    beta_tag = f"b{int(round(cfg.prior_beta * 100)):02d}"
    save_path = os.path.join(cfg.ckpt_dir, f"{tag}_{beta_tag}_{cfg.ckpt_name}")
    save_checkpoint(save_path, model, revin, cfg, epoch=cfg.epochs)

    if ddp_rank0():
        print(f"[{tag}][{beta_tag}] prior_sigmas={cfg.prior_sigmas} prior_beta={cfg.prior_beta} CKPT saved to {save_path}")

    met_te = evaluate_reg_pair_horizons(model, revin, dl_te, cfg, steps=(1, max(cfg.horizons)))
    if ddp_rank0():
        a1 = met_te[1]
        ah = met_te[max(cfg.horizons)]
        print(f"[{tag}][TEST-REG] 1s → MAE {a1['MAE']:.4f} | RMSE {a1['RMSE']:.4f} | sMAPE {a1['sMAPE%']:.2f}% | MedAE {a1['MedianAE']:.4f}")
        print(f"[{tag}][TEST-REG] {max(cfg.horizons)}s → MAE {ah['MAE']:.4f} | RMSE {ah['RMSE']:.4f} | sMAPE {ah['sMAPE%']:.2f}% | MedAE {ah['MedianAE']:.4f}")

    cls_res = predict_cls_prob_on_test(model, revin, cfg, dl_te, y_test_full=y_test, debug=ddp_rank0(), splat=False)
    ddp_barrier()

    if ddp_rank0():
        print(f"[{tag}][CLASS HEAD] (point_adjust={bool(getattr(cfg,'use_point_adjust',False))})")
        if cls_res is None:
            print("  (cls_res is None) - unexpected on rank0")
        else:
            for k in cfg.horizons:
                k = int(k)
                r = cls_res.get(k, None)
                if r is None:
                    continue
                print(
                    f"  k={k:3d} | F1={r['F1']:.4f} | PAD-AUPRC={r['AUPRC']:.4f} | thr={r['thr']:.6f}"
                )

    if ddp_rank0():
        maxH = max(cfg.horizons)
        a1 = met_te[1]
        ah = met_te[maxH]

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "beta": float(cfg.prior_beta),
            "beta_tag": beta_tag,
            "horizons": json.dumps([int(x) for x in cfg.horizons]),
            "L_enc": int(cfg.L_enc),
            "L_pred": int(cfg.L_pred),
            "prior_sigmas": json.dumps([float(s) for s in cfg.prior_sigmas]),
            "prior_beta": float(cfg.prior_beta),

            "reg_1s_MAE": float(a1["MAE"]),
            "reg_1s_RMSE": float(a1["RMSE"]),
            "reg_1s_sMAPE": float(a1["sMAPE%"]),
            "reg_1s_MedAE": float(a1["MedianAE"]),

            f"reg_{maxH}s_MAE": float(ah["MAE"]),
            f"reg_{maxH}s_RMSE": float(ah["RMSE"]),
            f"reg_{maxH}s_sMAPE": float(ah["sMAPE%"]),
            f"reg_{maxH}s_MedAE": float(ah["MedianAE"]),
        }

        if cls_res is not None:
            for k in cfg.horizons:
                k = int(k)
                r = cls_res.get(k, None)
                row[f"cls_AUPRC_k{k}"] = float(r.get("AUPRC", np.nan)) if r is not None else np.nan
                row[f"cls_F1_k{k}"] = float(r.get("F1", np.nan)) if r is not None else np.nan
                row[f"cls_cover_k{k}"] = float(r.get("cover", np.nan)) if r is not None else np.nan

        append_results_csv(results_csv_path, row)
        print(f"[{tag}][{row['beta_tag']}] appended → {results_csv_path}")

# =========================================================
# CLI
# =========================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, default=None, help="(compat) = --train_csv")
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--test_label", type=str, default=None)

    p.add_argument("--mode", type=str, default="sweep", choices=["sweep", "single"])
    p.add_argument("--ddp", action="store_true", help="torchrun 사용 시 켜기")
    p.add_argument("--amp", action="store_true", help="AMP 켜기")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--L_enc", type=int, default=192)
    p.add_argument("--L_pred", type=int, default=60)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 60])
    p.add_argument("--prior_beta", type=float, default=0.5)
    p.add_argument("--betas", nargs="+", type=float, default=[0.3, 0.5, 0.7])

    p.add_argument("--ckpt_dir", type=str, default="checkpoints_multi")
    p.add_argument("--results_csv", type=str, default=None)

    return p.parse_args()

# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    if args.train_csv is None and args.data_path is not None:
        args.train_csv = args.data_path

    if args.train_csv is None or args.test_csv is None or args.test_label is None:
        raise RuntimeError("필수: --train_csv --test_csv --test_label (또는 --data_path + 나머지)")

    set_seed()
    _ = ddp_setup(args.ddp)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train_df = load_csv_numeric(args.train_csv)
    x_test_df = load_csv_numeric(args.test_csv)

    if not os.path.exists(args.test_label):
        raise RuntimeError("test_label.csv 필요 (0/1 라벨)")
    y_test_df = pd.read_csv(args.test_label, index_col=0)
    y_col = y_test_df.select_dtypes(include=[np.number]).columns[0]
    y_test = y_test_df[y_col].astype(int).values

    m_all_train = x_train_df.index.to_numpy(dtype=float)
    m_test = x_test_df.index.to_numpy(dtype=float)
    t_train = build_time_seconds(m_all_train)
    t_test = build_time_seconds(m_test)

    if ddp_rank0():
        print(f"train shape: {x_train_df.shape} test shape: {x_test_df.shape}")
        print(f"y_test length: {len(y_test)}")
        print("[TIME] assume 1 step = 1 sec")

    base_cfg = TrainCfg()
    base_cfg.device = device
    base_cfg.epochs = int(args.epochs)
    base_cfg.batch_size = int(args.batch_size)
    base_cfg.lr = float(args.lr)
    base_cfg.wd = float(args.wd)
    base_cfg.grad_clip = float(args.grad_clip)
    base_cfg.hop = int(args.hop)

    base_cfg.use_amp = bool(args.amp)
    base_cfg.amp_dtype = str(args.amp_dtype)

    base_cfg.ckpt_dir = str(args.ckpt_dir)
    os.makedirs(base_cfg.ckpt_dir, exist_ok=True)

    results_csv_path = args.results_csv or os.path.join(base_cfg.ckpt_dir, "results.csv")

    try:
        if args.mode == "single":
            tag = f"H{min(args.horizons)}_{max(args.horizons)}"
            run_scenario(
                tag=tag,
                horizons=tuple(int(x) for x in args.horizons),
                L_enc=int(args.L_enc),
                L_pred=int(args.L_pred),
                base_cfg=base_cfg,
                x_train_df=x_train_df,
                x_test_df=x_test_df,
                y_test=y_test,
                t_train=t_train,
                t_test=t_test,
                prior_beta=float(args.prior_beta),
                results_csv_path=results_csv_path,
                num_workers=int(args.num_workers),
            )
        else:
            betas = [float(x) for x in args.betas]
            scenarios = [
                ("H1_60", (1, 60), 192, 60),
                ("H1_30", (1, 30), 144, 30),
                ("H1_10", (1, 10), 96, 10),
            ]
            for prior_beta in betas:
                for tag, horizons, L_enc, L_pred in scenarios:
                    run_scenario(
                        tag=tag,
                        horizons=horizons,
                        L_enc=L_enc,
                        L_pred=L_pred,
                        base_cfg=base_cfg,
                        x_train_df=x_train_df,
                        x_test_df=x_test_df,
                        y_test=y_test,
                        t_train=t_train,
                        t_test=t_test,
                        prior_beta=prior_beta,
                        results_csv_path=results_csv_path,
                        num_workers=int(args.num_workers),
                    )

        ddp_barrier()
    finally:
        ddp_cleanup()

    # =====================================================
    # summary (rank0 only) - uses AUPRC only
    # =====================================================
    if ddp_rank0() and os.path.exists(results_csv_path):
        try:
            df = pd.read_csv(results_csv_path)
        except Exception:
            bk = results_csv_path + f".broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.replace(results_csv_path, bk)
                print(f"[SUMMARY] results.csv is broken. backed up to: {bk}")
            except Exception:
                print("[SUMMARY] results.csv is broken and backup failed.")
            return

        auprc_cols = [c for c in df.columns if c.startswith("cls_AUPRC_k")]
        if len(auprc_cols) == 0:
            print("[SUMMARY] No AUPRC columns found in results.csv")
            return

        print("\n================= SUMMARY: Best (max AUPRC) per horizon =================")
        for c in sorted(auprc_cols, key=lambda x: int(x.split("k")[-1])):
            k = int(c.split("k")[-1])
            d = df.dropna(subset=[c]).copy()
            if len(d) == 0:
                print(f"[BEST] k={k}: no valid rows")
                continue
            best = d.sort_values(c, ascending=False).iloc[0]
            print(
                f"[BEST] k={k}: AUPRC={best[c]:.4f} | tag={best['tag']} beta={best['beta']:.2f} "
                f"L_enc={int(best['L_enc'])} L_pred={int(best['L_pred'])} horizons={best['horizons']} "
                f"sigmas={best['prior_sigmas']}"
            )

        df["AUPRC_mean"] = df[auprc_cols].mean(axis=1, skipna=True)
        best_mean = df.sort_values("AUPRC_mean", ascending=False).iloc[0]
        print("\n================= SUMMARY: Best by mean AUPRC =================")
        print(
            f"[BEST-MEAN] meanAUPRC={best_mean['AUPRC_mean']:.4f} | tag={best_mean['tag']} beta={best_mean['beta']:.2f} "
            f"L_enc={int(best_mean['L_enc'])} L_pred={int(best_mean['L_pred'])} horizons={best_mean['horizons']} "
            f"sigmas={best_mean['prior_sigmas']}"
        )

        topN = df.sort_values("AUPRC_mean", ascending=False).head(10)[
            ["tag", "beta", "L_enc", "L_pred", "horizons", "prior_sigmas", "AUPRC_mean"] + auprc_cols
        ]
        print("\n================= TOP 10 by mean AUPRC =================")
        print(topN.to_string(index=False))

if __name__ == "__main__":
    main()

