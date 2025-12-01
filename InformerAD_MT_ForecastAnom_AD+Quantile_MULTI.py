# -*- coding: utf-8 -*-
"""
InformerAD_MT_ForecastAnom_AD+Quantile_MULTI.py
- 세 시나리오를 순차 실행:
    S1: horizons=(1,10),  L_pred=10,  L_enc=96
    S2: horizons=(1,30),  L_pred=30,  L_enc=144
    S3: horizons=(1,60),  L_pred=60,  L_enc=192
- 각 시나리오마다 epoch=20 으로 고정 학습
- 각 시나리오 종료 시, (회귀) 1초/해당 지평의 성능을 따로 계산하여 출력
- (분류) 기존 HEAD로 지평별 AD 성능도 함께 출력
"""

import os, sys, math, copy, random, time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support
)

# -----------------------------
# DDP utils
# -----------------------------
def _dist_initialized()->bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def ddp_is_available()->bool:
    return torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE","1"))>1

def ddp_setup():
    if ddp_is_available() and not _dist_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return int(os.environ.get("LOCAL_RANK",0))

def ddp_rank0()->bool:
    if not _dist_initialized(): return True
    return dist.get_rank()==0

def ddp_barrier():
    if _dist_initialized(): dist.barrier()

# -----------------------------
# misc
# -----------------------------
SEED=42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cuda_amp_enabled(device:str)->bool:
    return ('cuda' in str(device).lower()) and torch.cuda.is_available()

# -----------------------------
# IO
# -----------------------------
def load_csv_numeric(path, index_col=0):
    df = pd.read_csv(path, index_col=index_col)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols)==0: raise RuntimeError(f"{path} 숫자형 컬럼 없음")
    df = df[num_cols].copy()
    bad0 = (~np.isfinite(df.values)).sum()
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df = df.interpolate(method="linear",limit_direction="both",axis=0).ffill().bfill().fillna(0.0)
    bad1 = (~np.isfinite(df.values)).sum()
    if bad0>0 and ddp_rank0(): print(f"[clean] {path}: non-finite {bad0}개 → {bad1}개(잔여)")
    df = df.clip(lower=-1e9, upper=1e9)
    return df

# -----------------------------
# RevIN
# -----------------------------
class RevIN(nn.Module):
    def __init__(self, num_features:int, eps:float=1e-5, affine:bool=True):
        super().__init__(); self.eps=eps; self.affine=affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1,1,num_features))
            self.beta  = nn.Parameter(torch.zeros(1,1,num_features))
    def forward(self, x, mode='norm', stats=None):
        if mode=='norm':
            m=x.mean(1,keepdim=True); v=x.var(1,keepdim=True, unbiased=False)
            z=(x-m)/torch.sqrt(v+self.eps)
            if self.affine: z=z*self.gamma+self.beta
            return z,(m,v)
        elif mode=='denorm':
            m,v=stats
            if self.affine: x=(x-self.beta)/(self.gamma+1e-12)
            return x*torch.sqrt(v+self.eps)+m, None
        else: raise ValueError

# -----------------------------
# window maker
# -----------------------------
def make_forecast_windows(X: np.ndarray, L:int, K:int, hop:int=1):
    T,D=X.shape; past=[]; fut=[]; idxs=[]
    for s in range(0, T-(L+K)+1, hop):
        e_p=s+L; e_f=s+L+K
        past.append(X[s:e_p]); fut.append(X[e_p:e_f]); idxs.append((s,e_p,e_f))
    if not past: return np.empty((0,L,D)), np.empty((0,K,D)), idxs
    return np.stack(past,0), np.stack(fut,0), idxs

# -----------------------------
# Positional/Embedding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model): super().__init__(); self.d_model=d_model
    def forward(self,x):
        B,L,D=x.shape; dev=x.device; dt=x.dtype
        pos=torch.arange(L,device=dev,dtype=dt).unsqueeze(1)
        div=torch.exp(torch.arange(0,D,2,device=dev,dtype=dt)*(-math.log(10000.0)/D))
        pe=torch.zeros(L,D,device=dev,dtype=dt)
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        return x+pe.unsqueeze(0)

class DataEmbedding(nn.Module):
    def __init__(self,d_in,d_model, d_time=0,dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(d_in,d_model); self.pos=PositionalEncoding(d_model)
        self.use_time=d_time>0; self.time_proj=nn.Linear(d_time,d_model) if self.use_time else None
        self.drop=nn.Dropout(dropout)
    def forward(self,x,x_time=None):
        h=self.proj(x); h=self.pos(h)
        if self.use_time and x_time is not None: h=h+self.time_proj(x_time)
        return self.drop(h)

# -----------------------------
# Informer core (compact)
# -----------------------------
class ProbSparseSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1,factor=5.0,sample_mult=2.0):
        super().__init__(); assert d_model%n_heads==0
        self.d_model=d_model; self.n_heads=n_heads; self.dh=d_model//n_heads
        self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model)
        self.o=nn.Linear(d_model,d_model); self.ad=nn.Dropout(dropout); self.od=nn.Dropout(dropout)
        self.factor=factor; self.sample_mult=sample_mult
    def _split(self,x): return x.view(x.size(0),x.size(1),self.n_heads,self.dh).transpose(1,2)
    def _merge(self,x): return x.transpose(1,2).contiguous().view(x.size(0),x.size(2),self.n_heads*self.dh)
    def forward(self,x,attn_mask=None,return_attn=False):
        B,L,_=x.shape; Q=self._split(self.q(x)); K=self._split(self.k(x)); V=self._split(self.v(x))
        scale=1.0/math.sqrt(self.dh)
        u=max(1,int(self.factor*math.log(L+1))); u=min(u,L)
        m=max(1,int(self.sample_mult*math.log(L+1))); m=min(m,L)
        idx=torch.randint(0,L,(m,),device=x.device)
        S=torch.matmul(Q, K[:,:,idx,:].transpose(-2,-1))*scale
        sparsity=S.max(-1).values - S.mean(-1)
        top=sparsity.topk(k=u,dim=-1).indices
        Qs=torch.gather(Q,2, top.unsqueeze(-1).expand(-1,-1,-1,self.dh))
        Sc=torch.matmul(Qs, K.transpose(-2,-1))*scale
        if attn_mask is not None:
            M=attn_mask.unsqueeze(0).unsqueeze(0)
            Ms=torch.gather(M.expand(B,self.n_heads,-1,-1),2, top.unsqueeze(-1).expand(-1,-1,-1,L))
            Sc=Sc.masked_fill(Ms, float('-inf'))
        Sc=Sc.to(torch.float32); Sc=Sc-Sc.max(-1,keepdim=True).values
        A=torch.softmax(Sc,-1).to(V.dtype); A=self.ad(A)
        out_sel=torch.matmul(A, V)
        ctx=V.mean(2,keepdim=True)
        out=ctx.expand(B,self.n_heads,L,self.dh).clone()
        out.scatter_(2, top.unsqueeze(-1).expand(-1,-1,-1,self.dh), out_sel)
        out=self.o(self.od(self._merge(out)))
        return (out,(A,top,L)) if return_attn else (out,None)

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff=1024,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(d_ff,d_model),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff=1024,dropout=0.1,factor=5.0,sample_mult=2.0):
        super().__init__()
        self.attn=ProbSparseSelfAttention(d_model,n_heads,dropout,factor,sample_mult)
        self.ff=FeedForward(d_model,d_ff,dropout)
        self.n1=nn.LayerNorm(d_model); self.n2=nn.LayerNorm(d_model); self.d=nn.Dropout(dropout)
    def forward(self,x,attn_mask=None,return_attn=False):
        h,enc=self.attn(x,attn_mask,return_attn); x=self.n1(x+self.d(h))
        h=self.ff(x); x=self.n2(x+self.d(h)); return x,enc

class ConvDistill(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super().__init__(); self.c=nn.Conv1d(d_model,d_model,3,2,1); self.a=nn.GELU(); self.d=nn.Dropout(dropout)
    def forward(self,x): x=x.transpose(1,2); x=self.d(self.a(self.c(x))); return x.transpose(1,2)

class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff=1024,dropout=0.1,attn_dropout=0.1):
        super().__init__()
        self.self_mha=nn.MultiheadAttention(d_model,n_heads,dropout=attn_dropout,batch_first=True)
        self.cross_mha=nn.MultiheadAttention(d_model,n_heads,dropout=attn_dropout,batch_first=True)
        self.ff=FeedForward(d_model,d_ff,dropout)
        self.n1=nn.LayerNorm(d_model); self.n2=nn.LayerNorm(d_model); self.n3=nn.LayerNorm(d_model); self.d=nn.Dropout(dropout)
    def forward(self,x,mem,self_mask=None,return_attn=False):
        h,_=self.self_mha(x,x,x,attn_mask=self_mask,need_weights=False); x=self.n1(x+self.d(h))
        h,attn=self.cross_mha(x,mem,mem,need_weights=True,average_attn_weights=False)
        x=self.n2(x+self.d(h)); h=self.ff(x); x=self.n3(x+self.d(h))
        if return_attn:
            assoc=attn.mean(1) if (attn.dim()==4 and attn.size(1)>1) else attn
            return x,assoc
        return x,None

def causal_mask(L): return torch.triu(torch.ones(L,L,dtype=torch.bool),1)

# -----------------------------
# Quantile key helper (avoid '.' in module names)
# -----------------------------
def qname(q: float) -> str:
    return f"q{int(round(q*100)):02d}"

# -----------------------------
# Informer + Heads (reg + cls, quantile)
# -----------------------------
class InformerMT(nn.Module):
    def __init__(self,d_in,d_out,d_model=512,n_heads=8,e_layers=3,d_layers=3,d_ff=1024,dropout=0.1,
                 use_distill=True, probsparse_factor=5.0, sample_mult=2.0, d_time=0, use_checkpoint=True,
                 horizons_idx: Tuple[int,...]=(1,10,30,60), quantiles: Tuple[float,...]=(0.1,0.5,0.9)):
        super().__init__()
        self.enc_emb=DataEmbedding(d_in,d_model,d_time,dropout); self.dec_emb=DataEmbedding(d_in,d_model,d_time,dropout)
        self.e_layers=nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout,probsparse_factor,sample_mult) for _ in range(e_layers)])
        self.use_distill=use_distill and (e_layers>1)
        self.dists=nn.ModuleList([ConvDistill(d_model,dropout) for _ in range(e_layers-1)])
        self.d_layers=nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(d_layers)])
        self.q_list = tuple(float(q) for q in quantiles)
        self.q_names = tuple(qname(q) for q in self.q_list)
        self.proj_delta_q = nn.ModuleDict({ name: nn.Linear(d_model,d_out) for name in self.q_names })
        self.use_checkpoint=use_checkpoint
        self.horizons_idx = tuple(sorted(set([int(k) for k in horizons_idx if k>=1])))
        self.cls_heads = nn.ModuleDict({ str(k): nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1)
        ) for k in self.horizons_idx })

    def forward(self,x_enc,t_enc,x_dec,t_dec,return_assoc=False,return_enc_attn=False):
        enc=self.enc_emb(x_enc,t_enc); enc_attn_list=[]
        for i,layer in enumerate(self.e_layers):
            ret=return_enc_attn
            if self.use_checkpoint and self.training:
                enc=torch.utils.checkpoint.checkpoint(lambda e,lyr=layer,ret=ret: lyr(e,return_attn=ret)[0],
                                                      enc, use_reentrant=False)
            else:
                enc,enc_attn=layer(enc,return_attn=ret)
                if ret: enc_attn_list.append(enc_attn)
            if self.use_distill and i < len(self.dists):
                enc=self.dists[i](enc)
        dec=self.dec_emb(x_dec,t_dec); Ld=dec.size(1); mask=causal_mask(Ld).to(dec.device)
        assoc_last=None
        for j,layer in enumerate(self.d_layers):
            ret=return_assoc and (j==len(self.d_layers)-1)
            if self.use_checkpoint and self.training:
                def _run(d,lyr=layer,m=mask,r=ret): out,_=lyr(d,enc,self_mask=m,return_attn=r); return out
                dec=torch.utils.checkpoint.checkpoint(_run,dec,use_reentrant=False)
                if ret: dec,assoc_last=layer(dec,enc,self_mask=mask,return_attn=True)
            else:
                dec,assoc=layer(dec,enc,self_mask=mask,return_attn=ret)
                if ret: assoc_last=assoc

        y_reg_q = { name: head(dec) for name,head in self.proj_delta_q.items() }
        y_cls = {}
        for k in self.horizons_idx:
            pos = k
            if pos < Ld:
                token = dec[:, pos, :]
                y_cls[str(k)] = self.cls_heads[str(k)](token).squeeze(-1)
        return y_reg_q, y_cls, assoc_last, enc_attn_list, dec

# -----------------------------
# Train config
# -----------------------------
@dataclass
class TrainCfg:
    L_enc:int=96
    L_label:int=24
    L_pred:int=60
    hop:int=2
    d_model:int=512; n_heads:int=8; d_ff:int=1024; e_layers:int=3; d_layers:int=3
    dropout:float=0.10
    batch_size:int=64; epochs:int=40; patience:int=0
    lr:float=2e-4; wd:float=1e-4; grad_clip:float=1.0
    use_distill:bool=True; use_revin:bool=True
    use_amp:bool=True; amp_dtype:str="bf16"
    autoreg:bool=True; tf_start:float=0.7; tf_end:float=0.2; tf_warm_epochs:int=10
    horizons: Tuple[int,...] = (1,10)
    lambda_cls: float = 0.2
    use_ad: bool = True
    lambda_ad: float = 0.1
    alpha_ad_logit: float = 1.0
    predict_delta: bool = True
    delta_tanh_clip: Optional[float] = 3.0
    quantiles: Tuple[float,...]=(0.1,0.5,0.9)
    time_unit:str="seconds"; d_time_seconds:int=8; d_time_minutes:int=6
    use_checkpoint:bool=True
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    use_revin_affine: bool = False
    ckpt_dir:str="checkpoints_multi"
    ckpt_name:str="informer_mt_best.pt"

# -----------------------------
# helpers (loss terms)
# -----------------------------
def _linear_sched(p0,p1,t,T):
    if T<=1: return p1
    t=max(0,min(t,T-1)); return float(p0+(p1-p0)*(t/(T-1)))

def _denorm_if_needed(t,revin,stats,use_revin):
    if use_revin and (revin is not None) and (stats is not None):
        z,_=revin(t,mode='denorm',stats=stats); return z
    return t

def variance_ratio_loss(pred, target, eps=1e-8):
    vp = torch.var(pred, dim=(0,1), unbiased=False)
    vt = torch.var(target, dim=(0,1), unbiased=False) + eps
    ratio = vp / vt
    return torch.mean((ratio - 1.0)**2)

def neg_corr_loss(pred, target, eps=1e-8):
    p = pred - pred.mean(dim=(0,1), keepdim=True)
    t = target - target.mean(dim=(0,1), keepdim=True)
    num = (p*t).mean(dim=(0,1))
    den = torch.sqrt((p*p).mean(dim=(0,1)) * (t*t).mean(dim=(0,1)) + eps)
    corr = num / (den + eps)
    return 1.0 - corr.mean()

def pinball_loss(y_pred, y_true, q:float):
    e = y_true - y_pred
    return torch.mean(torch.maximum(q*e, (q-1)*e))

def _safe_softmax(x, dim=-1, eps=1e-8):
    x = x - x.max(dim=dim, keepdim=True).values
    p = torch.softmax(x, dim=dim)
    return torch.clamp(p, eps, 1.0)

def _js_divergence(p, q, dim=-1, eps=1e-8):
    m = 0.5*(p+q)
    kl_pm = torch.sum(p*(torch.log(p+eps)-torch.log(m+eps)), dim=dim)
    kl_qm = torch.sum(q*(torch.log(q+eps)-torch.log(m+eps)), dim=dim)
    return 0.5*(kl_pm+kl_qm)

def _enc_series_importance(enc_attn_list):
    if not enc_attn_list: return None
    A, top, Lenc = enc_attn_list[-1]
    imp = A.mean(dim=2).mean(dim=1)
    imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-8)
    return imp

def _dec_prior_importance(assoc_last, pos_slice: slice):
    if assoc_last is None: return None
    tgt = assoc_last[:, pos_slice, :]
    imp = tgt.mean(dim=1)
    imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-8)
    return imp

# -----------------------------
# Dataset
# -----------------------------
class WindowsToInformerDatasetWithTime(Dataset):
    def __init__(self,past,future,idxs,time_all,label_len):
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

# -----------------------------
# AR decoding (Δ 누적; quantile median 사용)
# -----------------------------
def decode_autoregressive(model,revin,cfg,x_enc_n,t_enc,x_dec_seed_n,t_dec,
                          teacher_future_n=None,teacher_prob=0.5,train_mode=True):
    model.train(mode=train_mode)
    B,Llabel,D=x_dec_seed_n.shape; K=cfg.L_pred
    cur=x_dec_seed_n.clone(); preds=[]
    q_mid = qname(0.5)
    for k in range(K):
        zeros_k=torch.zeros(B,K-k,D,device=cur.device,dtype=cur.dtype)
        dec_in=torch.cat([cur,zeros_k],1)
        y_reg_q, _, _, _, _ = model(x_enc_n,t_enc,dec_in,t_dec)
        delta = y_reg_q[q_mid][:,Llabel+k,:].unsqueeze(1)
        if cfg.delta_tanh_clip is not None:
            delta = torch.tanh(delta) * cfg.delta_tanh_clip
        step_n = cur[:,-1:,:] + delta if cfg.predict_delta else delta
        if train_mode and (teacher_future_n is not None):
            use_t=(torch.rand((),device=cur.device)<teacher_prob).item()
            step_n = teacher_future_n[:,k:k+1,:] if use_t else step_n
        cur=torch.cat([cur,step_n],1); preds.append(step_n)
    return torch.cat(preds,1)

# -----------------------------
# Pseudo thresholds for train (for cls targets)
# -----------------------------
def build_train_pseudo_labels(train_df: pd.DataFrame, horizons: Tuple[int,...], p_upper:float=0.98, p_lower:float=0.02)->Dict[str, pd.Series]:
    upper_thr = train_df.quantile(p_upper)
    lower_thr = train_df.quantile(p_lower)
    return {'upper': upper_thr, 'lower': lower_thr}

# -----------------------------
# Train/Eval loops
# -----------------------------
def train_one_epoch(model,revin,loader,opt,scaler,cfg,crit_cls,epoch:int, thr_train):
    model.train(); use_amp=(cfg.use_amp and cuda_amp_enabled(cfg.device))
    amp_dtype={'bf16':torch.bfloat16,'fp16':torch.float16}.get(cfg.amp_dtype,torch.bfloat16)
    sum_loss=0.0; N=0
    q_keys=[qname(q) for q in cfg.quantiles]

    for it,b in enumerate(loader):
        x_enc,t_enc,x_dec,t_dec,y,idxs=b
        x_enc=x_enc.to(cfg.device,non_blocking=True); x_dec=x_dec.to(cfg.device,non_blocking=True)
        y=y.to(cfg.device,non_blocking=True)
        t_enc=t_enc.to(cfg.device,non_blocking=True) if t_enc is not None else None
        t_dec=t_dec.to(cfg.device,non_blocking=True) if t_dec is not None else None

        if cfg.use_revin and (revin is not None):
            x_enc_n,stats=revin(x_enc,mode='norm'); x_dec_n,_=revin(x_dec,mode='norm'); y_n,_=revin(y,mode='norm')
        else:
            x_enc_n,x_dec_n,y_n,stats=x_enc,x_dec,y,None

        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=amp_dtype,enabled=use_amp):
            tfp=_linear_sched(cfg.tf_start,cfg.tf_end,epoch,cfg.tf_warm_epochs) if cfg.autoreg else 0.0
            pred_n=decode_autoregressive(model,revin,cfg,x_enc_n,t_enc,x_dec_n[:,:cfg.L_label,:],t_dec,
                                         teacher_future_n=y_n if cfg.autoreg else None,teacher_prob=tfp,train_mode=True)
            y_pred = _denorm_if_needed(pred_n,revin,stats,cfg.use_revin)

            dec_in_full = torch.cat([x_dec_n[:,:cfg.L_label,:], torch.zeros_like(y_n)], dim=1)
            y_reg_q, y_cls_logits, assoc_last, enc_attn_list, _ = model(
                x_enc_n,t_enc,dec_in_full,t_dec, return_assoc=cfg.use_ad, return_enc_attn=cfg.use_ad
            )
            loss_q = 0.0
            for qk in q_keys:
                delta_n = []
                for k in range(cfg.L_pred):
                    pos = cfg.L_label + k
                    delta_n.append(y_reg_q[qk][:,pos:pos+1,:])
                delta_n = torch.cat(delta_n, dim=1)
                hist_last = x_dec_n[:,cfg.L_label-1:cfg.L_label,:]
                seq_n = torch.cumsum(delta_n, dim=1) + hist_last
                y_hat = _denorm_if_needed(seq_n, revin, stats, cfg.use_revin)
                q = int(qk[1:]) / 100.0
                loss_q = loss_q + pinball_loss(y_hat, y, q)
            loss_q = loss_q / max(1,len(q_keys))

            loss_reg_aux = 0.1*variance_ratio_loss(y_pred, y) + 0.1*neg_corr_loss(y_pred, y)

            B,K,D = y.shape
            y_cls_targets = {}
            up = thr_train['upper'].values.reshape(1,D)
            lo = thr_train['lower'].values.reshape(1,D)
            for k in cfg.horizons:
                if k<=K:
                    yy = y[:, k-1, :].detach().cpu().numpy()
                    target = ((yy>=up).any(1) | (yy<=lo).any(1)).astype(np.float32)
                    y_cls_targets[str(k)] = torch.from_numpy(target).to(y.device)

            loss_ad = torch.tensor(0.0, device=y.device)
            ad_boost = { str(k): torch.zeros(B, device=y.device) for k in cfg.horizons }
            if cfg.use_ad and (assoc_last is not None) and (enc_attn_list is not None and len(enc_attn_list)>0):
                series_imp = _enc_series_importance(enc_attn_list)
                prior_imp_all = _dec_prior_importance(assoc_last, slice(cfg.L_label, cfg.L_label+cfg.L_pred))
                if (series_imp is not None) and (prior_imp_all is not None):
                    js_all = _js_divergence(series_imp, prior_imp_all, dim=1)
                    loss_ad = js_all.mean()
                    for k in cfg.horizons:
                        pos = cfg.L_label + (k-1)
                        if pos < assoc_last.size(1):
                            prior_k = assoc_last[:, pos, :]
                            prior_k = prior_k / (prior_k.sum(dim=1, keepdim=True)+1e-8)
                            js_k = _js_divergence(series_imp, prior_k, dim=1)
                            ad_boost[str(k)] = js_k

            loss_cls = torch.tensor(0.0, device=y.device)
            cnt=0
            for k in cfg.horizons:
                if str(k) in y_cls_logits and str(k) in y_cls_targets:
                    logits = y_cls_logits[str(k)]
                    if cfg.use_ad:
                        z = ad_boost[str(k)]
                        z = z - z.mean()
                        logits = logits + cfg.alpha_ad_logit * z
                    target = y_cls_targets[str(k)]
                    loss_cls += nn.BCEWithLogitsLoss()(logits, target)
                    cnt+=1
            if cnt>0: loss_cls = loss_cls / cnt

            loss = loss_q + loss_reg_aux + cfg.lambda_cls * loss_cls + (cfg.lambda_ad * loss_ad if cfg.use_ad else 0.0)

        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()
        bs = x_enc.size(0); sum_loss += (loss.item()*bs); N += bs

    if _dist_initialized():
        t=torch.tensor([sum_loss, N], device=cfg.device, dtype=torch.float64); dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss, N = t.tolist()
    return sum_loss/max(1,N)

@torch.no_grad()
def evaluate_reg_pair_horizons(model,revin,loader,cfg, steps: Tuple[int,int]):
    """
    steps: (k1, k2) 형태. 각 스텝만 따로 집계하여 (MAE/RMSE/sMAPE/MedianAE) 리턴.
    """
    model.eval(); use_amp=(cfg.use_amp and cuda_amp_enabled(cfg.device))
    amp_dtype={'bf16':torch.bfloat16,'fp16':torch.float16}.get(cfg.amp_dtype,torch.bfloat16)
    Y_all=[]; P_all=[]
    for it,b in enumerate(loader):
        x_enc,t_enc,x_dec,t_dec,y,idxs=b
        x_enc=x_enc.to(cfg.device,non_blocking=True); x_dec=x_dec.to(cfg.device,non_blocking=True)
        y=y.to(cfg.device,non_blocking=True)
        t_enc=t_enc.to(cfg.device,non_blocking=True) if t_enc is not None else None
        t_dec=t_dec.to(cfg.device,non_blocking=True) if t_dec is not None else None
        if cfg.use_revin and (revin is not None):
            x_enc_n,stats=revin(x_enc,mode='norm'); x_dec_n,_=revin(x_dec,mode='norm'); y_n,_=revin(y,mode='norm')
        else:
            x_enc_n,x_dec_n,y_n,stats=x_enc,x_dec,y,None
        with torch.autocast('cuda',dtype=amp_dtype,enabled=use_amp):
            pred_n=decode_autoregressive(model,revin,cfg,x_enc_n,t_enc,x_dec_n[:,:cfg.L_label,:],t_dec,
                                         teacher_future_n=None,teacher_prob=0.0,train_mode=False)
            pred=_denorm_if_needed(pred_n,revin,stats,cfg.use_revin)
        Y_all.append(y.detach().cpu().numpy()); P_all.append(pred.detach().cpu().numpy())
    if len(Y_all)==0:
        return {steps[0]:{'MAE':np.nan,'RMSE':np.nan,'sMAPE%':np.nan,'MedianAE':np.nan},
                steps[1]:{'MAE':np.nan,'RMSE':np.nan,'sMAPE%':np.nan,'MedianAE':np.nan}}
    Y=np.concatenate(Y_all,0); P=np.concatenate(P_all,0)  # (N,K,D)

    out={}
    for k in steps:
        if k<1 or k>cfg.L_pred:  # 방어
            out[k]={'MAE':np.nan,'RMSE':np.nan,'sMAPE%':np.nan,'MedianAE':np.nan}; continue
        idx=k-1
        yk=Y[:,idx,:]; pk=P[:,idx,:]
        mae=float(np.mean(np.abs(pk-yk)))
        rmse=float(np.sqrt(np.mean((pk-yk)**2)))
        smape=float(np.mean(np.abs(pk-yk)/np.maximum((np.abs(pk)+np.abs(yk))/2,1e-8))*100.0)
        medae=float(np.median(np.abs(pk-yk)))
        out[k]={'MAE':mae,'RMSE':rmse,'sMAPE%':smape,'MedianAE':medae}
    return out

@torch.no_grad()
def predict_cls_prob_on_test(model,revin,cfg,loader_test, y_test_full: np.ndarray):
    model.eval()
    T_test = len(y_test_full)
    prob_map = {k: np.zeros(T_test, dtype=np.float64) for k in cfg.horizons}
    cnt_map  = {k: np.zeros(T_test, dtype=np.int64)   for k in cfg.horizons}

    for it,b in enumerate(loader_test):
        x_enc,t_enc,x_dec,t_dec,y,idxs=b
        x_enc=x_enc.to(cfg.device); x_dec=x_dec.to(cfg.device)
        t_enc=t_enc.to(cfg.device) if t_enc is not None else None
        t_dec=t_dec.to(cfg.device) if t_dec is not None else None
        idxs=idxs.numpy()
        if cfg.use_revin and (revin is not None):
            x_enc_n,stats=revin(x_enc,mode='norm'); x_dec_n,_=revin(x_dec,mode='norm')
        else:
            x_enc_n,x_dec_n,stats=x_enc,x_dec,None
        with torch.no_grad():
            y_reg_q, y_cls_logits, assoc_last, enc_attn_list, _ = model(
                x_enc_n,t_enc,x_dec_n,t_dec, return_assoc=cfg.use_ad, return_enc_attn=cfg.use_ad
            )
        B = x_enc.shape[0]
        if cfg.use_ad and (assoc_last is not None) and (enc_attn_list is not None and len(enc_attn_list)>0):
            series_imp = _enc_series_importance(enc_attn_list)
            for b_idx in range(B):
                s,e_p,e_f = idxs[b_idx].tolist()
                for k in cfg.horizons:
                    if str(k) in y_cls_logits:
                        pos = cfg.L_label + (k-1)
                        logits = y_cls_logits[str(k)][b_idx]
                        if pos < assoc_last.size(1):
                            prior_k = assoc_last[b_idx:b_idx+1, pos, :]
                            prior_k = prior_k / (prior_k.sum(dim=1, keepdim=True)+1e-8)
                            js_k = _js_divergence(series_imp[b_idx:b_idx+1,:], prior_k, dim=1)
                            z = js_k - js_k.mean()
                            logits = logits + cfg.alpha_ad_logit * z.squeeze(0)
                        p = torch.sigmoid(logits).item()
                        t = e_p + (k-1)
                        if t < T_test:
                            prob_map[k][t] += p
                            cnt_map[k][t]  += 1
        else:
            for b_idx in range(B):
                s,e_p,e_f = idxs[b_idx].tolist()
                for k in cfg.horizons:
                    if str(k) in y_cls_logits:
                        p = torch.sigmoid(y_cls_logits[str(k)][b_idx]).item()
                        t = e_p + (k-1)
                        if t < T_test:
                            prob_map[k][t] += p
                            cnt_map[k][t]  += 1

    for k in cfg.horizons:
        m = cnt_map[k] > 0
        prob_map[k][m] = prob_map[k][m] / np.maximum(cnt_map[k][m],1)

    results={}
    for k in cfg.horizons:
        prob = prob_map[k]
        bin0 = (prob >= 0.5).astype(int)
        y_true = y_test_full[:len(prob)].astype(int)
        def _ranges(arr):
            n=len(arr); i=0; R=[]
            while i<n:
                if arr[i]==1:
                    j=i+1
                    while j<n and arr[j]==1: j+=1
                    R.append((i,j)); i=j
                else: i+=1
            return R
        bin_adj = bin0.copy()
        for ts,te in _ranges(y_true):
            if bin0[ts:te].any():
                bin_adj[ts:te]=1
        P,R,F1,_ = precision_recall_fscore_support(y_true, bin_adj, average="binary", zero_division=0)
        try: AUROC = roc_auc_score(y_true, prob)
        except: AUROC = float('nan')
        try: AUPRC = average_precision_score(y_true, prob)
        except: AUPRC = float('nan')
        results[k] = dict(P=float(P), R=float(R), F1=float(F1), AUROC=float(AUROC), AUPRC=float(AUPRC))
    return results

# -----------------------------
# Save/Load
# -----------------------------
def save_checkpoint(path, model, revin, cfg:TrainCfg, epoch:int, extra:dict=None):
    if not ddp_rank0(): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mdl = model.module if hasattr(model,'module') else model
    payload = {
        "model_state": mdl.state_dict(),
        "revin_state": (revin.state_dict() if revin is not None else None),
        "cfg": cfg.__dict__,
        "epoch": epoch
    }
    if extra: payload.update(extra)
    torch.save(payload, path)

def load_checkpoint(path, model, revin, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    mdl = model.module if hasattr(model,'module') else model
    mdl.load_state_dict(ckpt["model_state"])
    if (revin is not None) and (ckpt.get("revin_state") is not None):
        revin.load_state_dict(ckpt["revin_state"])
    return ckpt

# -----------------------------
# One-shot runner for a scenario
# -----------------------------
def run_scenario(tag:str, horizons:Tuple[int,...], L_enc:int, L_pred:int,
                 base_cfg:TrainCfg, x_train_df:pd.DataFrame, x_test_df:pd.DataFrame,
                 y_test:np.ndarray, t_train:np.ndarray, t_test:np.ndarray):
    cfg = copy.deepcopy(base_cfg)
    cfg.horizons = horizons
    cfg.L_enc = L_enc
    cfg.L_pred = L_pred
    assert cfg.L_pred >= max(horizons), f"L_pred({cfg.L_pred}) must cover max horizon {max(horizons)}"
    if ddp_rank0():
        print(f"\n=== [{tag}] horizons={horizons} | L_enc={L_enc} | L_label={cfg.L_label} | L_pred={L_pred} | epochs={cfg.epochs} ===")

    # split train/val
    L=cfg.L_enc; K=cfg.L_pred; HOP=cfg.hop
    cut=max(L+K, int(len(x_train_df)*0.8))
    tr_df=x_train_df.iloc[:cut].copy(); va_df=x_train_df.iloc[cut:].copy()
    tr_p,tr_f,tr_idx = make_forecast_windows(tr_df.values.astype(np.float32), L,K,HOP)
    va_p,va_f,va_idx = make_forecast_windows(va_df.values.astype(np.float32), L,K,HOP)
    te_p,te_f,te_idx = make_forecast_windows(x_test_df.values.astype(np.float32), L,K,HOP)
    t_tr = t_train[:cut]; t_va=t_train[cut:]; t_te=t_test

    ds_tr=WindowsToInformerDatasetWithTime(tr_p,tr_f,tr_idx,t_tr, cfg.L_label)
    ds_va=WindowsToInformerDatasetWithTime(va_p,va_f,va_idx,t_va, cfg.L_label)
    ds_te=WindowsToInformerDatasetWithTime(te_p,te_f,te_idx,t_te, cfg.L_label)

    sm_tr=DistributedSampler(ds_tr,shuffle=True) if ddp_is_available() else None
    sm_va=DistributedSampler(ds_va,shuffle=False) if ddp_is_available() else None
    sm_te=DistributedSampler(ds_te,shuffle=False) if ddp_is_available() else None

    dl_tr=DataLoader(ds_tr,batch_size=cfg.batch_size,shuffle=(sm_tr is None),sampler=sm_tr,drop_last=True,num_workers=2,pin_memory=True)
    dl_va=DataLoader(ds_va,batch_size=cfg.batch_size,shuffle=False,sampler=sm_va,drop_last=False,num_workers=2,pin_memory=True)
    dl_te=DataLoader(ds_te,batch_size=cfg.batch_size,shuffle=False,sampler=sm_te,drop_last=False,num_workers=2,pin_memory=True)

    FEAT_DIM=x_train_df.shape[1]
    model=InformerMT(d_in=FEAT_DIM,d_out=FEAT_DIM,
                     d_model=cfg.d_model,n_heads=cfg.n_heads,e_layers=cfg.e_layers,d_layers=cfg.d_layers,
                     d_ff=cfg.d_ff,dropout=cfg.dropout,use_distill=cfg.use_distill, d_time=8,
                     use_checkpoint=cfg.use_checkpoint, horizons_idx=cfg.horizons, quantiles=cfg.quantiles).to(cfg.device)
    revin=RevIN(FEAT_DIM, affine=cfg.use_revin_affine).to(cfg.device) if cfg.use_revin else None

    if ddp_is_available():
        model=torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
        )

    if ddp_rank0():
        print(f"[{tag}] Model params={count_params(model.module if hasattr(model,'module') else model)/1e6:.2f}M | device={cfg.device}")

    # pseudo thresholds from TRAIN ONLY
    thr_train = build_train_pseudo_labels(tr_df, horizons=cfg.horizons, p_upper=0.98, p_lower=0.02)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    scaler=torch.amp.GradScaler('cuda',enabled=(cfg.use_amp and cuda_amp_enabled(cfg.device)))
    crit_cls=nn.BCEWithLogitsLoss()

    # train
    for ep in range(1, cfg.epochs+1):
        if isinstance(dl_tr.sampler, DistributedSampler): dl_tr.sampler.set_epoch(ep)
        tr_loss = train_one_epoch(model,revin,dl_tr,opt,scaler,cfg,crit_cls,epoch=ep-1, thr_train=thr_train)
        met_va = evaluate_reg_pair_horizons(model,revin,dl_va,cfg, steps=(1, max(horizons)))
        if ddp_rank0():
            a1=met_va[1]; ah=met_va[max(horizons)]
            print(f"[{tag}][{ep:03d}] train {tr_loss:.4f} | "
                  f"1s MAE {a1['MAE']:.4f} RMSE {a1['RMSE']:.4f} sMAPE {a1['sMAPE%']:.2f}% || "
                  f"{max(horizons)}s MAE {ah['MAE']:.4f} RMSE {ah['RMSE']:.4f} sMAPE {ah['sMAPE%']:.2f}%")

    # save
    save_path=os.path.join(cfg.ckpt_dir, f"{tag}_{cfg.ckpt_name}")
    save_checkpoint(save_path, model, revin, cfg, epoch=cfg.epochs)
    if ddp_rank0():
        print(f"[{tag}] CKPT saved to {save_path}")

    # --- TEST EVAL (회귀: 1초 & 목표 지평만 따로) ---
    met_te = evaluate_reg_pair_horizons(model,revin,dl_te,cfg, steps=(1, max(horizons)))
    if ddp_rank0():
        a1=met_te[1]; ah=met_te[max(horizons)]
        print(f"[{tag}][TEST-REG] 1s → MAE {a1['MAE']:.4f} | RMSE {a1['RMSE']:.4f} | sMAPE {a1['sMAPE%']:.2f}% | MedAE {a1['MedianAE']:.4f}")
        print(f"[{tag}][TEST-REG] {max(horizons)}s → MAE {ah['MAE']:.4f} | RMSE {ah['RMSE']:.4f} | sMAPE {ah['sMAPE%']:.2f}% | MedAE {ah['MedianAE']:.4f}")

    # --- TEST EVAL (분류: 지평별) ---
    if ddp_rank0():
        res = predict_cls_prob_on_test(model,revin,cfg,dl_te, y_test_full=y_test)
        print(f"[{tag}][POINT-ADJUST PER HORIZON - CLASS HEAD]")
        for k in cfg.horizons:
            r = res.get(k, None)
            if r is None: continue
            print(f"  k={k:3d} : P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f} | AUROC={r['AUROC']:.4f} AUPRC={r['AUPRC']:.4f}")

    return

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    set_seed(); local_rank=ddp_setup()
    dirpath=os.getcwd()
    train_csv=os.path.join(dirpath,"RANSynCoders/data","train.csv")
    test_csv =os.path.join(dirpath,"RANSynCoders/data","test.csv")
    test_lbl =os.path.join(dirpath,"RANSynCoders/data","test_label.csv")

    x_train_df=load_csv_numeric(train_csv)
    x_test_df =load_csv_numeric(test_csv)
    if not os.path.exists(test_lbl): raise RuntimeError("test_label.csv 필요 (0/1 라벨)")
    y_test_df=pd.read_csv(test_lbl,index_col=0)
    y_col=y_test_df.select_dtypes(include=[np.number]).columns[0]
    y_test=y_test_df[y_col].astype(int).values

    if ddp_rank0():
        print(f"train shape: {x_train_df.shape} test shape: {x_test_df.shape}")
        print(f"y_test length: {len(y_test)}")
        print("[TIME] assume 1 step = 1 sec")

    # time features (seconds)
    def build_time_seconds(seconds: np.ndarray):
        s=seconds.astype(np.float64)
        som=np.mod(s,60.0)/60.0*2*np.pi; soh=np.mod(s,3600.0)/3600.0*2*np.pi
        sod=np.mod(s,86400.0)/86400.0*2*np.pi; sow=np.mod(s,604800.0)/604800.0*2*np.pi
        feats=np.stack([np.sin(som),np.cos(som),np.sin(soh),np.cos(soh),
                        np.sin(sod),np.cos(sod),np.sin(sow),np.cos(sow)],1).astype(np.float32)
        return feats

    m_all_train=x_train_df.index.to_numpy(dtype=float)
    m_test=x_test_df.index.to_numpy(dtype=float)
    t_train=build_time_seconds(m_all_train)
    t_test =build_time_seconds(m_test)

    base_cfg=TrainCfg()
    os.makedirs(base_cfg.ckpt_dir, exist_ok=True)

    # 시나리오 정의: (tag, horizons, L_enc, L_pred)
    scenarios = [
        ("H1_10", (1,10),  96,  10),
        ("H1_30", (1,30),  144, 30),
        ("H1_60", (1,60),  192, 60),
    ]

    for tag, horizons, L_enc, L_pred in scenarios:
        run_scenario(tag, horizons, L_enc, L_pred, base_cfg,
                     x_train_df, x_test_df, y_test, t_train, t_test)

    if _dist_initialized(): dist.destroy_process_group()

