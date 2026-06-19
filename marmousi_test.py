#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
advanced_bnn_with_physics_integrated.py
- Physics-Informed BNN + Well constraints
- Auto well mapping (无需命令行调 B/C)、分层划分、井样本强制进入每批
- Physics loss 做标准化，尺度更稳
- 训练/验证可视化+预测可视化齐全（含角度道 原始 vs 合成）
"""

import os, json, math, random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy import stats as _stats
from itertools import chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset, Sampler
import numpy as np, torch, torch.nn.functional as F
from numpy.fft import rfft
from scipy.stats import norm
plt.rcParams["figure.dpi"] = 120
# === ADD: imports (放在其他 import 之后均可) ===
import torch.nn.functional as F
from numpy.fft import rfft, rfftfreq

EPS = 1e-6

# ---------------- bayesian-torch optional ----------------
try:
    from bayesian_torch.layers import LinearReparameterization, LinearFlipout
    _HAS_BAYES = True
except Exception:
    LinearReparameterization = None
    LinearFlipout = None
    _HAS_BAYES = False


# ====================== Physics ======================
class LearnableMultiFreqPhysics(nn.Module):
    """
    多频 Ricker 混合 + 角度权重（可学习），只返回“中心采样点”的合成反射系数幅度
    """

    def __init__(self, angle_degs, K=4, fmin=8.0, fmax=80.0,
                 ricker_len=100, dt=0.001, center_avg=3, eps=0.02):
        super().__init__()
        # 角度缓冲
        self.register_buffer("angle_degs", torch.tensor(angle_degs, dtype=torch.float32))
        self.K = int(K)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.dt = float(dt)
        self.ricker_len = int(ricker_len)
        self.center = self.ricker_len // 2
        self.center_avg = int(center_avg)
        self.eps = float(eps)

        # 频率混合参数
        self.freq_logits = nn.Parameter(torch.zeros(self.K))                  # [K]
        self.alpha = nn.Parameter(torch.zeros(len(angle_degs), self.K))       # [A,K]

        # 角度仿射：gamma, delta
        self.register_parameter("gamma", nn.Parameter(torch.ones(len(angle_degs))))
        self.register_parameter("delta", nn.Parameter(torch.zeros(len(angle_degs))))

        # Ricker 时间轴
        t = torch.arange(self.ricker_len, dtype=torch.float32) * self.dt - (self.ricker_len * self.dt) / 2
        self.register_buffer("t", t)  # [L]

    def ricker_bank(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: [K] (Hz)，返回 [K,L] 的 Ricker 子波
        - 全程保持在 f.device 上
        """
        device = f.device
        t = self.t.to(device)[None, :]     # [1,L]
        f = f.view(-1, 1)                  # [K,1]
        x = (torch.pi * f * t)             # [K,L]
        w = (1.0 - 2.0 * (x ** 2)) * torch.exp(-(x ** 2))
        return w                           # [K,L]

    @staticmethod
    def aki_richards_center(props_denorm: torch.Tensor,
                            angle_deg: torch.Tensor,
                            eps: float = 0.01) -> torch.Tensor:
        """
        支持：
          props_denorm: (B,3)    -> return (B,A)
          props_denorm: (B,T,3)  -> return (B,T,A)
        """
        Vp = props_denorm[..., 0:1]
        Vs = props_denorm[..., 1:2]
        Rho = props_denorm[..., 2:3]

        Vp2 = Vp * (1.0 + eps)
        Vs2 = Vs * (1.0 + eps)
        R2 = Rho * (1.0 + eps)

        Vpm = (Vp + Vp2) / 2.0
        Vsm = (Vs + Vs2) / 2.0
        Rm = (Rho + R2) / 2.0

        dVp = (Vp2 - Vp) / torch.clamp(Vpm, min=1e-6)
        dVs = (Vs2 - Vs) / torch.clamp(Vsm, min=1e-6)
        dR = (R2 - Rho) / torch.clamp(Rm, min=1e-6)

        # angle: [A] -> 广播到最后一维
        th = angle_deg * torch.pi / 180.0
        while th.dim() < props_denorm.dim():
            th = th.unsqueeze(0)
        # props=(B,3)   -> th shape becomes (1,A)
        # props=(B,T,3) -> th shape becomes (1,1,A)

        sin2 = torch.sin(th) ** 2
        tan2 = torch.tan(th) ** 2

        A0 = 0.5 * (dVp + dR)
        B0 = 0.5 * dVp - 2.0 * (Vsm / Vpm) ** 2 * (2.0 * dVs + dR)
        C0 = 0.5 * dVp

        rc = A0 + B0 * sin2 + C0 * (tan2 - sin2)
        return rc

    def forward(self, props_denorm: torch.Tensor) -> torch.Tensor:
        """
        支持：
          props_denorm: (B,3)    -> return (B,A)
          props_denorm: (B,T,3)  -> return (B,T,A)
        """
        device = props_denorm.device

        f = self.fmin + (self.fmax - self.fmin) * torch.sigmoid(self.freq_logits)  # [K]
        w = self.ricker_bank(f)  # [K,L]

        m = self.center_avg
        w_avg = w[:, self.center - m:self.center + m + 1].mean(dim=1)  # [K]

        alpha = torch.softmax(self.alpha, dim=1)  # [A,K]
        mix = torch.matmul(alpha, w_avg)  # [A]

        angle = self.angle_degs
        rc = self.aki_richards_center(props_denorm, angle, eps=self.eps)

        gamma = self.gamma
        delta = self.delta

        if props_denorm.dim() == 2:
            # rc: (B,A)
            synth = rc * mix[None, :]
            synth = gamma[None, :] * synth + delta[None, :]
            return synth

        elif props_denorm.dim() == 3:
            # rc: (B,T,A)
            synth = rc * mix.view(1, 1, -1)
            synth = gamma.view(1, 1, -1) * synth + delta.view(1, 1, -1)
            return synth

        else:
            raise ValueError(f"forward expects (B,3) or (B,T,3), got {tuple(props_denorm.shape)}")

    # ===== VAIM 风格带限“参数” =====
    def bandlimit_props(self, props_denorm: torch.Tensor) -> torch.Tensor:
        """
        使用与合成地震相同的多频 Ricker 混合，在“纵向”上对 VP/VS/RHOB 做带限平滑。

        - 输入 (B,3)：当前框架只有单点，没有真实“纵向维度”，直接返回；
        - 输入 (B,T,3)：对 T 维做 1D 卷积带限平滑；若 T 太短(<子波长度)，也直接返回。
        """
        import torch.nn.functional as F
        device = props_denorm.device

        # 情况1：当前训练大多是 (B,3)，不做卷积
        if props_denorm.dim() == 2:
            return props_denorm

        if props_denorm.dim() != 3:
            raise ValueError("bandlimit_props expects (B,3) or (B,T,3)")

        B, T, C = props_denorm.shape

        # ★★ 如果纵向样点太少（比如 T<10），没法定义合理带限，直接返回
        if T < 10:
            return props_denorm

        # --- 构造一个“有效带限子波” w_eff: (1,1,L) ---
        f = self.fmin + (self.fmax - self.fmin) * torch.sigmoid(self.freq_logits.to(device))  # (K,)
        w_bank = self.ricker_bank(f.to(device))                                                # (K, L)
        freq_w = torch.softmax(self.freq_logits.to(device), dim=0)                             # (K,)

        w_eff = (freq_w[:, None] * w_bank).sum(dim=0, keepdim=True)                            # (1, L)

        # 能量归一化，避免卷积幅值漂移
        w_eff = w_eff / (torch.sqrt(torch.sum(w_eff * w_eff, dim=-1, keepdim=True)) + 1e-12)

        w_eff = w_eff.view(1, 1, -1)                                                           # (1, 1, L)
        kernel_len = int(w_eff.shape[-1])

        # padding 选 (kernel_len-1)//2，使 T_out ≈ T
        pad = (kernel_len - 1) // 2

        x = props_denorm.permute(0, 2, 1).reshape(B * C, 1, T)                                 # (B*C, 1, T)

        # 再防一手：如果 T + 2*pad < kernel_len，说明太短，直接退回原值
        if T + 2 * pad < kernel_len:
            return props_denorm

        y = F.conv1d(x, w_eff.to(device), padding=pad)                                         # (B*C, 1, T_out)

        # 若 T_out != T，裁剪/补齐回 T
        if y.shape[-1] > T:
            y = y[..., :T]
        elif y.shape[-1] < T:
            y = F.pad(y, (0, T - y.shape[-1]))

        y = y.reshape(B, C, T).permute(0, 2, 1).contiguous()      # (B,T,3)
        return y

class PhysicsInformedLossLearnable:
    def __init__(self, physics_model, physics_weight=0.1, data_weight=1.0,
                 x_mean=None, x_std=None, standardize=True, prior_weight: float = 0.05,
                 # === 岩石物理先验相关参数 ===
                 rp_weight: float = 0.05,
                 vp_vs_bounds=(0.4, 0.7),   # Vs/Vp 合理区间（可按你工区再调）
                 rho_min: float = 1.9):     # 密度下限（g/cc，大致值）

        self.physics_model = physics_model          # 可学习正演核
        self.physics_weight = float(physics_weight) # 地震域物理一致性权重（raw）
        self.data_weight = float(data_weight)       # 这里只做记录
        self.prior_weight = float(prior_weight)     # VAIM 带限先验权重（raw）

        # 岩石物理先验
        self.rp_weight = float(rp_weight)           # 岩石物理先验 raw 权重
        self.vp_vs_bounds = tuple(vp_vs_bounds)
        self.rho_min = float(rho_min)

        self.mse = torch.nn.MSELoss()

        # 还原地震振幅用的均值 / 方差
        self.x_mean = None if x_mean is None else torch.as_tensor(x_mean, dtype=torch.float32)
        self.x_std  = None if x_std  is None else torch.as_tensor(x_std,  dtype=torch.float32)
        self.standardize = bool(standardize)

        self._debug_printed_once = False   # 控制 debug 打印频率

    # --------- 岩石物理先验：Vs/Vp & RHOB 下限 ----------

    def _rock_physics_prior(self, props_denorm: torch.Tensor) -> torch.Tensor:
        """
        简单岩石物理先验（带尺度控制）：
        - Vs/Vp 比例应在 [low, high] 之间（超出范围按“偏离比例”惩罚）
        - 密度 RHOB 不应太小 (>= rho_min)，用相对偏差惩罚
        返回值控制在 ~[0, 10] 左右，避免爆炸。
        """
        import torch
        import torch.nn.functional as F

        Vp  = props_denorm[..., 0]
        Vs  = props_denorm[..., 1]
        Rho = props_denorm[..., 2]

        # 1) Vs/Vp 比（无量纲）
        ratio = Vs / Vp.clamp_min(1e-6)
        low, high = self.vp_vs_bounds
        low  = float(low)
        high = float(high)

        # 超出区间的偏离量，按区间宽度归一化
        width = max(high - low, 1e-3)
        under = F.relu(low  - ratio) / width     # ratio < low
        over  = F.relu(ratio - high) / width     # ratio > high
        penalty_ratio = under + over             # >=0，一般不会太大

        # 2) 密度下限（用相对偏差）
        #   drho = (rho_min - Rho)/rho_min，低于 rho_min 的地方 drho>0
        rel_deficit = (self.rho_min - Rho) / max(self.rho_min, 1e-3)
        penalty_rho = F.relu(rel_deficit)        # 一般在 [0,1] 之间

        # 3) 组合 & 限幅
        rp = penalty_ratio + 0.5 * penalty_rho   # 每点 penalty
        rp = rp.mean()                           # 对 batch 取均值
        rp = torch.clamp(rp, 0.0, 10.0)          # 最多 10，防止极端 batch
        return rp


    # ------------------- 主调用 -------------------

    def __call__(self, pred, target, kl_divergence, seismic_input, denormalize_fn=None):
        import torch
        import torch.nn.functional as F

        device = pred.device

        # pred / target: (B, win, 3)
        if pred.dim() != 3 or pred.size(-1) < 3:
            raise RuntimeError(f"[PHYS] pred must be (B,win,3), got {tuple(pred.shape)}")
        if (target is not None) and (target.dim() != 3 or target.size(-1) < 3):
            raise RuntimeError(f"[PHYS] target must be (B,win,3), got {tuple(target.shape)}")

        B, W, C = pred.shape

        # ---- 0) 数据项（仅记录，不进 physics_loss）----
        if target is not None:
            data_loss = self.mse(pred, target)
        else:
            data_loss = torch.zeros((), device=device)

        # ---- 1) 反归一化得到物理参数 VP/VS/RHOB ----
        if denormalize_fn is not None:
            props_pred = denormalize_fn(pred).clone()  # (B,win,3)
            props_true = denormalize_fn(target).clone() if target is not None else None
        else:
            props_pred = pred
            props_true = target

        # ---- 2) 正演得到合成地震 ----
        # sequence版：physics_model 应支持 (B,win,3) -> (B,win,A)
        synth = self.physics_model(props_pred).clone()
        if synth.dim() != 3:
            raise RuntimeError(f"[PHYS] synth must be (B,win,A), got {tuple(synth.shape)}")

        A = int(synth.shape[-1])
        W_s = int(synth.shape[1])
        if W_s != W:
            raise RuntimeError(f"[PHYS] synth win mismatch: pred W={W}, synth W={W_s}")

        # ---- 3) 从输入地震中取“中心 line 的整条时间窗振幅”（兼容 2.5D）----
        # 目标：center_norm shape = (B, win, A)
        if seismic_input.dim() == 4:
            # seismic_input: [B, A_in, line_ctx, win]
            A_in = int(seismic_input.shape[1])
            hl = int(seismic_input.shape[2] // 2)  # center line

            # 取中心 line 的整个 win
            # -> [B, A_use, win]
            center_aw = seismic_input[:, :min(A, A_in), hl, :].contiguous()
            # -> [B, win, A_use]
            center_norm = center_aw.transpose(1, 2).contiguous()

        elif seismic_input.dim() == 3:
            # seismic_input: [B, A_in, win]
            A_in = int(seismic_input.shape[1])
            center_aw = seismic_input[:, :min(A, A_in), :].contiguous()
            center_norm = center_aw.transpose(1, 2).contiguous()  # [B, win, A_use]

        elif seismic_input.dim() == 2:
            # seismic_input: [B, A*win] or [B, A*line_ctx*win]
            if seismic_input.shape[1] % A != 0:
                raise RuntimeError(f"[PHYS] cannot reshape seismic_input flat dim={seismic_input.shape[1]} with A={A}")
            K = int(seismic_input.shape[1] // A)
            Xr = seismic_input.view(B, A, K)  # [B, A, K]
            center_norm = Xr.transpose(1, 2).contiguous()  # [B, K, A]
            if K != W:
                raise RuntimeError(f"[PHYS] flat seismic recovered K={K}, but pred win={W}")
        else:
            raise RuntimeError(f"[PHYS] unexpected seismic_input.dim={seismic_input.dim()}")

        A_use = min(int(center_norm.shape[-1]), int(A))
        center_norm = center_norm[:, :, :A_use]
        synth = synth[:, :, :A_use]

        # ---- 还原到物理振幅（与 X 的标准化口径一致）----
        if (self.x_mean is None) or (self.x_std is None):
            center = center_norm
        else:
            xm = self.x_mean.to(device)
            xs = self.x_std.to(device)

            # Scheme A: xm/xs = [A, win]
            if xm.dim() == 2 and xs.dim() == 2:
                # 截断到实际 A_use / W
                A_stat = min(int(xm.shape[0]), int(xs.shape[0]), A_use)
                W_stat = min(int(xm.shape[1]), int(xs.shape[1]), int(center_norm.shape[1]))

                center = center_norm[:, :W_stat, :A_stat] * xs[:A_stat, :W_stat].transpose(0, 1).unsqueeze(0) \
                         + xm[:A_stat, :W_stat].transpose(0, 1).unsqueeze(0)

                synth = synth[:, :W_stat, :A_stat]

            # 旧版：xm/xs = [A]
            elif xm.dim() == 1 and xs.dim() == 1:
                A_stat = min(int(xm.numel()), int(xs.numel()), A_use)

                center = center_norm[:, :, :A_stat] * xs[:A_stat].view(1, 1, -1) \
                         + xm[:A_stat].view(1, 1, -1)

                synth = synth[:, :, :A_stat]

            else:
                raise RuntimeError(f"[PHYS] unexpected x_mean/x_std shape: xm={tuple(xm.shape)}, xs={tuple(xs.shape)}")

        # ---- 4) 标准化 synth / center（保持同一套统计量）----
        if self.standardize:
            # 对 batch+time 维做统计，按角度标准化
            c_mean = center.mean(dim=(0, 1), keepdim=True).detach()  # [1,1,A]
            c_std = (center.std(dim=(0, 1), keepdim=True) + 1e-6).detach()  # [1,1,A]

            center_std = (center - c_mean) / c_std
            synth_std = (synth - c_mean) / c_std
        else:
            center_std = center
            synth_std = synth

        # 软饱和
        sat = 3.0
        center_std = sat * torch.tanh(center_std / sat)
        synth_std = sat * torch.tanh(synth_std / sat)

        # sequence版 misfit：整条 win × angle
        physics_misfit_raw = F.smooth_l1_loss(synth_std, center_std, reduction="mean", beta=1.0)
        physics_misfit = physics_misfit_raw

        # ---- 5) 带限先验 ----
        prior_loss = torch.zeros((), device=device)
        if (props_true is not None) and hasattr(self.physics_model, "bandlimit_props"):
            m_bl_pred = self.physics_model.bandlimit_props(props_pred)  # (B,win,3)
            m_bl_true = self.physics_model.bandlimit_props(props_true)

            mu = m_bl_true.mean(dim=(0, 1), keepdim=True)
            std = m_bl_true.std(dim=(0, 1), keepdim=True) + 1e-6

            m_bl_pred_n = (m_bl_pred - mu) / std
            m_bl_true_n = (m_bl_true - mu) / std

            prior_loss = F.smooth_l1_loss(m_bl_pred_n, m_bl_true_n, reduction="mean", beta=0.5)

        # ---- 6) 岩石物理先验 ----
        rp_raw = torch.zeros((), device=device)
        if self.rp_weight > 0.0:
            rp_raw = self._rock_physics_prior(props_pred)  # 已支持 [...,3]
            rp_raw = torch.clamp(rp_raw, 0.0, 10.0)

        # fallback physics loss（外层再乘 physics_weight）
        physics_loss = physics_misfit + float(self.prior_weight) * prior_loss + float(self.rp_weight) * rp_raw
        physics_loss = torch.nan_to_num(physics_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if (not self._debug_printed_once) or (torch.rand(()) < 0.001):
            print(
                f"[PHYS DEBUG] misfit_raw={physics_misfit_raw.item():.3e}, "
                f"prior={prior_loss.item():.3e}, rp_raw={rp_raw.item():.3e}, "
                f"physics_loss(fallback, no w_phys)={physics_loss.item():.3e}, "
                f"phys_w(outside)={float(self.physics_weight):.3e}, "
                f"prior_w(rel)={float(self.prior_weight):.3e}, rp_w(rel)={float(self.rp_weight):.3e}"
            )
            self._debug_printed_once = True

        return {
            "physics_loss": physics_loss,
            "physics_misfit": physics_misfit,
            "prior_loss": prior_loss,
            "rock_prior": rp_raw,
        }


# ====================== Utils ======================
import numpy as np

def _as_np_idx(idx):
    if idx is None:
        return np.array([], dtype=np.int64)
    if hasattr(idx, "cpu"):  # torch tensor
        idx = idx.cpu().numpy()
    return np.asarray(idx, dtype=np.int64).reshape(-1)

def check_split_leakage(ds, train_indices, val_indices, *, line_ctx=1, loow_lt_set=None, verbose=1):
    """
    检查两类泄漏：
      1) (l,t) 是否同时出现在 train / val（window 随机切最常见）
      2) 2.5D 邻域泄漏：line_ctx>1 时，train 是否落入 val 的同 trace 线向邻域带（会让 patch“看到”val附近信息）
    """
    train_indices = _as_np_idx(train_indices)
    val_indices   = _as_np_idx(val_indices)

    samples = np.asarray(ds.samples).astype(int)  # [N,3] = (l,t,tau)
    lt_train = samples[train_indices, :2]
    lt_val   = samples[val_indices,   :2]

    # --- 1) (l,t) group overlap ---
    train_lt_set = set(map(tuple, lt_train.tolist()))
    val_lt_set   = set(map(tuple, lt_val.tolist()))
    overlap = train_lt_set.intersection(val_lt_set)

    if verbose:
        print(f"[LEAK-CHECK] train_idx={len(train_indices)} val_idx={len(val_indices)}")
        print(f"[LEAK-CHECK] unique (l,t): train={len(train_lt_set)} val={len(val_lt_set)} overlap={len(overlap)}")

    if len(overlap) > 0:
        ex = list(overlap)[:10]
        raise RuntimeError(f"[LEAK][(l,t)-OVERLAP] train/val share same (l,t)! examples={ex}")

    # --- 2) LOOW: val 必须只包含 loow_lt_set（如果提供）---
    if loow_lt_set is not None and len(loow_lt_set) > 0:
        bad = [tuple(x) for x in lt_val.tolist() if tuple(x) not in loow_lt_set]
        if len(bad) > 0:
            raise RuntimeError(f"[LEAK][LOOW-VAL-NONLOOW] val contains non-loow (l,t)! examples={bad[:10]}")
        if verbose:
            print(f"[LEAK-CHECK][LOOW] val (l,t) all in loow_lt_set size={len(loow_lt_set)}")

    # --- 3) 2.5D 邻域泄漏防护检查 ---
    hl = int(line_ctx) // 2
    if hl > 0:
        band = 2 * hl  # 你之前的 guard 逻辑：line_ctx=7 -> hl=3 -> band=6
        lt_train_arr = lt_train.astype(int)

        # 需要一个“目标集合”：优先用 loow_lt_set，否则用 val_lt_set（通用）
        target = loow_lt_set if (loow_lt_set is not None and len(loow_lt_set) > 0) else val_lt_set

        bad2 = []
        for (lw, tw) in target:
            lw = int(lw); tw = int(tw)
            hit = np.where((lt_train_arr[:, 1] == tw) & (np.abs(lt_train_arr[:, 0] - lw) <= band))[0]
            if hit.size > 0:
                bad2.append((lw, tw, int(hit.size)))
        if len(bad2) > 0:
            raise RuntimeError(f"[LEAK][2.5D-NEIGHBOR] train has samples within ±{band} lines on same trace of VAL/LOOW! "
                               f"examples={bad2[:10]} (line_ctx={line_ctx})")

        if verbose:
            print(f"[LEAK-CHECK][2.5D] OK: no neighbor overlap within ±{band} lines on same trace (line_ctx={line_ctx}).")

    print("[LEAK-CHECK] ✅ PASS")


def check_mod_channel_health(ds, indices, name="SET", props_idx=(0,1,2)):
    """
    检查标签通道是否退化（RHOB 常数/近似全零/方差极小）。
    """
    idx = _as_np_idx(indices)
    samples = np.asarray(ds.samples).astype(int)
    l = samples[idx, 0]; t = samples[idx, 1]; tau = samples[idx, 2]

    # 取标签
    Y = ds.mod[l, t, : , tau]  # [M,C]
    # 只关心 props_idx
    Yp = Y[:, list(props_idx)].astype(np.float64)  # [M,3]
    mean = Yp.mean(axis=0)
    std  = Yp.std(axis=0)
    vmin = Yp.min(axis=0)
    vmax = Yp.max(axis=0)

    print(f"[Y-HEALTH][{name}] M={len(idx)}")
    for i, nm in enumerate(["VP","VS","RHOB"]):
        print(f"  - {nm}: mean={mean[i]:.6g} std={std[i]:.6g} min={vmin[i]:.6g} max={vmax[i]:.6g}")

    # RHOB 红线：std 非常小
    if std[2] < 1e-6:
        print(f"[Y-HEALTH][{name}][WARN] RHOB std too small ({std[2]:.3e}). "
              f"大概率：第三通道常数/全零/单位或读取错误。")
    return mean, std







def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_big_numeric(d: dict):
    best_key, best_arr, best_score = None, None, -1
    for k, v in d.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            score = int(np.prod(v.shape))
            if score > best_score:
                best_key, best_arr, best_score = k, v, score
    if best_arr is None:
        raise ValueError(".mat 里没有找到数值数组")
    return best_key, best_arr

def load_4d(path: str):
    d = sio.loadmat(path)
    key, arr = pick_big_numeric(d)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4: raise ValueError(f"{path} 主数组不是 4D，得到 {arr.shape}")
    print(f"{os.path.basename(path)}: key='{key}', shape={arr.shape}, dtype={arr.dtype}")
    return arr, key

def quick_stats(name, arr):
    arr = np.asarray(arr)
    print(f"[CHECK] {name:<12s} shape={arr.shape}  min={arr.min():.6g}  max={arr.max():.6g}  mean={arr.mean():.6g}  std={arr.std():.6g}")

def load_wells_xlsx(path, base=1, swap_lt=True):
    import pandas as pd
    T = pd.read_excel(path, header=None, engine='openpyxl')
    if T.shape[1] < 3:
        raise ValueError("井位表至少三列：井名/线号/道号")
    lines = pd.to_numeric(T.iloc[:,1], errors="coerce").values
    traces= pd.to_numeric(T.iloc[:,2], errors="coerce").values
    if swap_lt: lines, traces = traces, lines
    wells = np.stack([lines - base, traces - base], axis=1).astype(float)
    mask = np.isfinite(wells).all(axis=1)
    wells = wells[mask].astype(int)
    wells = wells[(wells[:,0] >= 0) & (wells[:,1] >= 0)]
    print(f"[INFO] Loaded {len(wells)} wells (base={base}, swap_lt={swap_lt}) from {path}")
    return wells


def _spectral_mse(x_seq, y_seq, eps=1e-8):
    if x_seq.ndim == 1: x_seq = x_seq[:, None]
    if y_seq.ndim == 1: y_seq = y_seq[:, None]
    T, C = x_seq.shape
    loss = 0.0
    for i in range(C):
        xf = np.abs(rfft(x_seq[:, i])); yf = np.abs(rfft(y_seq[:, i]))
        xf = xf / (np.linalg.norm(xf) + eps); yf = yf / (np.linalg.norm(yf) + eps)
        loss += float(np.mean((xf - yf)**2))
    return loss / C

def _grad_l1(x_seq, y_seq):
    if isinstance(x_seq, np.ndarray): x = torch.from_numpy(x_seq)
    else: x = x_seq
    if isinstance(y_seq, np.ndarray): y = torch.from_numpy(y_seq)
    else: y = y_seq
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]
    dx = x[1:] - x[:-1]; dy = y[1:] - y[:-1]
    return F.smooth_l1_loss(dx, dy)

@torch.no_grad()
def _gather_sequence_inputs(ds, l, t, device, Tlen=None):
    L, Tn, C, N = ds.stack.shape
    half = ds.win // 2
    if Tlen is None: Tlen = min(128, N - 2*half)
    start = np.random.randint(half, max(half+1, N - half - Tlen))
    idxs  = range(start, start + Tlen)

    Xs, Y = [], []
    for tau in idxs:
        X = ds.stack[l, t, ds.angles_idx, tau-half:tau+half+1]
        Xn = (X - ds.X_mean[:, None]) / ds.X_std[:, None]
        Xs.append(torch.from_numpy(Xn.reshape(-1)).float())
        Y.append(ds.mod[l, t, ds.props_idx, tau])
    X_batch = torch.stack(Xs, 0).to(device)
    Y_true  = torch.tensor(np.stack(Y, 0), dtype=torch.float32)
    return X_batch, Y_true, np.array(list(idxs))

def aux_sequence_loss_step(model, ds, device, beta_spec=0.10, beta_grad=0.05, beta_tv=1e-4,
                           Tmc=6, Tlen=128):
    L, Tn, _, _ = ds.stack.shape
    l = np.random.randint(0, L); t = np.random.randint(0, Tn)
    X_batch, Y_true, _ = _gather_sequence_inputs(ds, l, t, device, Tlen=Tlen)
    model.train()
    mus = []
    for _ in range(Tmc):
        out = model(X_batch, return_kl=False)
        mu = out[0] if isinstance(out, tuple) else out
        mus.append(mu)
    mu_n = torch.stack(mus, 0).mean(0)         # [T,3] (Z空间)
    MU   = mu_n * torch.tensor(ds.Y_std)[None,:] + torch.tensor(ds.Y_mean)[None,:]  # 物理
    L_spec = _spectral_mse(MU.detach().cpu().numpy(), Y_true.numpy())
    L_grad = _grad_l1(MU, Y_true)
    L_tv   = (MU[1:] - MU[:-1]).pow(2).mean()
    loss = beta_spec * torch.tensor(L_spec, device=device) + beta_grad * L_grad + beta_tv * L_tv
    return loss

# ====== NEW: hard-sample weights (optional) ======
def compute_trace_grad_weights(ds, k=1.5, eps=1e-6):
    # 需要 ds.indices: 每个训练样本的 (l,t,tau)
    L, Tn, C, N = ds.stack.shape
    w_list = []
    for (l, t, tau) in ds.indices:
        tau_l = max(0, tau-1); tau_r = min(N-1, tau+1)
        y_l = ds.mod[l, t, ds.props_idx, tau_l]; y_r = ds.mod[l, t, ds.props_idx, tau_r]
        g   = np.abs(y_r - y_l).mean()
        w_list.append(g)
    w = np.array(w_list); w = (w - w.min()) / (w.max() - w.min() + eps)
    w = 1.0 + k * w
    return w.astype(np.float32)

# ====== NEW: per-channel temperature scaling + probability plots ======
# ====================== Uncertainty calibration ======================
@torch.no_grad()
def fit_channel_temperature(y_true_d, mu_d, std_d, eps=1e-12):
    """
    按通道拟合温度 tau_c，使得 E[(y-μ)^2] ≈ tau_c^2 * E[σ^2]
    返回：tau (3,), 以及校准后的 std_cal
    """
    import numpy as np
    num = np.mean((y_true_d - mu_d) ** 2, axis=0)          # (3,)
    den = np.mean((std_d) ** 2, axis=0) + eps              # (3,)
    tau = np.sqrt(np.clip(num / den, a_min=eps, a_max=None))
    std_cal = std_d * tau[None, :]
    return tau, std_cal


def save_tau_yaml(tau, out_dir, fname="calibration_tau.yaml"):
    import os, yaml
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fname), "w") as f:
        yaml.safe_dump({ "tau_per_channel": [float(t) for t in tau] }, f, sort_keys=False)

def fit_temperature_per_channel(y_true_d, mu_d, std_d):
    num = np.mean((y_true_d - mu_d)**2, axis=0)
    den = np.mean(std_d**2, axis=0) + 1e-12
    return np.sqrt(num / den)  # (3,)

def apply_temperature(std_d, tau_vec):
    return std_d * tau_vec[None, :]



def gaussian_crps(y, mu, sigma):
    z = (y - mu) / (sigma + 1e-12)
    return sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))

# === ADD: Student-t NLL 小工具 ===
def student_t_nll(err: torch.Tensor,
                  logvar: torch.Tensor,
                  df: float = 4.0,
                  sigma_floor: float = 0.05,
                  clamp_logvar=(-10.0, 6.0),
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Independent Student-t NLL for heteroscedastic regression.
    err: (B,C) = (pred - y) or (y - pred) (only squared used)
    logvar: (B,C) predicted log-variance
    returns: nll elementwise (B,C)
    """
    df = float(df)
    logvar = torch.clamp(logvar, min=clamp_logvar[0], max=clamp_logvar[1])
    var = torch.exp(logvar).clamp_min(float(sigma_floor) ** 2)
    scale = torch.sqrt(var + eps)

    # log pdf: lgamma((ν+1)/2)-lgamma(ν/2)-0.5*log(νπ)-log(scale) - (ν+1)/2 * log(1 + (err/scale)^2 / ν)
    z2 = (err / scale) ** 2
    logC = (
        torch.lgamma(torch.tensor((df + 1.0) / 2.0, device=err.device, dtype=err.dtype))
        - torch.lgamma(torch.tensor(df / 2.0, device=err.device, dtype=err.dtype))
        - 0.5 * (math.log(df) + math.log(math.pi))
        - torch.log(scale)
    )
    logp = logC - 0.5 * (df + 1.0) * torch.log1p(z2 / df)
    nll = -logp
    return nll



# -------- 工程线/道 -> 体索引映射 + 自动标定 --------
def map_wells_to_volume(
    wells_raw, L, T, swap=False, one_based=True,
    line_origin=2064.0, trace_origin=1677.0,
    line_scale=451.0/(2334.0-2064.0), trace_scale=351.0/(1941.0-1677.0),
    line_offset=0.0, trace_offset=0.0, round_mode="round", clip=True,
    verbose=True   # ← 新增：控制是否打印日志
):
    wr = np.asarray(wells_raw, dtype=float)
    if wr.ndim != 2 or wr.shape[1] < 2:
        raise ValueError(f"wells_raw 维度异常：{wr.shape}")
    if wr.shape[1] > 2: wr = wr[:, -2:]
    if swap: wr = wr[:, [1, 0]]
    if one_based: wr = wr - 1.0

    finite_mask0 = np.isfinite(wr).all(axis=1)
    wr = wr[finite_mask0]
    if wr.size == 0:
        if verbose: print("[WARN] wells 全部非有限，返回空")
        return np.empty((0,2), dtype=np.int32)

    # 打印点 #1：输入范围
    if verbose:
        print(f"[INFO] wells 输入范围：line[{wr[:,0].min():.3f},{wr[:,0].max():.3f}] "
              f"trace[{wr[:,1].min():.3f},{wr[:,1].max():.3f}]")

    line_idx_f  = (wr[:,0] - line_origin)  * line_scale  + line_offset
    trace_idx_f = (wr[:,1] - trace_origin) * trace_scale + trace_offset

    if   round_mode == "floor": line_idx, trace_idx = np.floor(line_idx_f), np.floor(trace_idx_f)
    elif round_mode == "ceil":  line_idx, trace_idx = np.ceil(line_idx_f),  np.ceil(trace_idx_f)
    else:                       line_idx, trace_idx = np.rint(line_idx_f),  np.rint(trace_idx_f)

    mask_fin = np.isfinite(line_idx) & np.isfinite(trace_idx)
    line_idx, trace_idx = line_idx[mask_fin], trace_idx[mask_fin]
    if line_idx.size == 0:
        if verbose: print("[WARN] 映射结果全 NaN/Inf")
        return np.empty((0,2), dtype=np.int32)

    wells_mapped = np.stack([line_idx, trace_idx], axis=1)
    if clip:
        wells_mapped[:,0] = np.clip(wells_mapped[:,0], 0, L-1)
        wells_mapped[:,1] = np.clip(wells_mapped[:,1], 0, T-1)
        dropped_oor = 0
    else:
        in_mask = (wells_mapped[:,0] >= 0) & (wells_mapped[:,0] < L) & \
                  (wells_mapped[:,1] >= 0) & (wells_mapped[:,1] < T)
        dropped_oor = int((~in_mask).sum())
        wells_mapped = wells_mapped[in_mask]

    before = wells_mapped.shape[0]
    wells_mapped = wells_mapped.astype(np.int64)
    if wells_mapped.size > 0:
        wells_mapped = np.unique(wells_mapped, axis=0)
    dedup = before - wells_mapped.shape[0]

    # 打印点 #2～#3：映射后范围与统计
    if wells_mapped.size > 0:
        if verbose:
            print(f"[INFO] 映射后范围：line[{wells_mapped[:,0].min()},{wells_mapped[:,0].max()}] "
                  f"trace[{wells_mapped[:,1].min()},{wells_mapped[:,1].max()}]")
    else:
        if verbose: print("[WARN] 映射后无有效井点")

    if verbose:
        print(f"[INFO] 有效井点 {wells_mapped.shape[0]} "
              f"(越界移除:{dropped_oor}, 去重:{dedup}, clip={clip})")

    return wells_mapped.astype(np.int32)


def auto_calibrate_wells(
    wells_raw, L, T,
    line_origin=2064.0, trace_origin=1677.0,
    line_scale=451.0/(2334.0-2064.0), trace_scale=351.0/(1941.0-1677.0),
    line_offset=0.0, trace_offset=0.0, verbose=True, out_dir=None
):
    swaps = [False, True]; one_baseds = [True, False]; rounds = ["round","floor","ceil"]
    scale_mults = [0.9, 1.0, 1.1, 1.25]; offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]

    def score(m):
        if m.size == 0: return (-1, 10**9)
        in_mask = (m[:,0] >= 0) & (m[:,0] < L) & (m[:,1] >= 0) & (m[:,1] < T)
        uniq = np.unique(m[in_mask], axis=0).shape[0]
        return (uniq, -abs(uniq - m.shape[0]))  # 多且不越界优先

    best_s = None; best_cfg=None; wr = np.asarray(wells_raw, dtype=float)
    for sw in swaps:
        for ob in one_baseds:
            for rd in rounds:
                for sml in scale_mults:
                    for smt in scale_mults:
                        for ofl in offsets:
                            for oft in offsets:
                                # 搜索阶段（试算）：不打印
                                m = map_wells_to_volume(
                                    wr, L, T, swap=sw, one_based=ob,
                                    line_origin=line_origin, trace_origin=trace_origin,
                                    line_scale=line_scale * sml, trace_scale=trace_scale * smt,
                                    line_offset=line_offset + ofl, trace_offset=trace_offset + oft,
                                    round_mode=rd, clip=False, verbose=False
                                )

                                # 最终确定（clip=True）：打印
                                best = map_wells_to_volume(
                                    wr, L, T, swap=sw, one_based=ob,
                                    line_origin=line_origin, trace_origin=trace_origin,
                                    line_scale=line_scale * sml, trace_scale=trace_scale * smt,
                                    line_offset=line_offset + ofl, trace_offset=trace_offset + oft,
                                    round_mode=rd, clip=True, verbose=True
                                )

                                s = score(m)
                                if best_s is None or s > best_s:
                                    best_s, best_cfg = s, (sw, ob, rd, sml, smt, ofl, oft)
    if best_cfg is None:
        if verbose: print("[AUTO-WELLS] 无可用组合")
        return np.empty((0,2), dtype=np.int32), {}

    sw, ob, rd, sml, smt, ofl, oft = best_cfg
    best = map_wells_to_volume(
        wr, L, T, swap=sw, one_based=ob,
        line_origin=line_origin, trace_origin=trace_origin,
        line_scale=line_scale*sml, trace_scale=trace_scale*smt,
        line_offset=line_offset+ofl, trace_offset=trace_offset+oft,
        round_mode=rd, clip=True
    )
    params = {
        "swap": sw, "one_based": ob, "round_mode": rd,
        "line_origin": line_origin, "trace_origin": trace_origin,
        "line_scale": line_scale*sml, "trace_scale": trace_scale*smt,
        "line_offset": line_offset+ofl, "trace_offset": trace_offset+oft
    }
    if verbose:
        print(f"[AUTO-WELLS] best in-bounds wells = {best.shape[0]}")
        print(f"[AUTO-WELLS] best params: {params}")
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "wells_autocalib.json"), "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[AUTO-WELLS] 保存失败：", e)
    return best, params

def _spectral_stats_and_loss(seq_pred: torch.Tensor,
                             seq_true: torch.Tensor,
                             fs_hz: float,
                             fband=(5.0, 200.0)) -> torch.Tensor:
    """
    seq_*: (M, T, C)  —— 在 GPU 上即可；该函数全程 GPU
    返回：|centroid_pred-true| + |bw6db_pred-true| 的 L1
    """
    device = seq_pred.device
    assert seq_pred.shape == seq_true.shape, "seq_pred/seq_true 形状必须一致"
    M, T, C = seq_pred.shape

    # rFFT over time dim (dim=1)
    Sp = torch.abs(rfft(seq_pred, dim=1)) + 1e-12    # (M, T/2+1, C)
    St = torch.abs(rfft(seq_true, dim=1)) + 1e-12

    # 频率轴（GPU）
    freq = rfftfreq(T, d=1.0 / fs_hz).to(device)     # (T/2+1,)
    band = (freq >= fband[0]) & (freq <= fband[1])
    Sp = Sp[:, band, :]                               # (M, K, C)
    St = St[:, band, :]
    f  = freq[band].view(1, -1, 1)                    # (1, K, 1)

    # 频谱质心
    cen_p = (Sp * f).sum(dim=1) / (Sp.sum(dim=1) + 1e-12)  # (M, C)
    cen_t = (St * f).sum(dim=1) / (St.sum(dim=1) + 1e-12)

    # -6 dB 带宽（以 1/sqrt(2) 阈值近似）
    def _bw_6db(S, f_lin):  # S: (M,K,C), f_lin: (K,)
        M, K, C = S.shape
        thr = S.amax(dim=1, keepdim=True) / 2**0.5         # (M,1,C)
        mask = (S >= thr)                                   # (M,K,C)
        idx = torch.arange(K, device=S.device).view(1, K, 1)
        left  = (mask * idx).argmax(dim=1)                  # (M,C)
        right = (mask * (K - 1 - idx)).argmax(dim=1)
        right = (K - 1) - right
        f_flat = f_lin.view(1, K).expand(M, K)              # (M,K)
        fL = torch.gather(f_flat, 1, left.clamp(0, K - 1))
        fR = torch.gather(f_flat, 1, right.clamp(0, K - 1))
        return (fR - fL)                                    # (M,C)

    bw_p = _bw_6db(Sp, f.squeeze(0))
    bw_t = _bw_6db(St, f.squeeze(0))

    centroid_loss  = torch.mean(torch.abs(cen_p - cen_t))
    bandwidth_loss = torch.mean(torch.abs(bw_p  - bw_t))
    return centroid_loss + bandwidth_loss

def cyclical_beta(epoch_idx: int, iter_idx: int, iters_per_epoch: int,
                  beta_max: float = 5e-6, mode: str = "cosine") -> float:
    """
    余弦/三角循环 KL 退火：每个 epoch 内从 ~0 -> beta_max -> ~0.5*beta_max。
    - epoch_idx: 从 1 开始
    - iter_idx : 当前 batch 索引，从 0 开始
    - iters_per_epoch: 本 epoch 的 batch 数
    - beta_max: 你的 args.beta_kl 作为上界
    """
    if iters_per_epoch <= 1:
        return beta_max * 0.5
    phase = (iter_idx + 1) / float(iters_per_epoch)  # (0,1]
    if mode == "cosine":
        # 先上升到 1，再回落到 0.5（平滑）
        up = 0.5 * (1 - math.cos(math.pi * min(phase * 2, 1.0)))  # 0->1
        if phase <= 0.5:
            v = up
        else:
            # 从 1 回落到 0.5
            tail_phase = (phase - 0.5) / 0.5  # 0->1
            v = 1.0 - 0.5 * tail_phase
        return beta_max * float(v)
    else:
        # 线性三角：0->1->0.5
        if phase <= 0.5:
            v = phase / 0.5
        else:
            v = 1.0 - 0.5 * ((phase - 0.5) / 0.5)
        return beta_max * float(v)

# ====================== Aux losses & schedules ======================
def seq_aux_loss(pred, target, metas, w_shape=0.6):
    """
    针对同一 (line, trace) 的时间序列做辅助约束：
    - 一阶差分一致性（形状保持）
    - 二阶差分（平滑/薄层不失真时可减小权重）
    返回：标量 loss
    """
    import torch, math
    device = pred.device
    B, D = pred.shape

    # 将 batch 内样本按 (l,t) 分组并按 tau 排序
    groups = {}
    for i, m in enumerate(metas):
        # metas: (l, t, tau, is_well, gidx) —— 你上次已按此返回
        l, t, tau = int(m[0]), int(m[1]), int(m[2])
        groups.setdefault((l, t), []).append((tau, i))

    if not groups:
        return torch.tensor(0.0, device=device)

    d1_terms, d2_terms = [], []
    for _, lst in groups.items():
        if len(lst) < 3:  # 至少 3 个点才算二阶
            continue
        lst.sort(key=lambda z: z[0])  # 按 tau
        idxs = torch.tensor([j for (_, j) in lst], device=device, dtype=torch.long)

        p = torch.index_select(pred, 0, idxs).contiguous()
        y = torch.index_select(target, 0, idxs).contiguous()

        # 一阶差分
        dp = p[1:] - p[:-1]
        dy = y[1:] - y[:-1]
        d1 = torch.nn.functional.smooth_l1_loss(dp, dy, reduction='mean', beta=0.5)
        d1_terms.append(d1)

        # 二阶差分（可更弱）
        ddp = dp[1:] - dp[:-1]
        ddy = dy[1:] - dy[:-1]
        d2 = torch.nn.functional.smooth_l1_loss(ddp, ddy, reduction='mean', beta=0.5)
        d2_terms.append(d2)

    if len(d1_terms) == 0:
        return torch.tensor(0.0, device=device)

    d1_mean = torch.stack(d1_terms).mean()
    d2_mean = torch.stack(d2_terms).mean() if len(d2_terms) > 0 else pred.new_tensor(0.0)

    # 组合：一阶主导，二阶较弱
    return d1_mean + w_shape * d2_mean


def anneal_weight(epoch, *,
                  warmup=5, peak=1.0, hold=0, cool_start=None, cool_end=None, floor=0.3, mode="cos"):
    """
    通用退火曲线：warmup -> (可选)hold -> cool 到 floor
    - cool_start/cool_end: 若为 None，则不降温
    - mode: "cos" 更平滑
    """
    import math
    e = float(epoch)
    if e <= warmup:
        if mode == "cos":
            return peak * 0.5 * (1 - math.cos(math.pi * e / warmup))
        return peak * (e / warmup)

    if hold > 0 and e <= warmup + hold:
        return peak

    if (cool_start is None) or (cool_end is None) or (e < cool_start):
        return peak

    if e >= cool_end:
        return floor

    # 线性或余弦降温
    frac = (e - cool_start) / max(1.0, (cool_end - cool_start))
    if mode == "cos":
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * frac))
    return peak - (peak - floor) * frac

# === ADD: per-channel temperature module ===
class PerChannelTemp(torch.nn.Module):
    def __init__(self, C: int, init_tau: float = 1.0):
        super().__init__()
        # 原始参数存放在 Raw 空间，用 softplus+1 做正且 >=1 的约束
        init_raw = np.log(np.exp(init_tau - 1.0) - 1.0 + 1e-6)
        self.tau_raw = torch.nn.Parameter(torch.full((C,), float(init_raw)))
        # 默认先冻结，训练到一半再解冻（见 train loop）
        self.tau_raw.requires_grad_(False)

    def forward(self, sigma):  # sigma: (..., C)
        tau = F.softplus(self.tau_raw) + 1.0
        return sigma * tau, tau.detach()
# === ADD: 序列拼装 & 频谱工具 ===
def _gather_sequences_from_batch(
    pred,
    y,
    metas,
    denormalize_fn=None,
    min_len=16,
    use_diff=True,
    detach=False,
    pad_value=0.0,
    debug=False,
):
    """
    从当前 batch 按 (line, trace) 聚合，按 tau 升序凑出时间序列。

    返回:
      seq_pred, seq_true: [M, T, C]  (pad 到同长度)
      若没有任何组满足 min_len，则返回 (None, None)

    改动点：
      - denormalize_fn 允许为 None（identity）
      - 可控 detach（默认 False，允许谱/lowfreq loss 回传到 pred）
      - debug 输出 batch 内可用 trace 数量与长度分布
    """
    import torch
    import numpy as np

    # --------- 兼容 metas 为 list/tuple，且 meta 结构为 (l,t,tau,is_well) ----------
    B = int(pred.shape[0])
    C = int(pred.shape[1])

    groups = {}  # (l,t) -> [(tau, idx)]
    for i in range(B):
        m = metas[i]
        # 保险：meta 可能不是 tuple/list
        if not isinstance(m, (list, tuple)) or len(m) < 3:
            continue
        l, t, tau = int(m[0]), int(m[1]), int(m[2])
        groups.setdefault((l, t), []).append((tau, i))

    if debug:
        lens = [len(v) for v in groups.values()]
        if len(lens) == 0:
            print("[SEQ][DEBUG] no valid (l,t,tau) metas in this batch.")
        else:
            print(f"[SEQ][DEBUG] traces_in_batch={len(lens)} "
                  f"len_min={min(lens)} len_med={int(np.median(lens))} len_max={max(lens)} "
                  f"min_len={min_len}")

    # --------- denormalize_fn: None -> identity ----------
    def _denorm(z):
        if denormalize_fn is None:
            return z
        out = denormalize_fn(z)
        # 如果外部 denormalize_fn 返回了 numpy，会断梯度；这里强制转回 torch（但梯度已断）
        if torch.is_tensor(out):
            return out
        return torch.as_tensor(out, device=z.device, dtype=z.dtype)

    seq_pred_list, seq_true_list = [], []
    kept = 0
    dropped = 0

    for _, lst in groups.items():
        if len(lst) < int(min_len):
            dropped += 1
            continue

        kept += 1
        lst.sort(key=lambda z: z[0])
        idxs = [j for (_, j) in lst]

        MU = _denorm(pred[idxs])  # (T,C)
        YY = _denorm(y[idxs])     # (T,C)

        if detach:
            MU = MU.detach()
            YY = YY.detach()

        if use_diff:
            MU = MU[1:] - MU[:-1]
            YY = YY[1:] - YY[:-1]

        seq_pred_list.append(MU)
        seq_true_list.append(YY)

    if debug:
        print(f"[SEQ][DEBUG] kept_traces={kept} dropped_traces={dropped} "
              f"(reason: len < min_len)")

    if len(seq_pred_list) == 0:
        return None, None

    # --------- pad 到同长度 ----------
    T_max = max(int(s.shape[0]) for s in seq_pred_list)

    def _pad_to(s, T):
        if int(s.shape[0]) == int(T):
            return s
        pad = torch.full(
            (int(T) - int(s.shape[0]), C),
            float(pad_value),
            dtype=s.dtype,
            device=s.device
        )
        return torch.cat([s, pad], dim=0)

    seq_pred = torch.stack([_pad_to(s, T_max) for s in seq_pred_list], dim=0)  # [M,T,C]
    seq_true = torch.stack([_pad_to(s, T_max) for s in seq_true_list], dim=0)  # [M,T,C]

    return seq_pred, seq_true




def _spectral_stats_and_loss(seq_pred: torch.Tensor,
                             seq_true: torch.Tensor,
                             fs_hz: float,
                             fband: tuple = (5.0, 200.0),
                             mode: str = "centroid+bw",
                             reduce: str = "mean"):
    """
    seq_*: [M, T, C]，按时间维做 rFFT；比较频谱质心 & -6dB 带宽（只比形状）
    返回: freq_loss (scalar)
    """
    assert seq_pred.shape == seq_true.shape, "pred/true shape mismatch for spectral loss"
    M, T, C = seq_pred.shape
    # rFFT over time
    Fp = torch.fft.rfft(seq_pred, dim=1)       # [M, F, C]
    Ft = torch.fft.rfft(seq_true, dim=1)
    Pp = (Fp.real**2 + Fp.imag**2) + 1e-12
    Pt = (Ft.real**2 + Ft.imag**2) + 1e-12
    # 归一化功率谱（比较形状）
    Pp = Pp / Pp.sum(dim=1, keepdim=True)
    Pt = Pt / Pt.sum(dim=1, keepdim=True)
    # 频率轴 & 频带裁剪
    # rfftfreq: 长度 = T//2 + 1
    freq = torch.fft.rfftfreq(T, d=1.0/fs_hz).to(Pp.device)   # [F]
    fmin, fmax = fband
    band = (freq >= fmin) & (freq <= fmax)
    if band.sum() < 4:  # 带宽太窄直接跳过
        return seq_pred.new_tensor(0.0)

    Pp_b = Pp[:, band, :]   # [M,K,C]
    Pt_b = Pt[:, band, :]
    f_b  = freq[band].view(1, -1, 1)  # [1,K,1]

    # 频谱质心
    cen_p = (Pp_b * f_b).sum(dim=1) / (Pp_b.sum(dim=1) + 1e-12)  # [M,C]
    cen_t = (Pt_b * f_b).sum(dim=1) / (Pt_b.sum(dim=1) + 1e-12)

    # -6dB 带宽（相对峰值），用线性插值近似
    def _bw_6db(P):
        # P: [M,K,C] 已裁带
        M, K, C = P.shape
        # 归一化相对幅度
        peak = P.max(dim=1, keepdim=True).values  # [M,1,C]
        thr  = peak / math.sqrt(2.0)             # [M,1,C]
        # 找峰位置
        idx_peak = P.argmax(dim=1)               # [M,C]
        # 左右过阈近似（逐通道逐样本 loop，K 通常不大，开销很小）
        bw = torch.zeros(M, C, device=P.device, dtype=P.dtype)
        fb = f_b[0,:,0]  # [K]
        for m in range(M):
            for c in range(C):
                k0 = int(idx_peak[m, c].item())
                # 向左
                iL = k0
                while iL > 0 and P[m, iL, c] >= thr[m, 0, c]:
                    iL -= 1
                fL = fb[iL]
                # 向右
                iR = k0
                while iR < K-1 and P[m, iR, c] >= thr[m, 0, c]:
                    iR += 1
                fR = fb[iR]
                bw[m, c] = max(0.0, (fR - fL))
        return bw  # [M,C]

    bw_p = _bw_6db(Pp_b)
    bw_t = _bw_6db(Pt_b)

    # L1/L2 都可，这里用 L1 更鲁棒
    cen_loss = (cen_p - cen_t).abs()
    bw_loss  = (bw_p  - bw_t ).abs()
    if reduce == "mean":
        return (cen_loss.mean() + bw_loss.mean()) * 0.5
    else:
        return (cen_loss + bw_loss) * 0.5

def _safe_corrcoef_1d(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    """
    更稳的 1D NCC：
    - 自动过滤 NaN/Inf
    - 方差太小直接返回 nan（退化，不要硬算出 1.0）
    - 最终裁剪到 [-1, 1]（防数值飘）
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.size < 2 or b.size < 2:
        return float("nan")

    # 过滤非有限值（非常关键）
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    a = a[m]
    b = b[m]

    ma = a.mean()
    mb = b.mean()
    da = a - ma
    db = b - mb

    va = float(np.mean(da * da))
    vb = float(np.mean(db * db))
    if (va < float(eps)) or (vb < float(eps)):
        return float("nan")

    cov = float(np.mean(da * db))
    ncc = cov / (np.sqrt(va * vb) + float(eps))

    # 防止数值误差略微超界
    if not np.isfinite(ncc):
        return float("nan")
    ncc = float(np.clip(ncc, -1.0, 1.0))
    return ncc


def _build_seq_dict_from_batch(pred, y, metas, denorm_t, min_len=16, use_diff=False):
    """
    聚合 batch 得到 dict[(l,t)] -> (taus, MU, YY)  (numpy)
    pred/y: torch [B,3], metas: list of (l,t,tau,is_well)
    denorm_t: torch版反归一化函数 (tensor->tensor)
    """
    groups = {}
    B = pred.shape[0]
    for i in range(B):
        l, t, tau, _ = metas[i]
        key = (int(l), int(t))
        groups.setdefault(key, []).append((int(tau), i))

    out = {}
    for key, lst in groups.items():
        if len(lst) < min_len:
            continue
        lst.sort(key=lambda z: z[0])
        taus = np.array([tt for (tt, _) in lst], dtype=np.int32)
        idxs = [j for (_, j) in lst]

        MU = denorm_t(pred[idxs]).detach().cpu().numpy()  # (T,3)
        YY = denorm_t(y[idxs]).detach().cpu().numpy()

        if use_diff:
            MU = MU[1:] - MU[:-1]
            YY = YY[1:] - YY[:-1]
            taus = taus[1:]

        out[key] = (taus, MU, YY)
    return out
def sanity_corr_inputs(a, b, name="CONT"):
    """
    连续性/NCC 输入健检：
    - 展平
    - 过滤 NaN/Inf
    - 打印方差、长度、是否同一对象（同一块内存/引用）
    - 方差过小给出警告（NCC 易退化为 nan 或假 1.0）
    """
    import numpy as np

    a0 = np.asarray(a)
    b0 = np.asarray(b)

    # 展平（用 float64 稳一点）
    a = np.asarray(a0, dtype=np.float64).reshape(-1)
    b = np.asarray(b0, dtype=np.float64).reshape(-1)

    if a.size == 0 or b.size == 0:
        print(f"[{name}][WARN] empty arrays (a.size={a.size}, b.size={b.size})")
        return

    # 过滤非有限值
    m = np.isfinite(a) & np.isfinite(b)
    a_f = a[m]
    b_f = b[m]

    if a_f.size < 2 or b_f.size < 2:
        print(f"[{name}][WARN] too few finite samples after filtering: "
              f"finite={int(m.sum())}/{a.size}")
        return

    va = float(np.var(a_f))
    vb = float(np.var(b_f))

    # 是否同一对象 / 是否共享内存（帮助排查“拿同一个数组算相关=1.0”）
    same_object = (a0 is b0)
    share_mem = False
    try:
        share_mem = np.shares_memory(a0, b0)
    except Exception:
        share_mem = False

    print(f"[{name}][SANITY] finite={int(m.sum())}/{a.size} "
          f"var(a)={va:.3e} var(b)={vb:.3e} "
          f"same_object={same_object} share_mem={share_mem}")

    if va < 1e-12 or vb < 1e-12:
        print(f"[{name}][WARN] variance too small -> NCC may be ill-defined/degenerate.")

def compute_continuity_metrics(seq_dict, mode="trace", channel_wise=True,
                               min_common=8, sanity=False, var_eps=1e-12):
    """
    seq_dict: {(l,t):(taus, MU, YY)}
    mode: "trace" 比 (l,t) vs (l,t+1); "line" 比 (l,t) vs (l+1,t)
    返回：dict:
      - mean_ncc, mean_l1
      - (optional) ncc_ch, l1_ch
      - pairs: used neighbor pairs count
      - skipped_var: how many pairs skipped due to tiny variance
      - skipped_common: how many pairs skipped due to too few common taus
    """
    import numpy as np

    ncc_list = []
    l1_list = []
    ncc_ch = [[], [], []]
    l1_ch  = [[], [], []]

    pairs = 0
    skipped_var = 0
    skipped_common = 0

    for (l, t), (taus, MU, _) in seq_dict.items():
        nb = (l, t + 1) if mode == "trace" else (l + 1, t)
        if nb not in seq_dict:
            continue
        taus2, MU2, _ = seq_dict[nb]

        # 对齐 tau（取交集）
        set1 = {int(x): i for i, x in enumerate(taus)}
        set2 = {int(x): i for i, x in enumerate(taus2)}
        common = sorted(set(set1.keys()) & set(set2.keys()))
        if len(common) < int(min_common):
            skipped_common += 1
            continue

        i1 = np.array([set1[c] for c in common], dtype=np.int64)
        i2 = np.array([set2[c] for c in common], dtype=np.int64)

        A = MU[i1]    # (Tc,3)
        B = MU2[i2]   # (Tc,3)

        # ----------------------------
        # overall: NCC + L1
        # ----------------------------
        a = np.asarray(A, dtype=np.float64).reshape(-1)
        b = np.asarray(B, dtype=np.float64).reshape(-1)

        if sanity:
            sanity_corr_inputs(a, b, name=f"CONT/{mode}/overall")

        va = float(np.var(a))
        vb = float(np.var(b))
        if (va < float(var_eps)) or (vb < float(var_eps)):
            # 方差太小：NCC 退化，直接跳过（避免 nan 或 1.0 假象）
            skipped_var += 1
            continue

        ncc = _safe_corrcoef_1d(a, b)
        l1  = float(np.mean(np.abs(A - B)))

        pairs += 1
        if np.isfinite(ncc):
            ncc_list.append(float(ncc))
        if np.isfinite(l1):
            l1_list.append(float(l1))

        # ----------------------------
        # channel-wise
        # ----------------------------
        if channel_wise:
            C = int(A.shape[1])
            # 防呆：如果不是 3 通道，也能跑
            if C != 3:
                # 动态扩展容器
                if len(ncc_ch) != C:
                    ncc_ch = [[] for _ in range(C)]
                    l1_ch  = [[] for _ in range(C)]

            for c in range(C):
                ac = np.asarray(A[:, c], dtype=np.float64).reshape(-1)
                bc = np.asarray(B[:, c], dtype=np.float64).reshape(-1)

                if sanity:
                    sanity_corr_inputs(ac, bc, name=f"CONT/{mode}/ch{c}")

                vac = float(np.var(ac))
                vbc = float(np.var(bc))
                if (vac < float(var_eps)) or (vbc < float(var_eps)):
                    continue  # 单通道退化，跳过该通道 NCC

                ncc_c = _safe_corrcoef_1d(ac, bc)
                l1_c  = float(np.mean(np.abs(A[:, c] - B[:, c])))

                if np.isfinite(ncc_c):
                    ncc_ch[c].append(float(ncc_c))
                if np.isfinite(l1_c):
                    l1_ch[c].append(float(l1_c))

    out = {
        "mean_ncc": float(np.nanmean(ncc_list)) if len(ncc_list) else float("nan"),
        "mean_l1":  float(np.nanmean(l1_list))  if len(l1_list) else float("nan"),
        "pairs": int(pairs),
        "skipped_common": int(skipped_common),
        "skipped_var": int(skipped_var),
    }

    if channel_wise:
        out["ncc_ch"] = [float(np.nanmean(x)) if len(x) else float("nan") for x in ncc_ch]
        out["l1_ch"]  = [float(np.nanmean(x)) if len(x) else float("nan") for x in l1_ch]

    return out



# ====================== Dataset ======================
def _ricker_vec(L=101, dt=0.001, fdom=45.0):
    t = (np.arange(L, dtype=np.float32) - L//2) * dt
    x = np.pi * fdom * t
    w = (1.0 - 2.0 * x**2) * np.exp(-x**2)
    return w.astype(np.float32)
import torch
from torch.utils.data import Dataset
from scipy.signal import fftconvolve

class VolumeWindowDataset(Dataset):
    def __init__(self, stack4d, mod4d, angles_idx=(0,1,2), props_idx=(0,1,2),
                 win=9, time_stride=1, mode='full',
                 line_stride=1, trace_stride=1, phase_line=0, phase_trace=0,
                 wells=None, well_radius=6, far_keep_ratio=0.1,
                 fdom_bl=8.0, dt=0.001, build_bl=True,
                 line_ctx=1,                      # 2.5D 横向上下文条数（奇数）
                 add_line_diff: bool = False,  # ✅ 是否拼接 center-diff 特征（推荐 True 来抓局部结构）
                 force_lt=None,
                 defer_norm=False,                # True: 先不算 norm，split 后再调用 recompute_norm_stats
                 norm_seed=12345,
                 norm_max_samples=20000,
                 norm_indices=None,               # ✅ 如果传了：只用这些 sample indices 计算 mean/std（推荐传 train_indices）
                 norm_exclude_lt=None,            # ✅ 统计时排除某些 (l,t)（比如 LOOW & 2.5D 邻域）
                 # ✅ NEW: Scheme-A norm (angle × time-in-window)
                 norm_scheme: str = "angle_time",  # "angle_time"(推荐) / "angle"（旧）
                 norm_std_floor: float = 1e-3,  # std 下限，防止极弱点放大爆炸
                 norm_gain_cap: float = 20.0,  # 最大放大倍数 cap（1/std 的上限）
                 ):
        super().__init__()
        assert stack4d.shape[0] == mod4d.shape[0] and stack4d.shape[1] == mod4d.shape[1] and stack4d.shape[3] == \
               mod4d.shape[3], \
            f"shape mismatch: stack={stack4d.shape}, mod={mod4d.shape} (need L,T,N match)"
        self.stack = stack4d.astype(np.float32)
        self.mod = mod4d.astype(np.float32)
        self.angles_idx = tuple(angles_idx)
        self.props_idx = tuple(props_idx)
        assert len(self.angles_idx) >= 1 and len(self.props_idx) == 3

        self.win = int(win)
        self.time_stride = int(time_stride)
        assert self.win % 2 == 1
        self.mode = mode

        self.line_stride = int(line_stride)
        self.trace_stride = int(trace_stride)
        self.phase_line = int(phase_line) % max(1, self.line_stride)
        self.phase_trace = int(phase_trace) % max(1, self.trace_stride)

        self.wells = np.asarray(wells, dtype=int) if (wells is not None and len(wells) > 0) else None
        self.well_radius = int(well_radius)
        self.far_keep_ratio = float(far_keep_ratio)

        # ✅ line_ctx 设置
        self.line_ctx = int(line_ctx)
        assert self.line_ctx % 2 == 1 and self.line_ctx >= 1, "line_ctx 必须是 >=1 的奇数（1/3/5/7...）"
        self.add_line_diff = bool(add_line_diff)
        # ✅ FORCE_LT PATCH
        self.force_lt = force_lt
        # dims
        L, T, A, N = self.stack.shape
        self.L, self.T, self.A, self.N = int(L), int(T), int(A), int(N)

        # =========================
        # ✅ band-limited labels (stable amplitude)
        # =========================
        self.mod_bl = None
        if build_bl:
            print(f"[DATA] building band-limited Mod, fdom={fdom_bl} Hz ...")
            Lm, Tn, C, Nn = self.mod.shape

            # 1) build ricker and normalize (important!)
            w = _ricker_vec(L=101, dt=dt, fdom=fdom_bl).astype(np.float32)

            # ✅ 推荐：L2 能量归一化（避免卷积后幅值整体漂移）
            w = w / (np.sqrt(np.sum(w * w)) + 1e-12)

            bl = np.empty_like(self.mod, dtype=np.float32)

            # 2) convolve along time, then amplitude-correct per (l,t,c)
            #    目的：带限，但尽量保持每条曲线的 RMS 幅值不变
            eps = 1e-8
            for il in range(Lm):
                for it in range(Tn):
                    for ic in range(C):
                        x = self.mod[il, it, ic, :].astype(np.float32)

                        y = fftconvolve(x, w, mode="same").astype(np.float32)

                        # ✅ RMS match (per-trace): y *= rms(x)/rms(y)
                        rms_x = float(np.sqrt(np.mean(x * x) + eps))
                        rms_y = float(np.sqrt(np.mean(y * y) + eps))
                        y = y * (rms_x / max(rms_y, eps))

                        bl[il, it, ic, :] = y

            self.mod_bl = bl
            print("[DATA] band-limited Mod built (wavelet norm + RMS matched).")

        # =========================
        # build samples (l,t,tau)
        # =========================
        h = self.win // 2
        if self.mode == 'full':
            lt = [(l, t) for l in range(L) for t in range(T)]
        elif self.mode == 'spatial':
            lt = [(l, t) for l in range(L) for t in range(T)
                  if (l % self.line_stride == self.phase_line) and (t % self.trace_stride == self.phase_trace)]
        else:
            assert self.wells is not None and len(self.wells) > 0, "wellhood 模式需要 wells"
            mask = np.zeros((L, T), dtype=bool)
            wells_valid = []
            for wl, wt in self.wells:
                if 0 <= wl < L and 0 <= wt < T:
                    wells_valid.append((wl, wt))
            for wl, wt in wells_valid:
                l0, l1 = max(0, wl - self.well_radius), min(L - 1, wl + self.well_radius)
                t0, t1 = max(0, wt - self.well_radius), min(T - 1, wt + self.well_radius)
                mask[l0:l1 + 1, t0:t1 + 1] = True
            near = [(l, t) for l in range(L) for t in range(T) if mask[l, t]]
            far = [(l, t) for l in range(L) for t in range(T) if not mask[l, t]]
            keep = max(0, int(len(far) * self.far_keep_ratio))
            if keep > 0 and len(far) > 0:
                rng = np.random.default_rng(12345)
                far = [far[i] for i in rng.choice(len(far), size=keep, replace=False)]
            else:
                far = []
            lt = near + far
            random.shuffle(lt)
        # ==========================================================
        # ✅ FORCE_LT PATCH: 确保某些 (l,t) 一定出现在 lt 中
        #   - 解决 mode='wellhood' 时稀疏采样导致 LOOW core/区域缺失
        # ==========================================================
        if self.force_lt is not None and len(self.force_lt) > 0:
            extra = []
            for (l, t) in self.force_lt:
                l, t = int(l), int(t)
                if 0 <= l < L and 0 <= t < T:
                    extra.append((l, t))

            if len(extra) > 0:
                lt_set = set((int(a), int(b)) for (a, b) in lt)
                before = len(lt_set)
                lt_set.update(extra)
                after = len(lt_set)
                added = after - before
                if added > 0:
                    print(f"[DATA][FORCE_LT] added {added} forced (l,t) into lt (total unique lt={after}).")
                # 顺序不重要：直接 list
                lt = list(lt_set)
        samples = []
        for (l, t) in lt:
            for tau in range(h, N - h, self.time_stride):
                samples.append((l, t, tau))
        self.samples = np.array(samples, dtype=np.int32)

        # =========================
        # mark well samples (by (l,t))
        # =========================
        self._wells_set = set()
        if self.wells is not None and len(self.wells) > 0:
            if self.well_radius <= 0:
                for wl, wt in self.wells:
                    self._wells_set.add((int(wl), int(wt)))
            else:
                R = self.well_radius
                for wl, wt in self.wells:
                    for li in range(max(0, wl - R), min(L - 1, wl + R) + 1):
                        for ti in range(max(0, wt - R), min(T - 1, wt + R) + 1):
                            self._wells_set.add((int(li), int(ti)))

        self.is_well_sample = np.zeros(len(self.samples), dtype=bool)
        if len(self._wells_set) > 0:
            lt_arr = self.samples[:, :2].astype(int)
            # vectorized membership (faster than loop for big N)
            # fallback: set lookup per row
            for i in range(len(self.samples)):
                l, t = int(lt_arr[i, 0]), int(lt_arr[i, 1])
                if (l, t) in self._wells_set:
                    self.is_well_sample[i] = True

        labeled = int(self.is_well_sample.sum())
        print(f"[REPORT] samples={len(self.samples)}, well_labeled={labeled} ({labeled / max(1, len(self.samples)) * 100:.2f}%)")
        if labeled == 0:
            print("[WARN] 没有井样本被标注：检查井映射或增大 --well_radius")

        if self.mod_bl is None:
            raise RuntimeError("mod_bl is None: 请把 VolumeWindowDataset(..., build_bl=True) 打开，BNN-VAIM 需要带限标签。")

        # =========================
        # ✅ normalization stats
        # =========================
        self._norm_seed = int(norm_seed)
        self._norm_max_samples = int(norm_max_samples)
        self._norm_ready = False
        self.norm_scheme = str(norm_scheme)
        self.norm_std_floor = float(norm_std_floor)
        self.norm_gain_cap = float(norm_gain_cap)

        # init placeholders (will be overwritten)
        self.X_mean = None  # scheme A: [A, win]
        self.X_std = None  # scheme A: [A, win]
        self.Y_mean = None  # [3]
        self.Y_std = None  # [3]

        if not bool(defer_norm):
            self.recompute_norm_stats(
                train_indices=norm_indices,
                exclude_lt=norm_exclude_lt,
                max_samples=self._norm_max_samples,
                seed=self._norm_seed,
                verbose=True
            )
        else:
            print("[DATA][NORM] defer_norm=True: 请在 split 后调用 ds_all.recompute_norm_stats(train_indices=...)")

    # ✅ 2.5D patch 提取：返回 [A, line_ctx, win] 或 [2A, line_ctx, win]
    def _get_x_patch_25d(self, l, t, tau):
        h = self.win // 2
        hl = self.line_ctx // 2

        l_ids = np.clip(np.arange(l - hl, l + hl + 1), 0, self.L - 1)
        X = self.stack[l_ids, t, :, tau - h: tau + h + 1]  # [line_ctx, A_all, win]
        X = X[:, self.angles_idx, :]  # [line_ctx, A_sel, win]
        X = np.transpose(X, (1, 0, 2)).astype(np.float32)  # -> [A_sel, line_ctx, win]

        # =========================
        # ✅ NEW: center-diff features (highlight lateral local change)
        # =========================
        if getattr(self, "add_line_diff", False):
            Xc = X[:, hl, :]  # [A_sel, win]
            Xd = X - Xc[:, None, :]  # [A_sel, line_ctx, win]
            X = np.concatenate([X, Xd], axis=0)  # [2*A_sel, line_ctx, win]

        return X

    def recompute_norm_stats(self, train_indices=None, exclude_lt=None,
                             max_samples=20000, seed=12345, verbose=True):
        """
        ✅ Scheme A（推荐）：按 角度×窗内时间位置 统计 mean/std
          - X_mean: [A, win]
          - X_std : [A, win]
        统计维度：对 (sample 维 + line_ctx 维) 求均值/方差
        """
        rng = np.random.default_rng(int(seed))

        all_idx = np.arange(len(self.samples), dtype=np.int64)

        # 1) base candidates
        if train_indices is None:
            cand = all_idx
            src = "ALL"
        else:
            cand = np.asarray(train_indices, dtype=np.int64).reshape(-1)
            cand = cand[(cand >= 0) & (cand < len(self.samples))]
            src = "TRAIN"

        if cand.size == 0:
            raise RuntimeError("[DATA][NORM] empty candidate indices for norm stats")

        # 2) exclude by (l,t)
        if exclude_lt is not None:
            ex = set((int(a), int(b)) for (a, b) in exclude_lt)
            if len(ex) > 0:
                lt_cand = self.samples[cand, :2].astype(int)
                keep_mask = np.array([(int(l), int(t)) not in ex for (l, t) in lt_cand], dtype=bool)
                cand2 = cand[keep_mask]
                if cand2.size > 0:
                    cand = cand2
                else:
                    print("[DATA][NORM][WARN] exclude_lt removed all candidates; fallback to unexcluded candidates.")

        # 3) subsample
        ns = min(int(max_samples), int(cand.size))
        pick = cand if cand.size <= ns else cand[rng.choice(cand.size, size=ns, replace=False)]

        # 4) gather X/Y
        Xs = []
        Ys = []
        for i in pick:
            l, t, tau = self.samples[int(i)]
            Xs.append(self._get_x_patch_25d(int(l), int(t), int(tau)))  # [A, line_ctx, win]
            h = self.win // 2
            y_seq = self.mod[int(l), int(t), self.props_idx, tau - h: tau + h + 1]  # (3,win)
            y_seq = np.transpose(y_seq, (1, 0))  # (win,3)
            Ys.append(y_seq)

        Xc = np.stack(Xs, 0).astype(np.float32)  # [ns, A, line_ctx, win]
        Yc = np.stack(Ys, 0).astype(np.float32)  # [ns, 3]

        # -----------------------------
        # ✅ X stats: Scheme A (A × win)
        #   mean/std over (ns, line_ctx)
        # -----------------------------
        if getattr(self, "norm_scheme", "angle_time") in ("angle_time", "Axt", "A_win"):
            # Xc: [ns, A, line_ctx, win]
            # mean over ns(0) & line_ctx(2) -> [A, win]
            X_mean = Xc.mean(axis=(0, 2))
            X_std = Xc.std(axis=(0, 2))
        else:
            # fallback：旧版（仅按角度）
            # mean/std over ns(0), line_ctx(2), win(3) -> [A]
            X_mean = Xc.mean(axis=(0, 2, 3))
            X_std = Xc.std(axis=(0, 2, 3))

        # -----------------------------
        # ✅ 防爆：std_floor + gain_cap
        #   gain = 1/std ≤ gain_cap  => std ≥ 1/gain_cap
        # -----------------------------
        std_floor = float(getattr(self, "norm_std_floor", 1e-3))
        gain_cap = float(getattr(self, "norm_gain_cap", 20.0))
        std_min_by_gain = 1.0 / max(gain_cap, 1e-6)
        std_min = max(std_floor, std_min_by_gain)

        X_std = np.maximum(X_std, std_min).astype(np.float32)
        X_mean = X_mean.astype(np.float32)

        # -----------------------------
        # ✅ Y stats（保持不变：按通道）
        # -----------------------------
        self.Y_mean = Yc.mean(axis=(0, 1))  # (3,)
        self.Y_std = Yc.std(axis=(0, 1))
        self.Y_std = np.maximum(self.Y_std, 1e-6).astype(np.float32)

        self.X_mean = X_mean
        self.X_std = X_std
        self._norm_ready = True

        if self.Y_std[2] < 1e-6:
            print("[WARN] RHOB 通道近似常数/全零，请确认 Mod 第3通道是否正确。")

        if verbose:
            if self.X_mean.ndim == 2:
                print(f"[DATA][NORM] Scheme-A angle×time: X_mean/std shape={self.X_mean.shape} (A,win) "
                      f"from {src}: ns={ns}, line_ctx={self.line_ctx}, win={self.win}")
            else:
                print(f"[DATA][NORM] Fallback angle-only: X_mean/std shape={self.X_mean.shape} (A,) "
                      f"from {src}: ns={ns}, line_ctx={self.line_ctx}, win={self.win}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self._norm_ready:
            raise RuntimeError("[DATA][NORM] norm stats not ready. "
                               "Use defer_norm=False or call ds.recompute_norm_stats(train_indices=...) after split.")

        l, t, tau = self.samples[idx]

        # ---- 2.5D 地震窗口 ----
        X = self._get_x_patch_25d(int(l), int(t), int(tau))  # [A, line_ctx, win]

        # ✅ Scheme A：按 (A, win) 标准化，line_ctx 维广播
        if self.X_mean.ndim == 2:
            # X_mean/std: [A, win]
            Xn = (X - self.X_mean[:, None, :]) / self.X_std[:, None, :]
        else:
            # fallback：角度-only
            Xn = (X - self.X_mean[:, None, None]) / self.X_std[:, None, None]

        Xn = torch.from_numpy(Xn.astype(np.float32)).float()

        # ---- sequence 标签（win,3） ----
        h = self.win // 2

        # (3, win)
        y = self.mod[int(l), int(t), self.props_idx, tau - h: tau + h + 1]
        y = np.transpose(y, (1, 0))  # → (win,3)

        # normalize（broadcast）
        yn = (y - self.Y_mean[None, :]) / self.Y_std[None, :]
        yn = torch.from_numpy(yn.astype(np.float32)).float()

        # ---- 带限标签（win,3）----
        y_bl = self.mod_bl[int(l), int(t), self.props_idx, tau - h: tau + h + 1]
        y_bl = np.transpose(y_bl, (1, 0))  # (win,3)

        y_bl_n = (y_bl - self.Y_mean[None, :]) / self.Y_std[None, :]
        y_bl_n = torch.from_numpy(y_bl_n.astype(np.float32)).float()

        meta = (int(l), int(t), int(tau), bool(self.is_well_sample[idx]))
        return Xn, yn, y_bl_n, meta

    def denormalize_y(self, y_norm):
        if torch.is_tensor(y_norm):
            Ym = torch.as_tensor(self.Y_mean, dtype=y_norm.dtype, device=y_norm.device)
            Ys = torch.as_tensor(self.Y_std,  dtype=y_norm.dtype, device=y_norm.device)
            return y_norm * Ys + Ym
        else:
            return y_norm * self.Y_std + self.Y_mean

class FullGridDatasetForSection(Dataset):
    """
    在 dataset 侧补齐 full-grid 的 (l,t,tau)，保证 __getitem__ 真正能取到每个网格点的 X/y。
    - 复用 base_ds 的 _get_x_patch_25d / mod / mod_bl / 归一化参数
    - 自己维护 samples / is_well_sample，避免索引错乱
    """
    def __init__(self, base_ds, samples_full, is_well_full=None):
        self.base = base_ds
        self.samples = np.asarray(samples_full, dtype=int)  # [M,3]

        if is_well_full is None:
            self.is_well_sample = np.zeros((len(self.samples),), dtype=bool)
        else:
            self.is_well_sample = np.asarray(is_well_full, dtype=bool)
            assert len(self.is_well_sample) == len(self.samples)

        # 复用 base_ds 的必要属性（让 __getitem__ 同逻辑）
        # 注意：这里不深拷贝大数组，只引用
        self._norm_ready = getattr(base_ds, "_norm_ready", True)

        self.X_mean = base_ds.X_mean
        self.X_std = base_ds.X_std
        self.Y_mean = base_ds.Y_mean
        self.Y_std = base_ds.Y_std

        self.win = base_ds.win
        self.line_ctx = base_ds.line_ctx
        self.angles_idx = base_ds.angles_idx
        self.add_line_diff = getattr(base_ds, "add_line_diff", False)
        self.time_stride = getattr(base_ds, "time_stride", 1)

        self.L = base_ds.L
        self.T = base_ds.T
        self.A = base_ds.A
        self.N = base_ds.N

        self.mod = base_ds.mod
        self.mod_bl = base_ds.mod_bl
        self.props_idx = base_ds.props_idx

        # 复用 base_ds 的 patch 提取函数
        self._get_x_patch_25d = base_ds._get_x_patch_25d

    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        if not self._norm_ready:
            raise RuntimeError("[DATA][NORM] norm stats not ready.")

        l, t, tau = self.samples[idx]

        # ---- 2.5D 地震窗口 ----
        X = self._get_x_patch_25d(int(l), int(t), int(tau))  # [A, line_ctx, win]

        # ✅ Scheme A：按 (A, win) 标准化，line_ctx 维广播
        if self.X_mean.ndim == 2:
            Xn = (X - self.X_mean[:, None, :]) / self.X_std[:, None, :]
        else:
            Xn = (X - self.X_mean[:, None, None]) / self.X_std[:, None, None]

        Xn = torch.from_numpy(Xn.astype(np.float32)).float()

        # ---- sequence 标签（win,3） ----
        h = self.win // 2

        # (3, win)
        y = self.mod[int(l), int(t), self.props_idx, tau - h: tau + h + 1]
        y = np.transpose(y, (1, 0))  # → (win,3)

        # normalize（broadcast）
        yn = (y - self.Y_mean[None, :]) / self.Y_std[None, :]
        yn = torch.from_numpy(yn.astype(np.float32)).float()

        # ---- 带限标签（win,3）----
        y_bl = self.mod_bl[int(l), int(t), self.props_idx, tau - h: tau + h + 1]
        y_bl = np.transpose(y_bl, (1, 0))  # (win,3)

        y_bl_n = (y_bl - self.Y_mean[None, :]) / self.Y_std[None, :]
        y_bl_n = torch.from_numpy(y_bl_n.astype(np.float32)).float()

        meta = (int(l), int(t), int(tau), bool(self.is_well_sample[idx]))
        return Xn, yn, y_bl_n, meta

    def denormalize_y(self, y_norm):
        if torch.is_tensor(y_norm):
            Ym = torch.as_tensor(self.Y_mean, dtype=y_norm.dtype, device=y_norm.device)
            Ys = torch.as_tensor(self.Y_std,  dtype=y_norm.dtype, device=y_norm.device)
            return y_norm * Ys + Ym
        else:
            return y_norm * self.Y_std + self.Y_mean

# ====================== Model ======================

class DepthwiseTemporalConv(nn.Module):
    """
    把输入 (B, A*W) 视为 (B, A, W)，做两层 depthwise 1D conv（第二层 dilation=2）并短残差，
    最后再展平回 (B, A*W)。A: 角度数；W: 窗长
    """
    def __init__(self, n_angles: int, win: int, kernel_size: int = 9):
        super().__init__()
        pad1 = kernel_size // 2
        # 每个角度独立卷积
        self.conv1 = nn.Conv1d(n_angles, n_angles, kernel_size=kernel_size,
                               padding=pad1, groups=n_angles, bias=True)
        # 膨胀卷积，进一步扩大感受野，缓解“滞后”
        self.conv2 = nn.Conv1d(n_angles, n_angles, kernel_size=5,
                               padding=4, dilation=2, groups=n_angles, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

        self.n_angles = n_angles
        self.win = win
        self.act = nn.GELU()

    def forward(self, x_flat):
        B = x_flat.size(0)
        x = x_flat.view(B, self.n_angles, self.win)       # (B, A, W)
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y)) + y                   # 短残差
        return y.reshape(B, self.n_angles * self.win)     # 回到 (B, A*W)

class CNNEncoder1D(nn.Module):
    """
    简单 1D-CNN 编码器：
    输入 X: [B, A, win]，输出一个 feature 向量 h: [B, feat_dim]
    """
    def __init__(self, n_angles: int, win: int, feat_dim: int = 256):
        super().__init__()
        self.n_angles = n_angles
        self.win = win

        # 你可以按需要调通道数 / kernel_size
        self.conv1 = nn.Conv1d(n_angles, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 全局平均池化 + 线性映射到 feat_dim
        # 池化 3 次，下采样 2^3=8 倍，长度约 win/8
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, X_flat: torch.Tensor) -> torch.Tensor:
        """
        X_flat: [B, A*win]
        """
        B = X_flat.size(0)
        A = self.n_angles
        W = self.win
        x = X_flat.view(B, A, W)        # -> [B, A, win]

        x = self.pool(self.act(self.conv1(x)))   # [B,64,L1]
        x = self.pool(self.act(self.conv2(x)))   # [B,128,L2]
        x = self.pool(self.act(self.conv3(x)))   # [B,128,L3]

        # 沿 time 维做全局平均池化 -> [B,128]
        x = x.mean(dim=-1)
        h = self.proj(x)                        # [B, feat_dim]
        return h

class CNNEncoder2D25D(nn.Module):
    """
    2.5D 2D-CNN 编码器：
    输入 X: [B, C_in, line_ctx, win]
    """
    def __init__(self, in_ch: int, line_ctx: int, win: int, feat_dim: int = 256):
        super().__init__()
        self.in_ch = int(in_ch)
        self.line_ctx = int(line_ctx)
        self.win = int(win)

        self.conv1 = nn.Conv2d(self.in_ch, 32, kernel_size=(3, 7), padding=(1, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2))

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # only time
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            x = X
        elif X.dim() == 3:
            x = X.unsqueeze(2)
        elif X.dim() == 2:
            B = X.size(0)
            if X.size(1) == self.in_ch * self.win:
                x = X.view(B, self.in_ch, 1, self.win)
            elif X.size(1) == self.in_ch * self.line_ctx * self.win:
                x = X.view(B, self.in_ch, self.line_ctx, self.win)
            else:
                raise RuntimeError(f"[CNNEncoder2D25D] Unexpected flat dim: {X.size(1)}")
        else:
            raise RuntimeError(f"[CNNEncoder2D25D] unexpected X.dim={X.dim()}")

        if not getattr(self, "_dbg_printed", False):
            print(f"[DBG][Enc2D] X.shape={tuple(X.shape)} -> x.shape={tuple(x.shape)} (in_ch={self.in_ch})")
            self._dbg_printed = True

        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))

        x = self.gap(x).flatten(1)
        h = self.proj(x)
        return h


class SmallBNNHeadSeqMVT(nn.Module):
    """
    输出：
      mu:          (B, win, 3)
      chol_params: (B, win, 6)
      kl:          scalar tensor
      extra: dict
        - mu_comp:     (B, win, K, 3)
        - gate_logits: (B, win, K)
        - gate:        (B, win, K)
    """
    def __init__(self,
                 in_dim: int,
                 hidden: int,
                 win: int,
                 out_dim: int = 3,
                 bayes: bool = True,
                 n_comp: int = 3):
        super().__init__()
        assert out_dim == 3, "当前实现固定针对 3 参数 (VP, VS, RHOB)"
        self.bayes = bool(bayes)
        self.out_dim = int(out_dim)
        self.win = int(win)
        self.n_comp = int(n_comp)

        mu_comp_dim = self.win * self.n_comp * self.out_dim   # win * K * 3
        gate_dim = self.win * self.n_comp                     # win * K
        chol_dim = self.win * 6                               # win * 6

        if self.bayes and LinearReparameterization is not None:
            self.fc1 = LinearReparameterization(in_dim, hidden)
            self.fc_mu_comp = LinearReparameterization(hidden, mu_comp_dim)
            self.fc_gate = LinearReparameterization(hidden, gate_dim)
            self.fc_chol = LinearReparameterization(hidden, chol_dim)
        else:
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc_mu_comp = nn.Linear(hidden, mu_comp_dim)
            self.fc_gate = nn.Linear(hidden, gate_dim)
            self.fc_chol = nn.Linear(hidden, chol_dim)

    def forward(self, h: torch.Tensor, return_kl: bool = True):
        kl_total = 0.0

        # -----------------------------
        # shared hidden
        # -----------------------------
        if self.bayes and isinstance(self.fc1, LinearReparameterization):
            h, kl1 = self.fc1(h)
            kl_total = kl_total + kl1
        else:
            h = self.fc1(h)

        h = torch.relu(h)

        # -----------------------------
        # mixture mean + covariance
        # -----------------------------
        if self.bayes and isinstance(self.fc_mu_comp, LinearReparameterization):
            mu_comp_raw, kl_mu = self.fc_mu_comp(h)
            gate_logits_raw, kl_gate = self.fc_gate(h)
            chol_params, kl_ch = self.fc_chol(h)
            kl_total = kl_total + kl_mu + kl_gate + kl_ch
        else:
            mu_comp_raw = self.fc_mu_comp(h)
            gate_logits_raw = self.fc_gate(h)
            chol_params = self.fc_chol(h)

        B = h.shape[0]

        # [B, win, K, 3]
        mu_comp = mu_comp_raw.view(B, self.win, self.n_comp, self.out_dim)

        # [B, win, K]
        gate_logits = gate_logits_raw.view(B, self.win, self.n_comp)
        gate = torch.softmax(gate_logits, dim=-1)

        # mixture mean: [B, win, 3]
        mu = torch.sum(mu_comp * gate.unsqueeze(-1), dim=2)

        # [B, win, 6]
        chol_params = chol_params.view(B, self.win, 6)

        if not return_kl:
            return mu, chol_params, None, {
                "mu_comp": mu_comp,
                "gate_logits": gate_logits,
                "gate": gate,
            }

        if not torch.is_tensor(kl_total):
            kl_total = torch.as_tensor(kl_total, device=mu.device, dtype=mu.dtype)

        extra = {
            "mu_comp": mu_comp,
            "gate_logits": gate_logits,
            "gate": gate,
        }
        return mu, chol_params, kl_total, extra

class AttrNet25D(nn.Module):
    def __init__(self,
                 n_angles: int,
                 win: int,
                 line_ctx: int = 1,
                 in_ch: int = None,
                 feat_dim: int = 256,
                 hidden: int = 256,
                 out_dim: int = 3,
                 bayes: bool = True,
                 hetero: bool = True,
                 n_comp: int = 3):
        super().__init__()
        self.n_angles = int(n_angles)
        self.win = int(win)
        self.line_ctx = int(line_ctx)

        self.in_ch = int(in_ch) if (in_ch is not None) else self.n_angles

        self._dbg_i = 0
        self.encoder = CNNEncoder2D25D(
            in_ch=self.in_ch,
            line_ctx=self.line_ctx,
            win=self.win,
            feat_dim=feat_dim
        )
        self.head = SmallBNNHeadSeqMVT(
            in_dim=feat_dim,
            hidden=hidden,
            win=self.win,
            out_dim=out_dim,
            bayes=bayes,
            n_comp=n_comp,
        )

    def forward(self, X: torch.Tensor, return_kl: bool = True):
        if X.dim() == 4:
            x = X
        elif X.dim() == 3:
            x = X.unsqueeze(2)
        elif X.dim() == 2:
            B = X.size(0)
            if X.size(1) == self.n_angles * self.win:
                x = X.view(B, self.n_angles, 1, self.win)
            elif X.size(1) == self.n_angles * self.line_ctx * self.win:
                x = X.view(B, self.n_angles, self.line_ctx, self.win)
            else:
                raise RuntimeError(f"[AttrNet25D] Unexpected flat dim: {X.size(1)}")
        else:
            raise RuntimeError(f"[AttrNet25D] Unexpected X.dim={X.dim()}")

        self._dbg_i = getattr(self, "_dbg_i", 0) + 1
        if self._dbg_i % 200 == 1:
            print(f"[DBG][AttrNet25D] step={self._dbg_i} X.shape={tuple(X.shape)} -> x.shape={tuple(x.shape)}")

        h = self.encoder(x)

        out = self.head(h, return_kl=return_kl)
        if isinstance(out, (tuple, list)) and len(out) == 4:
            mu, chol_params, kl, extra = out
            return mu, chol_params, kl, extra
        else:
            mu, chol_params, kl = out
            return mu, chol_params, kl


class BNNVAIMPhysics(nn.Module):
    def __init__(self, attr_net: nn.Module,
                 physics_model: nn.Module,
                 denormalize_fn=None,
                 n_angles: int = 3,
                 win: int = 128):
        super().__init__()
        self.inv_net = attr_net
        self.fwd_net = None
        self.physics_model = physics_model
        self.denormalize_fn = denormalize_fn
        self._n_angles = n_angles
        self._win = win

    def forward(self, X: torch.Tensor, return_kl: bool = True):
        out = self.inv_net(X, return_kl=return_kl)

        extra = None
        if isinstance(out, (tuple, list)) and len(out) >= 4:
            mu_attr, chol_params, kl, extra = out[:4]
        elif isinstance(out, (tuple, list)) and len(out) >= 3:
            mu_attr, chol_params, kl = out[:3]
        else:
            raise RuntimeError(
                f"[BNNVAIMPhysics] Unexpected inv_net output type={type(out)} "
                f"len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
            )

        if self.denormalize_fn is not None:
            props_phys = self.denormalize_fn(mu_attr)
        else:
            props_phys = mu_attr

        B, W, C = props_phys.shape
        props_flat = props_phys.reshape(B * W, C)
        d_rec_flat = self.physics_model(props_flat)
        d_rec = d_rec_flat.view(B, W, -1)

        if extra is None:
            return mu_attr, chol_params, d_rec, kl
        return mu_attr, chol_params, d_rec, kl, extra

# ====================== Sampler & collate ======================

import numpy as np
from torch.utils.data import Sampler, Subset


class WellBalancedBatchSampler(Sampler):
    """
    保证每批至少含 min_well 个井样本（井样本不足时尽力而为）

    ✅ trace-block 采样（纵向 tau 连续）
    - trace_index_table: dict[(l,t)] -> [global idx...]，且已按 tau 排序
    - trace_block: 每个 trace 一次取多少个 tau 点
    - prefer_consecutive: True 时取连续 tau 块

    ✅ line-block 采样（横向 line 连续）
    - line_index_table: dict[(t,tau)] -> [global idx...]，且已按 l 排序
    - line_block: 每次从同一个 (t,tau) 取多少个相邻 l
    - line_quota: 每个 batch 里用于 line-block 的比例(0~1)，建议 0.2~0.4（有 patch 时更低）
    - prefer_line_consecutive: True 时取连续 l 块

    ✅ NEW: 2D patch 注入（line × tau 成片）
    - samples_full: ds.samples，要求 samples_full[global_idx] -> (l,t,tau)
    - grid_index_table: dict[(l,t,tau)] -> global_idx（可不传，传 samples_full 即自动建）
    - patch_line_block / patch_tau_block / patch_tau_stride
    - patch_quota: 每个 batch 预留给 patch 的比例（建议 0.4~0.6）
    - patch_min_keep: patch 命中点太少则丢弃（避免空 patch）
    """

    def __init__(self,
                 subset: Subset,
                 is_well_full: np.ndarray,
                 batch_size: int,
                 min_well: int = 4,
                 seed: int = 1234,
                 drop_last: bool = False,
                 nonwell_weight_table: dict | None = None,

                 # --- trace-block ---
                 trace_index_table: dict | None = None,
                 trace_block: int = 1,
                 prefer_consecutive: bool = True,

                 # --- line-block ---
                 line_index_table: dict | None = None,
                 line_block: int = 16,
                 line_quota: float = 0.7,              # 建议 0.2~0.4；0=关闭
                 prefer_line_consecutive: bool = True,

                 # --- ✅ NEW: 2D patch (line × tau) ---
                 samples_full=None,                    # ds.samples
                 grid_index_table: dict | None = None, # dict[(l,t,tau)]->global_idx
                 patch_line_block: int = 16,
                 patch_tau_block: int = 8,
                 patch_tau_stride: int = 1,
                 patch_quota: float = 0.0,             # 0=关闭，建议 0.4~0.6
                 patch_min_keep: int = 32,             # patch 至少命中多少点才算有效

                 # debug
                 debug_every: int = 0,
                 debug_first: int = 3):

        self.debug_every = int(debug_every)
        self.debug_first = int(debug_first)

        self.indices = np.array(subset.indices, dtype=np.int64)
        wflags = np.asarray(is_well_full).astype(bool)

        self.wflags_full = wflags
        self.well_idx = self.indices[wflags[self.indices]]
        self.non_idx  = self.indices[~wflags[self.indices]]

        self.bs = int(batch_size)
        self.min_well = int(min_well)
        self.drop_last = bool(drop_last)

        self.nonwell_weight_table = nonwell_weight_table

        self.seed = int(seed)
        self._epoch = 0

        # =============== trace-block ===============
        self.trace_index_table = trace_index_table
        self.trace_block = max(1, int(trace_block))
        self.prefer_consecutive = bool(prefer_consecutive)
        self._use_trace_block = (self.trace_index_table is not None) and (self.trace_block > 1)

        self.well_traces = []
        self.non_traces = []
        self.non_trace_weights = None

        if self._use_trace_block:
            trace_keys = list(self.trace_index_table.keys())
            for k in trace_keys:
                idxs = self.trace_index_table[k]
                if len(idxs) == 0:
                    continue
                is_well_trace = bool(wflags[np.asarray(idxs, dtype=np.int64)].any())
                if is_well_trace:
                    self.well_traces.append(k)
                else:
                    self.non_traces.append(k)

            if self.nonwell_weight_table is not None and len(self.non_traces) > 0:
                tw = []
                for k in self.non_traces:
                    idxs = self.trace_index_table[k]
                    ww = [float(self.nonwell_weight_table.get(int(i), 1.0)) for i in idxs]
                    tw.append(float(np.mean(ww)) if len(ww) > 0 else 1.0)
                tw = np.asarray(tw, dtype=np.float64)
                tw = tw / (tw.sum() + 1e-12)
                self.non_trace_weights = tw

            if len(self.well_traces) == 0:
                print("[WARN][Sampler] trace_block enabled but found 0 well traces in table. min_well may be weak.")
            print(f"[Sampler] trace_block enabled: trace_block={self.trace_block}, "
                  f"well_traces={len(self.well_traces)}, non_traces={len(self.non_traces)}")

        # =============== line-block ===============
        self.line_index_table = line_index_table
        self.line_block = int(line_block)
        self.line_quota = float(line_quota)
        self.prefer_line_consecutive = bool(prefer_line_consecutive)

        self._use_line_block = (self.line_index_table is not None) and (self.line_block > 1) and (self.line_quota > 0.0)
        self.line_keys = list(self.line_index_table.keys()) if (self.line_index_table is not None) else []

        if self._use_line_block:
            print(f"[Sampler] line_block enabled: line_block={self.line_block}, keys={len(self.line_keys)}, "
                  f"quota={self.line_quota:.2f}")

        # =============== ✅ 2D patch ===============
        self.samples_full = samples_full
        self.grid_index_table = grid_index_table

        self.patch_line_block = int(patch_line_block)
        self.patch_tau_block = int(patch_tau_block)
        self.patch_tau_stride = max(1, int(patch_tau_stride))
        self.patch_quota = float(patch_quota)
        self.patch_min_keep = int(patch_min_keep)

        self._use_patch = (self.patch_quota > 0.0) and (self.patch_line_block > 1) and (self.patch_tau_block > 1)

        # 若未给 grid_index_table，但给了 samples_full：自动为 subset 建映射
        self._lmin = self._lmax = None
        self._taumin = self._taumax = None
        self._tmin = self._tmax = None

        if self._use_patch:
            if self.samples_full is None:
                print("[WARN][Sampler] patch_quota>0 but samples_full is None, disable patch.")
                self._use_patch = False

        if self._use_patch:
            if self.grid_index_table is None:
                # 自动建 subset 映射（只针对 subset indices，避免全量）
                g = {}
                lvals, tvals, tauvals = [], [], []
                for gi in self.indices:
                    l, t, tau = self.samples_full[int(gi)]
                    l = int(l); t = int(t); tau = int(tau)
                    g[(l, t, tau)] = int(gi)
                    lvals.append(l); tvals.append(t); tauvals.append(tau)
                self.grid_index_table = g
                if len(lvals) > 0:
                    self._lmin, self._lmax = int(min(lvals)), int(max(lvals))
                    self._tmin, self._tmax = int(min(tvals)), int(max(tvals))
                    self._taumin, self._taumax = int(min(tauvals)), int(max(tauvals))
            else:
                # 尝试估计范围（可选）
                try:
                    keys = list(self.grid_index_table.keys())
                    if len(keys) > 0:
                        lvals = [k[0] for k in keys]
                        tvals = [k[1] for k in keys]
                        tauvals = [k[2] for k in keys]
                        self._lmin, self._lmax = int(min(lvals)), int(max(lvals))
                        self._tmin, self._tmax = int(min(tvals)), int(max(tvals))
                        self._taumin, self._taumax = int(min(tauvals)), int(max(tauvals))
                except Exception:
                    pass

            print(f"[Sampler] 2D patch enabled: line×tau=({self.patch_line_block}×{self.patch_tau_block}), "
                  f"stride={self.patch_tau_stride}, quota={self.patch_quota:.2f}, "
                  f"grid={len(self.grid_index_table) if self.grid_index_table is not None else 0}")

        # =============== 原始告警 ===============
        if len(self.well_idx) == 0:
            print("[WARN][Sampler] 训练集中没有井样本，min_well 约束无效。")
        elif len(self.well_idx) < self.min_well:
            print(f"[WARN][Sampler] 井样本总数 nW={len(self.well_idx)} < min_well={self.min_well}，"
                  f"无法保证每批至少 {self.min_well} 个井样本（只能做到每批最多 {len(self.well_idx)}）。")

    def __len__(self):
        total = len(self.indices)
        if self.drop_last:
            return total // self.bs
        return (total + self.bs - 1) // self.bs

    # ------------------ helpers ------------------

    def _take_block_from_trace(self, trace_key, k, rng):
        idxs = self.trace_index_table.get(trace_key, [])
        n = len(idxs)
        if n <= 0 or k <= 0:
            return []
        k = int(min(k, n))
        if n == k:
            return list(idxs)
        if self.prefer_consecutive:
            s = int(rng.integers(0, n - k + 1))
            return list(idxs[s:s + k])
        else:
            pick = rng.choice(n, size=k, replace=False)
            pick = sorted([idxs[i] for i in pick])
            return pick

    def _take_block_from_line(self, key_tt, k, rng):
        """key_tt = (t, tau)，idxs 已按 l 排序"""
        idxs = self.line_index_table.get(key_tt, [])
        n = len(idxs)
        if n <= 0 or k <= 0:
            return []
        k = int(min(k, n))
        if n == k:
            return list(idxs)
        if self.prefer_line_consecutive:
            s = int(rng.integers(0, n - k + 1))
            return list(idxs[s:s + k])
        else:
            pick = rng.choice(n, size=k, replace=False)
            pick = sorted([idxs[i] for i in pick])
            return pick

    def _take_2d_patch(self, l_center: int, t0: int, tau_center: int):
        """
        以 (l_center,t0,tau_center) 为中心，取 line×tau patch（按 stride）
        返回 global idx list（只返回 table 中存在的点）
        """
        if self.grid_index_table is None:
            return []

        half_l = self.patch_line_block // 2
        half_tau = self.patch_tau_block // 2

        # line 范围（尽量对称）
        l_start = int(l_center - half_l)
        ls = [l_start + i for i in range(self.patch_line_block)]

        # tau 范围（按 stride）
        tau_start = int(tau_center - half_tau * self.patch_tau_stride)
        taus = [tau_start + i * self.patch_tau_stride for i in range(self.patch_tau_block)]

        # 若估计过范围，则做轻度裁剪，提升命中率
        if self._lmin is not None:
            ls = [l for l in ls if (self._lmin <= l <= self._lmax)]
        if self._taumin is not None:
            taus = [tau for tau in taus if (self._taumin <= tau <= self._taumax)]
        if self._tmin is not None:
            if not (self._tmin <= t0 <= self._tmax):
                return []

        out = []
        grid = self.grid_index_table
        for tau in taus:
            for l in ls:
                gi = grid.get((int(l), int(t0), int(tau)), None)
                if gi is not None:
                    out.append(int(gi))
        return out

    # ------------------ iterator ------------------
    def __iter__(self):
        self._epoch += 1
        rng = np.random.default_rng(self.seed + self._epoch)

        total = len(self.indices)
        n_batches = len(self)

        # =========================================================
        # trace-block 分支
        # =========================================================
        if self._use_trace_block:
            well_traces = list(self.well_traces)
            non_traces = list(self.non_traces)
            rng.shuffle(well_traces)
            rng.shuffle(non_traces)
            wi = 0
            ni = 0

            for b in range(n_batches):

                # ---------------- batch size ----------------
                if (not self.drop_last) and (b == n_batches - 1):
                    cur_bs = total - (n_batches - 1) * self.bs
                    cur_bs = max(1, int(cur_bs))
                else:
                    cur_bs = self.bs

                # ---------------- well / non-well quota ----------------
                if len(self.well_idx) == 0:
                    kw = 0
                else:
                    kw = min(self.min_well, cur_bs)
                    if len(self.well_idx) < self.min_well:
                        kw = min(kw, len(self.well_idx))
                kn = cur_bs - kw

                chosen = []
                chosen_set = set()

                def _add_idxs(idxs):
                    for ii in idxs:
                        ii = int(ii)
                        if ii not in chosen_set:
                            chosen_set.add(ii)
                            chosen.append(ii)
                            if len(chosen) >= cur_bs * 3:  # 防止极端情况下爆炸
                                break

                # =====================================================
                # (0) ✅ 先注入 2D patch（line × tau）
                # =====================================================
                patch_used = 0
                if self._use_patch and self.grid_index_table is not None and self.samples_full is not None:
                    patch_size_nom = max(1, self.patch_line_block * self.patch_tau_block)
                    n_patch_target = int(round(cur_bs * self.patch_quota))
                    n_patch_target = max(0, min(n_patch_target, cur_bs))
                    n_patches = max(1, n_patch_target // patch_size_nom) if n_patch_target > 0 else 0

                    before0 = len(chosen)
                    for _ in range(n_patches):
                        if len(chosen) >= cur_bs:
                            break

                        # 尝试多次找“命中率高”的 patch
                        ok = False
                        for _try in range(20):
                            gi0 = int(rng.choice(self.indices))
                            l0, t0, tau0 = self.samples_full[int(gi0)]
                            pick = self._take_2d_patch(int(l0), int(t0), int(tau0))
                            if len(pick) >= self.patch_min_keep:
                                _add_idxs(pick)
                                ok = True
                                break
                        if not ok:
                            # 实在找不到也塞一次（可能数据稀疏）
                            gi0 = int(rng.choice(self.indices))
                            l0, t0, tau0 = self.samples_full[int(gi0)]
                            pick = self._take_2d_patch(int(l0), int(t0), int(tau0))
                            _add_idxs(pick)

                    patch_used = len(chosen) - before0

                # =====================================================
                # (0.5) ✅ 再注入 line-block（避免小 batch 强行 16）
                # =====================================================
                line_used = 0
                if self._use_line_block and len(self.line_keys) > 0 and len(chosen) < cur_bs:

                    # 只用非井 quota 塞 line-block，避免后面补井/补非井把它截碎
                    max_line_cap = max(0, int(kn))
                    if max_line_cap >= 3:
                        n_line = int(round(cur_bs * self.line_quota))
                        n_line = min(n_line, max_line_cap)
                        if n_line >= 3:
                            line_block_eff = int(min(self.line_block, n_line, max_line_cap, cur_bs))
                            if line_block_eff >= 3:
                                n_blocks = max(1, n_line // line_block_eff)
                                replace = (len(self.line_keys) < n_blocks)
                                keys = rng.choice(self.line_keys, size=n_blocks, replace=replace)

                                before1 = len(chosen)
                                used_keys = set()
                                for ktt in keys:
                                    ktt = tuple(ktt)
                                    if ktt in used_keys and len(self.line_keys) > 1:
                                        continue
                                    used_keys.add(ktt)

                                    pick = self._take_block_from_line(ktt, line_block_eff, rng)
                                    _add_idxs(pick)

                                    if len(chosen) - before1 >= n_line:
                                        break

                                line_used = len(chosen) - before1

                # 当前 well 数
                wcnt = int(self.wflags_full[np.asarray(chosen, dtype=np.int64)].sum()) if len(chosen) > 0 else 0

                # =====================================================
                # (1) 再补井点（trace-block）
                # =====================================================
                if kw > 0 and len(well_traces) > 0:
                    guard = 0
                    while wcnt < kw and guard < 10000 and len(chosen) < cur_bs:
                        guard += 1
                        if wi >= len(well_traces):
                            rng.shuffle(well_traces)
                            wi = 0
                        key = well_traces[wi]
                        wi += 1

                        need = kw - wcnt
                        kblk = min(self.trace_block, max(1, need))
                        pick = self._take_block_from_trace(key, kblk, rng)

                        before = len(chosen)
                        _add_idxs(pick)
                        if len(chosen) > before:
                            new = chosen[before:]
                            wcnt += int(self.wflags_full[np.asarray(new, dtype=np.int64)].sum())

                # 兜底：井点不足
                if kw > 0 and wcnt < kw and len(self.well_idx) > 0 and len(chosen) < cur_bs:
                    need = kw - wcnt
                    pick = rng.choice(self.well_idx, size=int(need * 3), replace=True).tolist()
                    _add_idxs(pick)
                    wcnt = int(self.wflags_full[np.asarray(chosen, dtype=np.int64)].sum())

                # =====================================================
                # (2) 补非井点（trace-block）
                # =====================================================
                while len(chosen) < cur_bs:
                    need = cur_bs - len(chosen)
                    kblk = min(self.trace_block, max(1, need))

                    if len(non_traces) > 0:
                        if self.non_trace_weights is not None:
                            tid = int(rng.choice(len(non_traces), p=self.non_trace_weights))
                            key = non_traces[tid]
                        else:
                            if ni >= len(non_traces):
                                rng.shuffle(non_traces)
                                ni = 0
                            key = non_traces[ni]
                            ni += 1
                        pick = self._take_block_from_trace(key, kblk, rng)
                        _add_idxs(pick)
                    else:
                        if len(self.non_idx) == 0:
                            break
                        pick = rng.choice(self.non_idx, size=int(need * 2), replace=True).tolist()
                        _add_idxs(pick)

                    if len(chosen_set) >= len(self.indices):
                        break

                # ---------------- truncate & shuffle ----------------
                if len(chosen) > cur_bs:
                    chosen = chosen[:cur_bs]

                batch = np.array(chosen, dtype=np.int64)
                rng.shuffle(batch)

                if self.drop_last and len(batch) < self.bs:
                    continue

                # ---------------- DEBUG ----------------
                do_dbg = (b < self.debug_first) or (self.debug_every > 0 and (b % self.debug_every == 0))
                if do_dbg:
                    n_w = int(self.wflags_full[batch].sum()) if len(batch) > 0 else 0
                    n_n = int(len(batch) - n_w)
                    print(
                        f"[SAMPLER][ep={self._epoch:03d} b={b:04d}] "
                        f"bs={len(batch)} | well_pts={n_w} non_pts={n_n} | "
                        f"patch_used={patch_used} (L×T={self.patch_line_block}×{self.patch_tau_block}, q={self.patch_quota:.2f}) | "
                        f"line_used={line_used} line_block={self.line_block} line_quota={self.line_quota:.2f} | "
                        f"trace_block={self.trace_block}"
                    )

                yield batch.tolist()

            return  # ✅ trace-block 分支结束

        # =========================================================
        # 点级采样分支（原逻辑） + 可选 patch / line-block 注入
        # =========================================================
        well_pool = self.well_idx.copy()
        non_pool  = self.non_idx.copy()
        rng.shuffle(well_pool)
        rng.shuffle(non_pool)

        nW = len(well_pool)
        nN = len(non_pool)

        w_non = None
        if self.nonwell_weight_table is not None and len(self.non_idx) > 0:
            w_non = np.array([self.nonwell_weight_table.get(int(i), 1.0) for i in self.non_idx], dtype=np.float64)
            w_non = w_non / (w_non.sum() + 1e-12)

        wi = 0
        ni = 0

        for b in range(n_batches):
            if (not self.drop_last) and (b == n_batches - 1):
                cur_bs = total - (n_batches - 1) * self.bs
                cur_bs = max(1, int(cur_bs))
            else:
                cur_bs = self.bs

            kw = 0 if nW == 0 else min(self.min_well, cur_bs)
            kn = cur_bs - kw

            chosen = []
            chosen_set = set()

            def _add_idxs(idxs):
                for ii in idxs:
                    ii = int(ii)
                    if ii not in chosen_set:
                        chosen_set.add(ii)
                        chosen.append(ii)
                        if len(chosen) >= cur_bs * 3:
                            break

            # ✅ (0) patch 注入
            patch_used = 0
            if self._use_patch and self.grid_index_table is not None and self.samples_full is not None:
                patch_size_nom = max(1, self.patch_line_block * self.patch_tau_block)
                n_patch_target = int(round(cur_bs * self.patch_quota))
                n_patch_target = max(0, min(n_patch_target, cur_bs))
                n_patches = max(1, n_patch_target // patch_size_nom) if n_patch_target > 0 else 0

                before0 = len(chosen)
                for _ in range(n_patches):
                    if len(chosen) >= cur_bs:
                        break
                    ok = False
                    for _try in range(20):
                        gi0 = int(rng.choice(self.indices))
                        l0, t0, tau0 = self.samples_full[int(gi0)]
                        pick = self._take_2d_patch(int(l0), int(t0), int(tau0))
                        if len(pick) >= self.patch_min_keep:
                            _add_idxs(pick)
                            ok = True
                            break
                    if not ok:
                        gi0 = int(rng.choice(self.indices))
                        l0, t0, tau0 = self.samples_full[int(gi0)]
                        pick = self._take_2d_patch(int(l0), int(t0), int(tau0))
                        _add_idxs(pick)

                patch_used = len(chosen) - before0

            # ✅ (0.5) line-block 注入
            line_used = 0
            if self._use_line_block and len(self.line_keys) > 0 and len(chosen) < cur_bs:
                n_line = int(round(cur_bs * self.line_quota))
                n_line = min(n_line, cur_bs)
                if n_line >= 3:
                    line_block_eff = int(min(self.line_block, n_line, cur_bs))
                    if line_block_eff >= 3:
                        n_blocks = max(1, n_line // line_block_eff)
                        replace = (len(self.line_keys) < n_blocks)
                        keys = rng.choice(self.line_keys, size=n_blocks, replace=replace)

                        before1 = len(chosen)
                        used_keys = set()
                        for ktt in keys:
                            ktt = tuple(ktt)
                            if ktt in used_keys and len(self.line_keys) > 1:
                                continue
                            used_keys.add(ktt)
                            pick = self._take_block_from_line(ktt, line_block_eff, rng)
                            _add_idxs(pick)
                        line_used = len(chosen) - before1

            # 再补 wells
            if kw > 0:
                guard = 0
                while int(self.wflags_full[np.asarray(chosen, dtype=np.int64)].sum()) < kw and guard < 10000 and len(chosen) < cur_bs:
                    guard += 1
                    if wi >= nW:
                        rng.shuffle(well_pool)
                        wi = 0
                    _add_idxs([well_pool[wi]])
                    wi += 1

            # 再补 non
            while len(chosen) < cur_bs:
                need = cur_bs - len(chosen)
                if nN == 0:
                    break
                if w_non is not None:
                    idx_sel = rng.choice(len(self.non_idx), size=int(need * 2), replace=True, p=w_non)
                    _add_idxs(self.non_idx[idx_sel].tolist())
                else:
                    if ni >= nN:
                        rng.shuffle(non_pool)
                        ni = 0
                    _add_idxs(non_pool[ni:ni + need].tolist())
                    ni += need

            if len(chosen) > cur_bs:
                chosen = chosen[:cur_bs]

            batch = np.array(chosen, dtype=np.int64)
            rng.shuffle(batch)

            if self.drop_last and len(batch) < self.bs:
                continue

            do_dbg = (b < self.debug_first) or (self.debug_every > 0 and (b % self.debug_every == 0))
            if do_dbg:
                n_w = int(self.wflags_full[batch].sum()) if len(batch) > 0 else 0
                n_n = int(len(batch) - n_w)
                print(
                    f"[SAMPLER][ep={self._epoch:03d} b={b:04d}] "
                    f"bs={len(batch)} | well_pts={n_w} non_pts={n_n} | "
                    f"patch_used={patch_used} (L×T={self.patch_line_block}×{self.patch_tau_block}, q={self.patch_quota:.2f}) | "
                    f"line_used={line_used} line_block={self.line_block} line_quota={self.line_quota:.2f} | "
                    f"use_trace_block={self._use_trace_block}"
                )

            yield batch.tolist()



def _collate(batch):
    """
    支持两种 Dataset 返回：
      1) (X, y, y_bl, meta)
      2) (X, y, meta)   -> 自动补 y_bl=None
    """
    import torch

    if len(batch) == 0:
        raise RuntimeError("_collate got empty batch")

    # 情况1：四元组
    if len(batch[0]) == 4:
        Xs, ys, y_bls, metas = zip(*batch)
        X = torch.stack(Xs, 0)
        y = torch.stack(ys, 0)
        if any(v is None for v in y_bls):
            y_bl = None
        else:
            y_bl = torch.stack(y_bls, 0)
        return X, y, y_bl, list(metas)



    # 情况2：三元组
    elif len(batch[0]) == 3:
        Xs, ys, metas = zip(*batch)
        X = torch.stack(Xs, 0)
        y = torch.stack(ys, 0)
        y_bl = None
        return X, y, y_bl, list(metas)

    else:
        raise RuntimeError(f"_collate: unexpected sample tuple size={len(batch[0])}")


def _flat_to_cholesky(tril_flat: torch.Tensor, D: int, jitter: float = 1e-6):
    """
    将 (B, D*(D+1)/2) 的下三角扁平参数还原为 L (B, D, D)，
    其中对角线通过 softplus 确保正数，并加上 jitter 稳定。
    """
    B = tril_flat.shape[0]
    L = tril_flat.new_zeros((B, D, D))
    idx = 0
    for i in range(D):
        for j in range(i+1):
            L[:, i, j] = tril_flat[:, idx]
            idx += 1
    # 对角线正定化：softplus + jitter
    diag = torch.nn.functional.softplus(torch.diagonal(L, dim1=1, dim2=2)) + jitter
    for i in range(D):
        L[:, i, i] = diag[:, i]
    return L


def gaussian_nll_fullcov(mu: torch.Tensor, L: torch.Tensor, target: torch.Tensor):
    """
    多元高斯 NLL（均值 mu，协方差 Σ=L L^T）。
    mu / target: (B, D)
    L: (B, D, D) 下三角（对角为正）
    返回标量 loss（batch 平均）
    """
    B, D = mu.shape
    diff = (target - mu).unsqueeze(-1)        # (B,D,1)

    # 解 L z = diff  ->  z = L^{-1} diff
    # 等价 mahal = ||z||^2
    # 使用 triangular_solve 更稳
    z, _ = torch.triangular_solve(diff, L, upper=False)  # (B,D,1)
    mahal = (z.squeeze(-1) ** 2).sum(dim=1)              # (B,)

    # log|Σ| = 2 * sum(log diag(L))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)  # (B,)

    nll = 0.5 * (mahal + logdet + D * math.log(2 * math.pi))                # (B,)
    return nll.mean()

# ===== 放到文件顶部某处（比如 train_epoch 前） =====
def _unpack_forward(out, device):
    """
    更稳健版：
      - 优先用 shape 判断 logvar（是否能 broadcast 到 pred）
      - KL 允许是标量或单元素张量
    """
    import torch

    def _is_scalar_tensor(x):
        return torch.is_tensor(x) and (x.dim() == 0 or x.numel() == 1)

    def _can_broadcast(lv, mu):
        if (lv is None) or (mu is None):
            return False
        if not (torch.is_tensor(lv) and torch.is_tensor(mu)):
            return False
        try:
            _ = mu + lv  # 利用 PyTorch 广播规则试一下
            return True
        except Exception:
            return False

    pred = None
    logvar = None
    kl = torch.tensor(0.0, dtype=torch.float32, device=device)

    if isinstance(out, (tuple, list)):
        L = len(out)

        if L == 4:
            # (mu, logvar, d_rec, kl)
            mu, lv, _, k = out
            pred = mu
            logvar = lv
            kl = k

        elif L == 3:
            a, b, c = out
            pred = a

            # 优先判断 b 是否像 logvar
            if _can_broadcast(b, pred):
                logvar = b
                # c 若像标量，则当 KL；否则忽略（比如 d_rec）
                if _is_scalar_tensor(c):
                    kl = c
            else:
                # b 不像 logvar，那更可能是 KL
                if _is_scalar_tensor(b):
                    kl = b
                # c 若像 logvar，则取 c
                if _can_broadcast(c, pred):
                    logvar = c

        elif L == 2:
            a, b = out
            pred = a

            # b 能 broadcast 到 pred -> 当 logvar；否则当 KL
            if _can_broadcast(b, pred):
                logvar = b
            else:
                if _is_scalar_tensor(b):
                    kl = b
                else:
                    # 兜底：既不是可广播logvar，也不是标量KL，那就当 logvar（或报错都行）
                    logvar = b

        elif L == 1:
            pred = out[0]
        else:
            raise ValueError(f"forward() 返回了长度为 {L} 的 tuple，无法解析")
    else:
        pred = out

    # ---- KL 兜底到 tensor scalar ----
    if kl is None:
        kl = torch.tensor(0.0, dtype=torch.float32, device=device)
    elif not torch.is_tensor(kl):
        kl = torch.as_tensor(kl, dtype=torch.float32, device=device)
    else:
        kl = kl.to(device=device, dtype=torch.float32)

    return pred, logvar, kl



def _resolve_student_df(model=None, default: float = 4.0) -> float:
    """
    统一解析 Student-t 自由度：
      1) model.student_df
      2) ARGS_HOOK.student_df / ARGS_HOOK["student_df"]
      3) fallback default
    """
    # 1) model
    if model is not None and hasattr(model, "student_df"):
        try:
            return float(getattr(model, "student_df"))
        except Exception:
            pass

    # 2) ARGS_HOOK
    hook = globals().get("ARGS_HOOK", None)
    if hook is not None:
        try:
            if isinstance(hook, dict) and ("student_df" in hook):
                return float(hook["student_df"])
            if hasattr(hook, "student_df"):
                return float(getattr(hook, "student_df"))
        except Exception:
            pass

    # 3) default
    return float(default)


def _student_t_nll_diag(pred: torch.Tensor,
                        target: torch.Tensor,
                        logvar: torch.Tensor | None,
                        nu: float = 3.0,
                        model=None) -> torch.Tensor:
    """
    对角 Student-t 负对数似然（逐样本逐通道求和再取均值）

    支持:
      pred/target: (..., D)
      logvar:      (..., D) 或 None

    公式（单维）:
       nll = 0.5*log(nu*pi) + log(s) + 0.5*(nu+1)*log(1 + ((y-mu)^2)/(nu*s^2))
       其中 s = exp(0.5*logvar)
    """
    if pred.shape != target.shape:
        raise ValueError(f"_student_t_nll_diag: pred.shape={pred.shape} != target.shape={target.shape}")

    nu_use = _resolve_student_df(model=model, default=nu)
    nu_t = torch.as_tensor(max(nu_use, 1e-3), device=pred.device, dtype=pred.dtype)

    eps = 1e-8
    diff2 = (target - pred) ** 2

    if logvar is None:
        s2 = torch.ones_like(diff2)
        log_s = torch.zeros_like(diff2)
    else:
        if logvar.shape != pred.shape:
            raise ValueError(f"_student_t_nll_diag: logvar.shape={logvar.shape} != pred.shape={pred.shape}")
        logvar_c = torch.clamp(logvar, min=-10.0, max=6.0)
        s2 = torch.exp(logvar_c).clamp_min(eps)
        log_s = 0.5 * torch.log(s2)

    c0 = 0.5 * torch.log(nu_t * torch.tensor(math.pi, device=pred.device, dtype=pred.dtype))
    t = c0 + log_s + 0.5 * (nu_t + 1.0) * torch.log1p(diff2 / (nu_t * s2 + eps))

    # 对最后一维 D 求和，其他维度取平均
    return t.sum(dim=-1).mean()


def student_t_nll_from_logvar(mu: torch.Tensor,
                              y: torch.Tensor,
                              logvar: torch.Tensor,
                              df: float = 4.0,
                              sigma_floor: float = 0.05,
                              clamp_min: float = -10.0,
                              clamp_max: float = 6.0,
                              eps: float = 1e-6,
                              model=None):
    """
    Heteroscedastic Student-t NLL.

    支持:
      mu,y,logvar: (..., C)

    Returns:
      nll_elem: [same shape as mu]  (per-element negative log-likelihood)
      logvar_c: clamped+floored version used for training
      var:      variance used
    """
    if mu.shape != y.shape or mu.shape != logvar.shape:
        raise ValueError(
            f"student_t_nll_from_logvar: shape mismatch mu={mu.shape}, y={y.shape}, logvar={logvar.shape}"
        )

    logvar_c = torch.clamp(logvar, min=clamp_min, max=clamp_max)
    var = torch.exp(logvar_c).clamp_min(float(sigma_floor) ** 2)
    logvar_c = torch.log(var)

    sq_err = (mu - y) ** 2

    df_use = _resolve_student_df(model=model, default=df)
    nu = torch.as_tensor(max(df_use, 1e-3), device=mu.device, dtype=mu.dtype)

    const = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * torch.tensor(math.pi, device=mu.device, dtype=mu.dtype))
    )

    nll_elem = (-const) + 0.5 * logvar_c + 0.5 * (nu + 1.0) * torch.log1p(sq_err / (nu * var + eps))
    return nll_elem, logvar_c, var


def _recon_loss(pred, target, aux=None, loss_type: str = "gauss", model=None):
    """
    通用重构项：

      - loss_type == "gauss":
          * aux is None            -> MSE
          * aux.shape[-1] == D     -> diag-Gaussian NLL
          * aux.shape[-1] == D*(D+1)//2 -> full-cov Gaussian NLL

      - loss_type == "student":
          * aux is None            -> diag Student-t（单位方差）
          * aux.shape[-1] == D     -> diag Student-t
          * aux.shape[-1] == D*(D+1)//2 -> 回退为 Gaussian full-cov

    支持:
      pred/target: (B, D) 或 (B, T, D)
    """
    if pred.shape != target.shape:
        raise ValueError(f"_recon_loss: pred.shape={pred.shape} != target.shape={target.shape}")

    if pred.dim() not in (2, 3):
        raise ValueError(f"_recon_loss: expect pred dim 2 or 3, got {pred.dim()}")

    D = int(pred.shape[-1])

    if loss_type == "gauss":
        if aux is None:
            return F.mse_loss(pred, target)

        if aux.shape[:-1] != pred.shape[:-1]:
            raise ValueError(f"_recon_loss(gauss): aux prefix shape mismatch aux={aux.shape}, pred={pred.shape}")

        if aux.shape[-1] == D:
            logvar = aux
            return torch.mean(torch.exp(-logvar) * (pred - target) ** 2 + logvar)

        elif aux.shape[-1] == D * (D + 1) // 2:
            # 这里假设你工程里已有 _flat_to_cholesky / gaussian_nll_fullcov
            L = _flat_to_cholesky(aux, D)
            return gaussian_nll_fullcov(pred, L, target)

        else:
            raise ValueError(
                f"_recon_loss(gauss): aux last dim={aux.shape[-1]} not matching D or D*(D+1)//2 (D={D})"
            )

    elif loss_type == "student":
        if aux is None:
            return _student_t_nll_diag(
                pred, target, None,
                nu=4.0,
                model=model
            )

        if aux.shape[:-1] != pred.shape[:-1]:
            raise ValueError(f"_recon_loss(student): aux prefix shape mismatch aux={aux.shape}, pred={pred.shape}")

        if aux.shape[-1] == D:
            logvar = aux
            nu_use = _resolve_student_df(model=model, default=4.0)
            return _student_t_nll_diag(pred, target, logvar, nu=nu_use, model=model)

        elif aux.shape[-1] == D * (D + 1) // 2:
            # Full Student-t 推导复杂，这里仍回退到 Gaussian full-cov
            L = _flat_to_cholesky(aux, D)
            return gaussian_nll_fullcov(pred, L, target)

        else:
            raise ValueError(f"_recon_loss(student): aux dim mismatch (got {aux.shape[-1]})")

    else:
        raise ValueError(f"Unknown loss_type={loss_type}")


def _spectral_loss_from_flat(x_flat, y_flat, n_angles, win, device):
    """
    x_flat,y_flat: (B, A*W) 的窗口输入
    对还原后的 (B, A, W) 沿 W 维做 rFFT 幅值对齐
    """
    if x_flat.shape != y_flat.shape:
        raise ValueError(f"_spectral_loss_from_flat: x_flat.shape={x_flat.shape} != y_flat.shape={y_flat.shape}")

    if x_flat.dim() != 2:
        raise ValueError(f"_spectral_loss_from_flat: expect 2D flat input, got {x_flat.dim()}D")

    B = int(x_flat.shape[0])
    need_dim = int(n_angles) * int(win)
    if int(x_flat.shape[1]) != need_dim:
        raise ValueError(
            f"_spectral_loss_from_flat: last dim={x_flat.shape[1]} != n_angles*win={need_dim}"
        )

    X = x_flat.reshape(B, int(n_angles), int(win))
    Y = y_flat.reshape(B, int(n_angles), int(win))

    XF = torch.fft.rfft(X, dim=-1)
    YF = torch.fft.rfft(Y, dim=-1)
    magX = torch.abs(XF)
    magY = torch.abs(YF)

    K = int(magX.shape[-1])
    w = torch.linspace(0.2, 1.0, K, device=device, dtype=magX.dtype)
    loss = ((magX - magY) ** 2 * w).mean()
    return loss


def _apply_temp_to_logvar(logvar: torch.Tensor | None, temp_scale) -> torch.Tensor | None:
    """
    把温度缩放应用到 logvar（σ -> τ·σ 等价于 logvar += 2logτ）
    支持形状:
      (B, C), (B, T, C), ... 只要最后一维是通道维
    """
    if (logvar is None) or (temp_scale is None):
        return logvar

    tau = F.softplus(temp_scale.tau_raw) + 1e-6   # (C,)
    view_shape = [1] * (logvar.dim() - 1) + [-1]
    return logvar + 2.0 * torch.log(tau.view(*view_shape))


def _lowfreq_seq(seq: torch.Tensor, fs_hz: float, fcut: float):
    """
    低通滤波（0 ~ fcut Hz），用于低频先验
    seq: (M, T, C)
    """
    if seq is None:
        return None

    if seq.dim() != 3:
        raise ValueError(f"_lowfreq_seq expects (M,T,C), got {tuple(seq.shape)}")

    M, T, C = seq.shape
    if T < 2:
        return seq

    S = torch.fft.rfft(seq, dim=1)  # (M, K, C)
    freq = torch.fft.rfftfreq(T, d=1.0 / float(fs_hz)).to(seq.device)

    mask = (freq <= float(fcut))  # (K,)
    while mask.dim() < S.dim():
        mask = mask.unsqueeze(0)

    S_lf = S * mask
    seq_lf = torch.fft.irfft(S_lf, n=T, dim=1)
    return seq_lf

class EMALossBalancer:
    """
    EMA-based loss balancing:
      scaled_i = raw_i * (target / (ema_i + eps))
    with optional clamp to avoid extreme scaling.

    Use: total, scales = balancer(raw_terms, step)
    """
    def __init__(self, names, decay=0.99, eps=1e-8, target=1.0,
                 clamp_min=0.05, clamp_max=20.0, warmup_steps=50, device="cpu"):
        self.names = list(names)
        self.decay = float(decay)
        self.eps = float(eps)
        self.target = float(target)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.warmup_steps = int(warmup_steps)

        self.ema = {n: torch.tensor(0.0, device=device) for n in self.names}
        self.inited = {n: False for n in self.names}

    @torch.no_grad()
    def update_ema(self, raw_terms: dict):
        for n in self.names:
            if n not in raw_terms:
                continue
            v = raw_terms[n]
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), device=self.ema[n].device)
            v = v.detach()
            if not torch.isfinite(v):
                continue
            if (not self.inited[n]) or (self.ema[n].item() == 0.0):
                self.ema[n].copy_(v)
                self.inited[n] = True
            else:
                self.ema[n].mul_(self.decay).add_(v * (1.0 - self.decay))

    def __call__(self, raw_terms: dict, step: int):
        """
        raw_terms: dict(name -> scalar tensor), 已包含语义权重(lambda_xxx)后的项
        step: global step
        returns:
          total_scaled_loss, scales(dict)
        """
        device = self.ema[self.names[0]].device

        # =========================
        # 1) 构造“用于更新 EMA 的安全项”
        #    - 只用 finite
        #    - 只用 > tiny 的项（避免 0 把 EMA 拉成 0 -> scale 飙到 clamp_max）
        #    - 对极端离群做截断（winsorize）
        # =========================
        safe_terms = {}
        tiny = 1e-8

        for n, v in raw_terms.items():
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), device=device)
            else:
                v = v.to(device)

            # 不更新非有限
            if not torch.isfinite(v).all():
                continue

            # 用绝对值判断是否“有效”
            v_abs = v.detach().abs()

            # 太小就不更新 EMA（但仍可参与 total loss）
            if v_abs.item() < tiny:
                continue

            # spike 抑制：如果已经有 EMA 基线，限制 v_abs 不要远超 EMA（避免尖峰把 EMA 拉爆）
            # 允许最大 50 倍（可调：20~100）
            if (n in self.ema) and self.inited.get(n, False):
                base = (self.ema[n].detach().abs() + self.eps)
                cap = 50.0 * base
                v_use = torch.clamp(v_abs, max=cap) * v.detach().sign()
            else:
                v_use = v

            safe_terms[n] = v_use

        if len(safe_terms) > 0 and step >= 0:  # 或 step >= self.warmup_steps//2
            self.update_ema(safe_terms)
        # =========================
        # 2) 计算 scales + total
        #    注意：scale 用 EMA；但 loss 总是用 raw_terms（不丢项）
        # =========================
        scales = {}
        total = None

        for n, v in raw_terms.items():
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), device=device)
            else:
                v = v.to(device)

            if (n in self.ema) and self.inited.get(n, False) and (step >= self.warmup_steps):
                # 若 EMA 非常小（接近 0），直接用 1，避免 scale 直接撞 clamp_max
                denom = (self.ema[n] + self.eps)
                if denom.item() < 1e-6:
                    s = torch.tensor(1.0, device=device)
                else:
                    s = self.target / denom
                    s = torch.clamp(s, self.clamp_min, self.clamp_max)
            else:
                s = torch.tensor(1.0, device=device)

            scales[n] = s
            term = v * s
            total = term if total is None else (total + term)

        if total is None:
            total = torch.tensor(0.0, device=device)

        return total, scales


class PhysEMANormalizer:
    """
    用 EMA 估计 misfit/prior 的典型尺度，并将其归一化到 O(1)
    """
    def __init__(self, decay=0.99, eps=1e-8, clamp_scale=(1e-3, 1e3), device="cpu"):
        self.decay = float(decay)
        self.eps = float(eps)
        self.clamp_min, self.clamp_max = clamp_scale
        self.device = torch.device(device)

        self._ema_misfit = None
        self._ema_prior  = None
        self.inited = False

    @torch.no_grad()
    def update(self, misfit: torch.Tensor, prior: torch.Tensor):
        m = misfit.detach().float().abs().mean().to(self.device)
        p = prior.detach().float().abs().mean().to(self.device)

        if not self.inited:
            self._ema_misfit = m.clone()
            self._ema_prior  = p.clone()
            self.inited = True
        else:
            self._ema_misfit.mul_(self.decay).add_(m * (1 - self.decay))
            self._ema_prior.mul_(self.decay).add_(p * (1 - self.decay))

        # clamp 防止极端 batch 让尺度崩掉
        self._ema_misfit.clamp_(self.clamp_min, self.clamp_max)
        self._ema_prior.clamp_(self.clamp_min, self.clamp_max)

    def normalize(self, misfit: torch.Tensor, prior: torch.Tensor):
        if not self.inited:
            return misfit, prior, None, None

        # ✅ scale 作为常数，不需要梯度
        s_m = self._ema_misfit.detach().to(device=misfit.device, dtype=misfit.dtype)
        s_p = self._ema_prior.detach().to(device=prior.device, dtype=prior.dtype)

        return misfit / (s_m + self.eps), prior / (s_p + self.eps), s_m, s_p

import torch
import torch.nn.functional as F

def lateral_second_order_loss_line(
    pred: torch.Tensor,
    metas,
    beta: float = 0.5,
    use_smoothl1: bool = True,
    chan_weight: torch.Tensor | None = None,   # shape (1,C)
    max_gap: int = 8,                          # 允许的 line 索引间隔
):
    """
    二阶横向连续性：对固定 (t, tau) 的 line 方向 l 做二阶差分约束
      d2 = p(l_{i-1}) - 2*p(l_i) + p(l_{i+1})

    pred : (B,C)
    metas: list of (l,t,tau,is_well)
    返回:
      loss: 标量 tensor (可导)
      triples: 使用到的三元组数量（用于日志）
    """
    device = pred.device
    B, C = pred.shape

    if B < 3:
        return pred.new_tensor(0.0), 0

    # (t, tau) -> [(l, idx), ...]
    groups = {}
    for i, m in enumerate(metas):
        if not (isinstance(m, (list, tuple)) and len(m) >= 3):
            continue
        l, t, tau = int(m[0]), int(m[1]), int(m[2])
        groups.setdefault((t, tau), []).append((l, i))

    d2_list = []
    triples = 0

    for (t, tau), lst in groups.items():
        if len(lst) < 3:
            continue
        lst.sort(key=lambda x: x[0])  # sort by l

        # 用相邻三点构三元组 (i-1, i, i+1)
        for k in range(1, len(lst) - 1):
            l0, i0 = lst[k - 1]
            l1, i1 = lst[k]
            l2, i2 = lst[k + 1]

            # gap 约束：左右相邻都不能太远
            if (l1 - l0) > max_gap or (l2 - l1) > max_gap:
                continue

            p0 = pred[i0]
            p1 = pred[i1]
            p2 = pred[i2]
            d2 = p0 - 2.0 * p1 + p2  # (C,)
            d2_list.append(d2)
            triples += 1

    if triples == 0:
        return pred.new_tensor(0.0), 0

    d2_all = torch.stack(d2_list, dim=0)  # (T,C)

    # 通道权重
    if chan_weight is not None:
        d2_all = d2_all * chan_weight  # broadcast to (T,C)

    if use_smoothl1:
        # 让 d2 -> 0
        loss = F.smooth_l1_loss(d2_all, torch.zeros_like(d2_all), reduction="sum", beta=beta)
    else:
        loss = (d2_all ** 2).sum()

    # ✅ 关键：按 triples 和通道数归一，稳定量级（也更利于调 lambda）
    loss = loss / (triples + 1e-6)
    loss = loss / (C + 1e-6)

    return loss, triples

import torch
import torch.nn.functional as F

def lateral_tv_loss_line_torch(
    pred: torch.Tensor,
    metas,
    max_gap: int = 3,
    beta: float = 0.5,
    chan_weight: torch.Tensor | None = None,
):
    """
    纯 torch 横向连续性（line 方向）：
    - 在 batch 内，根据 metas 把点按 (t, tau) 分组
    - 每组内按 l 排序，连接相邻点（gap<=max_gap）形成 pair
    - loss = SmoothL1( pred[j]-pred[i] , 0 )  -> 鼓励横向平滑/连续
    返回：lat_raw(torch.Tensor, requires_grad=True), lat_pairs(int)
    """
    device = pred.device
    B, C = pred.shape
    if chan_weight is None:
        chan_weight = pred.new_ones(1, C)  # (1,C)
    else:
        chan_weight = chan_weight.to(device=device, dtype=pred.dtype).view(1, C)

    # -------- 1) 用 python 建 pair（索引选择不可导没关系，只要 pred 的算子可导）--------
    groups = {}  # key=(t,tau) -> list[(l, idx)]
    for i, m in enumerate(metas):
        if not (isinstance(m, (list, tuple)) and len(m) >= 3):
            continue
        l, t, tau = int(m[0]), int(m[1]), int(m[2])
        groups.setdefault((t, tau), []).append((l, i))

    pairs_i = []
    pairs_j = []
    for (_, _), lst in groups.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: x[0])  # sort by l
        for k in range(len(lst) - 1):
            l1, i1 = lst[k]
            l2, i2 = lst[k + 1]
            if abs(l2 - l1) <= max_gap:
                pairs_i.append(i1)
                pairs_j.append(i2)

    lat_pairs = len(pairs_i)
    if lat_pairs == 0:
        return pred.new_tensor(0.0), 0

    idx_i = torch.as_tensor(pairs_i, device=device, dtype=torch.long)
    idx_j = torch.as_tensor(pairs_j, device=device, dtype=torch.long)

    # -------- 2) 关键：loss 必须直接从 pred 的张量运算得到（保证梯度）--------
    dp = pred.index_select(0, idx_j) - pred.index_select(0, idx_i)  # (P,C)
    dp = dp * chan_weight  # channel weight

    # SmoothL1 to 0
    lat_raw = F.smooth_l1_loss(dp, torch.zeros_like(dp), reduction="mean", beta=beta)
    lat_raw = torch.nan_to_num(lat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    return lat_raw, lat_pairs

class PositiveShift:
    """
    把任意标量 loss 平移到正数：x_pos = x_raw + shift
    shift 基于历史最小值，保证 x_pos >= floor
    """
    def __init__(self, floor: float = 1e-3, freeze_after: int = 2000):
        self.floor = float(floor)
        self.freeze_after = int(freeze_after)
        self.min_seen = float("inf")
        self.shift = 0.0
        self.n = 0

    def update(self, x: float):
        self.n += 1
        if self.n <= self.freeze_after:
            if x < self.min_seen:
                self.min_seen = x
            self.shift = max(0.0, -self.min_seen + self.floor)

    def apply(self, x_tensor):
        if self.shift == 0.0:
            return x_tensor
        return x_tensor + x_tensor.new_tensor(self.shift)

class LossShiftEMA:
    """
    只用于把可能为负的 loss（尤其 NLL）平移到正数区间，供显示/ratio/EMA-balancer 使用。
    不影响训练梯度：内部全部 detach + clamp。
    """
    def __init__(self, decay=0.99, eps=1e-6, device="cpu"):
        self.decay = float(decay)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.inited = False
        self.ema_neg_attr = None
        self.ema_neg_well = None  # 可选：well_term 也可能负

    @torch.no_grad()
    def _ema_update(self, ema, x):
        # ema <- decay*ema + (1-decay)*x
        if ema is None:
            return x.clone()
        ema.mul_(self.decay).add_(x * (1.0 - self.decay))
        return ema

    @torch.no_grad()
    def update(self, attr_loss=None, well_term=None, enabled=True):
        if not enabled:
            return
        if not self.inited:
            self.inited = True

        if attr_loss is not None:
            a = attr_loss.detach().float().to(self.device)
            neg = F.relu(-a)  # 只跟踪“负的幅度”
            self.ema_neg_attr = self._ema_update(self.ema_neg_attr, neg)

        if well_term is not None:
            w = well_term.detach().float().to(self.device)
            neg = F.relu(-w)
            self.ema_neg_well = self._ema_update(self.ema_neg_well, neg)

    @torch.no_grad()
    def shift_attr(self, attr_loss: torch.Tensor):
        if (not self.inited) or (self.ema_neg_attr is None):
            # 没初始化就至少 clamp 到正数，避免画图炸
            return torch.clamp(attr_loss.detach(), min=self.eps)
        s = self.ema_neg_attr.to(device=attr_loss.device, dtype=attr_loss.dtype)
        return torch.clamp(attr_loss.detach() + s + self.eps, min=self.eps)

    @torch.no_grad()
    def shift_well(self, well_term: torch.Tensor):
        if (not self.inited) or (self.ema_neg_well is None):
            return torch.clamp(well_term.detach(), min=0.0)
        s = self.ema_neg_well.to(device=well_term.device, dtype=well_term.dtype)
        return torch.clamp(well_term.detach() + s, min=0.0)



@torch.no_grad()
def predict_section_t_fused(
    model,
    ds,                 # VolumeWindowDataset 实例（已经有 ds.X_mean/X_std, ds.Y_mean/Y_std）
    device,
    t_idx: int,         # 固定 trace index
    l_ids=None,         # 要预测哪些 line（默认全 L）
    tau_min=None,       # tau 范围（默认 [h, N-h-1]）
    tau_max=None,
    batch_size: int = 256,
    fuse_radius_tau: int = 2,        # 时间融合半径（2 -> 5点核）
    fuse_sigma_tau: float = 1.0,     # 高斯 sigma
    fuse_kind: str = "gaussian",     # "gaussian" / "box" / "median"
    fuse_radius_line: int = 0,       # 可选：沿 line 再融合（建议 0~1）
    fuse_sigma_line: float = 1.0,
):
    """
    输出：
      pred_raw  : [L_sel, T_len, 3]  未融合（逐 tau 独立预测）
      pred_fuse : [L_sel, T_len, 3]  融合后（更连续）
      tau_ids   : [T_len]            对应的 tau index
      l_ids     : [L_sel]            对应的 line index
    """
    model.eval()
    L, T, A, N = ds.stack.shape
    h = ds.win // 2

    if l_ids is None:
        l_ids = np.arange(L, dtype=np.int32)
    else:
        l_ids = np.asarray(l_ids, dtype=np.int32)

    if tau_min is None:
        tau_min = h
    if tau_max is None:
        tau_max = N - h - 1
    tau_ids = np.arange(tau_min, tau_max + 1, dtype=np.int32)

    # --------- 先逐点预测（stride=1）---------
    # 结果存 [L_sel, T_len, 3]
    pred_raw = np.zeros((len(l_ids), len(tau_ids), 3), dtype=np.float32)

    # 预取归一化参数
    X_mean = ds.X_mean.astype(np.float32)  # [A]
    X_std  = ds.X_std.astype(np.float32)   # [A]
    X_std  = np.maximum(X_std, 1e-6)

    # 批处理组织：把所有 (li, tau) flatten 成一个列表
    pairs = [(i_li, int(li), int(t_idx), int(tau))
             for i_li, li in enumerate(l_ids)
             for tau in tau_ids]

    def _batch_to_tensor(batch_pairs):
        # 返回 X: [B, A, line_ctx, win]
        Xs = []
        for (_, li, ti, tau) in batch_pairs:
            X = ds._get_x_patch_25d(li, ti, tau)  # [A, line_ctx, win] float32
            Xn = (X - X_mean[:, None, None]) / X_std[:, None, None]
            Xs.append(Xn)
        Xb = np.stack(Xs, axis=0)  # [B,A,line_ctx,win]
        return torch.from_numpy(Xb).float()

    for s in range(0, len(pairs), batch_size):
        batch_pairs = pairs[s:s+batch_size]
        Xb = _batch_to_tensor(batch_pairs).to(device, non_blocking=(device.type=="cuda"))

        out = model(Xb, return_kl=False) if "return_kl" in model.forward.__code__.co_varnames else model(Xb)
        # out 可能是 (mu, logvar, kl) 或 mu
        if isinstance(out, (tuple, list)):
            mu = out[0]
        else:
            mu = out

        mu = mu.detach()
        # 反归一化到物理域
        mu_phys = ds.denormalize_y(mu).detach().cpu().numpy()  # [B,3]

        # 写回 pred_raw
        for k, (i_li, li, ti, tau) in enumerate(batch_pairs):
            j_tau = int(tau - tau_min)
            pred_raw[i_li, j_tau, :] = mu_phys[k]

    # --------- 融合（沿 tau）---------
    pred_fuse = pred_raw.copy()

    if fuse_radius_tau > 0:
        K = 2 * fuse_radius_tau + 1

        if fuse_kind == "gaussian":
            xs = np.arange(-fuse_radius_tau, fuse_radius_tau + 1, dtype=np.float32)
            w = np.exp(-0.5 * (xs / max(fuse_sigma_tau, 1e-6))**2)
            w = w / (w.sum() + 1e-12)  # [K]
        elif fuse_kind == "box":
            w = np.ones((K,), dtype=np.float32) / float(K)
        elif fuse_kind == "median":
            w = None
        else:
            raise ValueError(f"Unknown fuse_kind={fuse_kind}")

        # 用 torch 做 1D conv（更快 & pad 更方便）
        x = torch.from_numpy(pred_raw).permute(0, 2, 1).contiguous()  # [L_sel, 3, T_len]
        x = x.float()

        if fuse_kind == "median":
            # 中值滤波（简单实现：滑窗堆叠再取 median）
            pad = fuse_radius_tau
            x_pad = F.pad(x, (pad, pad), mode="reflect")  # [L,3,T+2pad]
            chunks = [x_pad[:, :, i:i+len(tau_ids)] for i in range(0, 2*pad+1)]
            x_stack = torch.stack(chunks, dim=0)  # [K,L,3,T]
            x_med = x_stack.median(dim=0).values  # [L,3,T]
            y = x_med
        else:
            w_t = torch.from_numpy(w).view(1, 1, K).float()
            w_t = w_t.to(x.device)
            pad = fuse_radius_tau
            x_pad = F.pad(x, (pad, pad), mode="reflect")
            y = F.conv1d(x_pad, w_t.expand(3, 1, K), groups=3)

        pred_fuse = y.permute(0, 2, 1).cpu().numpy()  # [L_sel,T,3]

    # --------- 可选：沿 line 再轻微融合（一般 0~1 就够）---------
    if fuse_radius_line > 0:
        K = 2 * fuse_radius_line + 1
        xs = np.arange(-fuse_radius_line, fuse_radius_line + 1, dtype=np.float32)
        w = np.exp(-0.5 * (xs / max(fuse_sigma_line, 1e-6))**2)
        w = w / (w.sum() + 1e-12)

        x = torch.from_numpy(pred_fuse).permute(2, 1, 0).contiguous()  # [3,T,L]
        x = x.float()
        w_t = torch.from_numpy(w).view(1, 1, K).float()
        pad = fuse_radius_line
        x_pad = F.pad(x, (pad, pad), mode="reflect")
        y = F.conv1d(x_pad, w_t.expand(3, 1, K), groups=3)  # [3,T,L]
        pred_fuse = y.permute(2, 1, 0).cpu().numpy()  # [L,T,3]

    return pred_raw, pred_fuse, tau_ids, l_ids

def build_scale_tril_3x3(chol_params: torch.Tensor,
                         diag_eps: float = 1e-4,
                         diag_max: float = 3.0,
                         offdiag_scale: float = 0.35) -> torch.Tensor:
    """
    chol_params: (B, win, 6)
    返回:
      L: (B, win, 3, 3), lower-triangular, diagonal > 0

    参数顺序约定:
      [a, b, c, d, e, f] ->
      [[l11,   0,   0],
       [l21, l22,   0],
       [l31, l32, l33]]

    稳定化策略：
      - 对角: softplus + clamp
      - 非对角: tanh 压缩，再按 sqrt(diag_i * diag_j) 缩放
    """
    import torch
    import torch.nn.functional as F

    assert chol_params.shape[-1] == 6, f"chol_params last dim must be 6, got {chol_params.shape}"

    a, b, c, d, e, f = torch.unbind(chol_params, dim=-1)

    # -----------------------------
    # diagonal > 0
    # -----------------------------
    l11 = F.softplus(a) + diag_eps
    l22 = F.softplus(c) + diag_eps
    l33 = F.softplus(f) + diag_eps

    # clamp diagonal to avoid extreme variance
    l11 = torch.clamp(l11, min=diag_eps, max=diag_max)
    l22 = torch.clamp(l22, min=diag_eps, max=diag_max)
    l33 = torch.clamp(l33, min=diag_eps, max=diag_max)

    # -----------------------------
    # off-diagonal stabilization
    #   raw -> tanh -> scaled by geometric mean of diagonals
    # -----------------------------
    s12 = torch.sqrt(torch.clamp(l11 * l22, min=diag_eps))
    s13 = torch.sqrt(torch.clamp(l11 * l33, min=diag_eps))
    s23 = torch.sqrt(torch.clamp(l22 * l33, min=diag_eps))

    l21 = torch.tanh(b) * offdiag_scale * s12
    l31 = torch.tanh(d) * offdiag_scale * s13
    l32 = torch.tanh(e) * offdiag_scale * s23

    # -----------------------------
    # assemble
    # -----------------------------
    B, W = a.shape
    L = torch.zeros(B, W, 3, 3, device=chol_params.device, dtype=chol_params.dtype)

    L[..., 0, 0] = l11
    L[..., 1, 0] = l21
    L[..., 1, 1] = l22
    L[..., 2, 0] = l31
    L[..., 2, 1] = l32
    L[..., 2, 2] = l33

    return L

def scale_tril_to_cov(L: torch.Tensor) -> torch.Tensor:
    """
    L: (B, win, 3, 3)
    cov = L @ L^T
    """
    return L @ L.transpose(-1, -2)

def multivariate_student_t_nll(y: torch.Tensor,
                               mu: torch.Tensor,
                               scale_tril: torch.Tensor,
                               nu,
                               reduction: str = "mean") -> torch.Tensor:
    assert y.shape == mu.shape
    assert y.shape[-1] == 3
    D = y.shape[-1]

    if not torch.is_tensor(nu):
        nu = torch.tensor(float(nu), device=y.device, dtype=y.dtype)
    else:
        nu = nu.to(device=y.device, dtype=y.dtype)

    nu = torch.clamp(nu, min=1e-3)

    diff = (y - mu).unsqueeze(-1)  # (B, win, 3, 1)

    maha_vec = torch.cholesky_solve(diff, scale_tril)  # (B, win, 3, 1)
    delta = (diff.transpose(-1, -2) @ maha_vec).squeeze(-1).squeeze(-1)  # (B, win)
    delta = torch.clamp(delta, min=0.0)

    diag_L = torch.diagonal(scale_tril, dim1=-2, dim2=-1)  # (B, win, 3)
    logdet = 2.0 * torch.sum(torch.log(diag_L + 1e-12), dim=-1)  # (B, win)

    c = (
            torch.lgamma((nu + D) / 2.0)
            - torch.lgamma(nu / 2.0)
            + 0.5 * logdet
            + (D / 2.0) * torch.log(nu * torch.tensor(math.pi, device=y.device, dtype=y.dtype))
    )

    t = ((nu + D) / 2.0) * torch.log1p(delta / nu)

    nll = c + t

    if reduction == "none":
        return nll
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll.mean()

def sample_multivariate_student_t(mu: torch.Tensor,
                                  scale_tril: torch.Tensor,
                                  nu,
                                  n_samples: int,
                                  max_scale: float = 3.0,
                                  max_mahalanobis_scale: float = 6.0) -> torch.Tensor:
    """
    稳健版 multivariate Student-t 采样

    参数:
      mu:         (..., 3)
      scale_tril: (..., 3, 3)
      nu:         scalar df
      n_samples:  采样数
      max_scale:  对 sqrt(nu/u) 的上限裁剪，抑制极端重尾
      max_mahalanobis_scale:
                  对最终 whitened sample 幅度再做一次软裁剪

    返回:
      samples: (n_samples, ..., 3)
    """
    import torch

    device = mu.device
    dtype = mu.dtype

    if not torch.is_tensor(nu):
        nu = torch.tensor(float(nu), device=device, dtype=dtype)
    else:
        nu = nu.to(device=device, dtype=dtype)

    nu = torch.clamp(nu, min=1e-3)

    base_shape = mu.shape[:-1]
    D = mu.shape[-1]
    assert D == 3, f"Expected last dim=3, got {D}"

    # -----------------------------
    # Gaussian base
    # -----------------------------
    z = torch.randn((n_samples,) + base_shape + (D, 1), device=device, dtype=dtype)

    # correlated Gaussian part
    Lz = torch.matmul(scale_tril.unsqueeze(0), z).squeeze(-1)   # (n_samples, ..., 3)

    # -----------------------------
    # Student-t radial scaling
    #   u ~ Chi2(nu) = Gamma(nu/2, rate=1/2)
    # -----------------------------
    gamma = torch.distributions.Gamma(concentration=nu / 2.0, rate=0.5)
    u = gamma.sample((n_samples,) + base_shape).to(device=device, dtype=dtype)

    scale = torch.sqrt(nu / (u + 1e-12)).unsqueeze(-1)   # (n_samples, ..., 1)

    # ✅ 稳健：裁掉极端尾部
    if max_scale is not None and max_scale > 0:
        scale = torch.clamp(scale, max=float(max_scale))

    # first pass
    delta = Lz * scale

    # ✅ 再做一次软裁剪：限制最终扰动过大
    if max_mahalanobis_scale is not None and max_mahalanobis_scale > 0:
        delta = max_mahalanobis_scale * torch.tanh(delta / max_mahalanobis_scale)

    samples = mu.unsqueeze(0) + delta
    return samples

def train_epoch_vaim(
    model, loader, optimizer_main, optimizer_phys, beta_kl, device,
    physics_loss_fn=None, phys_norm=None, lambda_well: float = 0.0, denormalize_fn=None,
    kl_scheduler=None, epoch_idx: int = 1, temp_scale=None,
    lambda_lat=0.05, warm_lat_ep=8, lat_max_gap: int = 8,
    dt: float = 0.001,
    fdom_vaim: float = 45.0,
    lambda_vaim: float = 0.10,
    lf_cut: float = 8.0,
    lowfreq_weight: float = 0.05,
    lambda_recon: float = 0.02,
    loss_balancer=None,
    global_step_start: int = 0,
    warmup_phys: int = 10,
    phys_debug_epochs=(10, 11, 12),
    ema_decay: float = 0.99,
    ema_target: float = 1.0,
    ema_warmup_steps: int = 100,
    ema_clamp_min: float = 0.05,
    ema_clamp_max: float = 20.0,
    grad_debug_every: int = 0,
    grad_debug_topk: int = 12,
    print_ema_every: int = 200,
    print_phys_debug_every: int = 200,
    y_mean_t=None,
    y_std_t=None,
    rhob_std_thresh: float = 1e-6,
    rhob_check_every: int = 200,
    auto_w_phys: bool = True,
    phys_auto_every: int = 200,
    phys_auto_start_ep: int = 3,
    phys_target_ratio: float = 10.0,
    phys_ema: float = 0.2,
    phys_w_min: float = 1e-3,
    phys_w_max: float = 2.0,
    shift_pack: dict = None,
):
    """
    Sequence + multivariate Student-t version
    ----------------------------------------
    约定：
      pred / y / y_bl : (B, win, 3)
      d_rec           : (B, win, A)
      chol_params     : (B, win, 6)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    # ---------------------------
    # dt_eff / fs_eff
    # ---------------------------
    ds_local = getattr(loader, "dataset", None)
    time_stride = int(getattr(ds_local, "time_stride", 1)) if ds_local is not None else 1
    if time_stride < 1:
        time_stride = 1
    dt_eff = float(dt) * float(time_stride)
    fs_eff = 1.0 / dt_eff

    # ---------------------------
    # ✅ Scheme A shifters
    # ---------------------------
    if shift_pack is None:
        shift_pack = getattr(train_epoch_vaim, "_shift_pack", None)
        if shift_pack is None:
            shift_pack = {
                "total": PositiveShift(floor=1e-3, freeze_after=2000),
                "attr":  PositiveShift(floor=1e-3, freeze_after=2000),
                "well":  PositiveShift(floor=1e-3, freeze_after=2000),
            }
            setattr(train_epoch_vaim, "_shift_pack", shift_pack)

    shift_total = shift_pack["total"]
    shift_attr  = shift_pack["attr"]
    shift_well  = shift_pack["well"]

    # ---------------------------
    # torch denorm
    # ---------------------------
    _warned_denorm = False
    _Y_mean = None
    _Y_std = None
    if (y_mean_t is not None) and (y_std_t is not None):
        _Y_mean = torch.as_tensor(y_mean_t, device=device, dtype=torch.float32).view(1, 1, -1)
        _Y_std  = torch.as_tensor(y_std_t,  device=device, dtype=torch.float32).view(1, 1, -1)

    def denorm_t(z: torch.Tensor) -> torch.Tensor:
        nonlocal _warned_denorm
        if denormalize_fn is not None:
            out = denormalize_fn(z)
            if torch.is_tensor(out):
                return out
            if not _warned_denorm:
                print("[DENORM][WARN] denormalize_fn returned non-torch; fallback to y_mean/y_std or identity.", flush=True)
                _warned_denorm = True
        if (_Y_mean is not None) and (_Y_std is not None):
            if z.dim() == 2:
                Ym = _Y_mean.view(1, -1).to(z.dtype)
                Ys = _Y_std.view(1, -1).to(z.dtype)
                return z * Ys + Ym
            return z * _Y_std.to(z.dtype) + _Y_mean.to(z.dtype)
        if not _warned_denorm:
            print("[DENORM][WARN] No denormalize_fn and no y_mean/y_std -> identity.", flush=True)
            _warned_denorm = True
        return z

    # ---------------------------
    # extract center line seismic
    # returns:
    #   X_centerline_seq : (B, win, A)
    #   X_center         : (B, A)
    # ---------------------------
    def _extract_centerline_and_center(X: torch.Tensor, A_hint: int = None):
        def _slice_channels(x: torch.Tensor) -> torch.Tensor:
            if A_hint is None:
                return x
            A = int(A_hint)
            if A <= 0:
                return x
            if x.dim() >= 2 and x.size(1) > A:
                return x[:, :A, ...].contiguous()
            return x

        if X.dim() == 2:
            if A_hint is None or int(A_hint) <= 0:
                raise ValueError("X is 2D [B,A*win] but A_hint is not provided.")
            A_hint = int(A_hint)
            in_dim = int(X.shape[1])
            if in_dim % A_hint != 0:
                raise ValueError(f"X.shape[1]={in_dim} not divisible by A_hint={A_hint}")
            win = in_dim // A_hint
            X_centerline = X.reshape(int(X.shape[0]), A_hint, win).contiguous()  # (B,A,win)
            X_centerline = _slice_channels(X_centerline)
            X_center = X_centerline[:, :, win // 2].contiguous()                 # (B,A)
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()         # (B,win,A)
            return X_centerline_seq, X_center

        if X.dim() == 3:
            # X: [B, C, win]
            X_centerline = _slice_channels(X.contiguous())                       # (B,A,win)
            X_center = X_centerline[:, :, X_centerline.shape[-1] // 2].contiguous()
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()         # (B,win,A)
            return X_centerline_seq, X_center

        if X.dim() == 4:
            # X: [B, C, line_ctx, win] -> take center line -> [B, C, win]
            lc = int(X.shape[2])
            X_centerline = X[:, :, lc // 2, :].contiguous()                      # (B,C,win)
            X_centerline = _slice_channels(X_centerline)
            X_center = X_centerline[:, :, X_centerline.shape[-1] // 2].contiguous()
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()         # (B,win,A)
            return X_centerline_seq, X_center

        raise ValueError(f"Unexpected X.dim={X.dim()} shape={tuple(X.shape)}")

    # ---------------------------
    # physics warmup freeze
    # ---------------------------
    phys_model = getattr(physics_loss_fn, "physics_model", None) if physics_loss_fn is not None else None
    if phys_model is not None:
        freeze = (int(epoch_idx) < int(warmup_phys))
        for p in phys_model.parameters():
            p.requires_grad_(not freeze)
        if int(epoch_idx) == int(warmup_phys) - 1:
            print(f"[PHYS] warmup active (<{warmup_phys}) -> physics_model frozen this epoch.", flush=True)
        if int(epoch_idx) == int(warmup_phys):
            print(f"[PHYS] warmup done (>= {warmup_phys}) -> physics_model unfrozen from this epoch.", flush=True)

    # ------------------ train begin ------------------
    model.train()
    progress_bar = tqdm(loader, desc="Training")
    iters_per_epoch = len(loader)

    total_samples = 0

    # RAW totals
    total_loss = 0.0
    total_attr = 0.0
    total_recon_raw = 0.0
    total_recon_term = 0.0
    total_kl = 0.0
    total_kl_w = 0.0
    total_phys = 0.0
    total_lat = 0.0
    total_lat_pairs = 0
    total_mse_for_log = 0.0
    total_well_loss = 0.0
    total_well_term = 0.0
    total_gate_reg = 0.0   # ✅ NEW

    # FOR_LOG totals
    total_loss_for_log = 0.0
    total_attr_for_log = 0.0
    total_well_term_for_log = 0.0

    rhob_bad = False
    _warned_rhob_bad = False
    _warned_no_drec = False

    # sequence weights: (1,1,3)
    chan_weight = torch.tensor([1.0, 1.0, 12.0], device=device).view(1, 1, -1)

    for it, (X, y, y_bl, metas) in enumerate(progress_bar):
        step = int(global_step_start) + int(it)
        nonblock = (device.type == "cuda")

        X = X.to(device, non_blocking=nonblock)
        y = y.to(device, non_blocking=nonblock)                 # (B,win,3)
        y_bl = y_bl.to(device, non_blocking=nonblock) if y_bl is not None else None

        if y_bl is not None:
            y_bl = torch.nan_to_num(y_bl, nan=0.0, posinf=0.0, neginf=0.0)
            y_bl = torch.clamp(y_bl, min=-8.0, max=8.0)

        B = int(X.shape[0])

        optimizer_main.zero_grad(set_to_none=True)
        if optimizer_phys is not None:
            optimizer_phys.zero_grad(set_to_none=True)

        # -------- forward --------
        try:
            out = model(X, return_kl=True)
        except TypeError:
            out = model(X)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(f"[TRAIN] Expected tuple/list model output, got type={type(out)}")

        extra = None
        if len(out) >= 5:
            mu_attr, chol_params, d_rec, kl, extra = out[:5]
        elif len(out) >= 4:
            mu_attr, chol_params, d_rec, kl = out[:4]
        else:
            raise RuntimeError(
                "[TRAIN] Expected model output like "
                "(mu_attr, chol_params, d_rec, kl[, extra]), "
                f"but got len={len(out)}"
            )

        pred = mu_attr.contiguous()                              # (B,win,3)
        y = y.contiguous()

        if chol_params is None:
            raise RuntimeError("[TRAIN] chol_params is None. multivariate Student-t requires chol_params.")
        chol_params = chol_params.contiguous()
        L_attr = build_scale_tril_3x3(chol_params)              # (B,win,3,3)

        if d_rec is not None and torch.is_tensor(d_rec):
            d_rec = d_rec.contiguous()

        if pred.dim() != 3 or pred.size(-1) < 3:
            raise RuntimeError(f"[TRAIN] pred must be (B,win,3), got shape={tuple(pred.shape)}")
        if y.dim() != 3 or y.size(-1) < 3:
            raise RuntimeError(f"[TRAIN] y must be (B,win,3), got shape={tuple(y.shape)}")

        C = int(pred.shape[-1])

        # ---- RHOB 退化检测（物理域 y std）----
        if (not rhob_bad) and ((it < 3) or ((it % int(rhob_check_every) == 0) and (it > 0))):
            try:
                y_phys = denorm_t(y)  # (B,win,3)
                if torch.is_tensor(y_phys) and (y_phys.dim() == 3) and (y_phys.size(-1) >= 3):
                    rh_std = float(y_phys[..., 2].detach().float().std().cpu().item())
                    if rh_std < float(rhob_std_thresh):
                        rhob_bad = True
                        if not _warned_rhob_bad:
                            print(
                                f"[TRAIN][RHOB][AUTO] degenerate RHOB: std={rh_std:.3e} < {float(rhob_std_thresh):.3e}. "
                                f"Mask RHOB in attr/vaim/well/lat/physics.",
                                flush=True
                            )
                            _warned_rhob_bad = True
            except Exception:
                pass

        chan_mask = pred.new_ones((1, 1, C))
        if (C >= 3) and rhob_bad:
            chan_mask[..., 2] = 0.0

        pred_m = pred * chan_mask
        y_m    = y * chan_mask
        y_bl_m = (y_bl * chan_mask) if (y_bl is not None and torch.is_tensor(y_bl) and y_bl.shape == y.shape) else y_bl

        # weights
        w = chan_weight
        if w.shape[-1] != C:
            w = torch.ones((1, 1, C), device=device, dtype=pred.dtype)
        w = w * chan_mask
        den_w = (w.sum() + 1e-12)

        # ---------------------------
        # recon
        # d_rec: (B,win,A)
        # X_centerline_seq: (B,win,A)
        # ---------------------------
        loss_recon = pred.new_tensor(0.0)
        recon_term = pred.new_tensor(0.0)
        if d_rec is not None and torch.is_tensor(d_rec):
            A = int(d_rec.shape[-1])
            X_centerline_seq, _ = _extract_centerline_and_center(X, A_hint=A)

            if torch.is_tensor(X_centerline_seq) and X_centerline_seq.dim() == 3 and X_centerline_seq.size(-1) != A:
                X_centerline_seq = X_centerline_seq[..., :A]

            loss_recon = _recon_loss(d_rec, X_centerline_seq, aux=None, loss_type="student")
            recon_term = float(lambda_recon) * loss_recon
        else:
            if not _warned_no_drec:
                print("[RECON][WARN] d_rec is None. recon_term=0.", flush=True)
                _warned_no_drec = True

        # ---------------------------
        # mse_for_log (per-sample mean over win+chan)
        # ---------------------------
        err = (pred_m - y_m)
        sq_err = err ** 2
        mse_for_log = ((sq_err * w).sum(dim=(1, 2)) / (float(pred.shape[1]) * den_w)).mean()

        # ---------------------------
        # attr_loss: multivariate Student-t NLL
        # ---------------------------
        df_use = float(getattr(model, "student_df", 4.0))
        attr_loss = multivariate_student_t_nll(
            y=y_m,
            mu=pred_m,
            scale_tril=L_attr,
            nu=df_use,
            reduction="mean"
        )

        # ---------------------------
        # ✅ gate entropy regularization
        #   防止 mixture 迅速塌成单分量
        # ---------------------------
        gate_reg = pred.new_tensor(0.0)
        if isinstance(extra, dict) and ("gate" in extra):
            gate = extra["gate"]   # [B,win,K]
            if torch.is_tensor(gate):
                gate_safe = gate.clamp_min(1e-8)
                gate_entropy = -(gate_safe * torch.log(gate_safe)).sum(dim=-1).mean()
                gate_reg = -1e-3 * gate_entropy   # 轻微鼓励混合
                gate_reg = torch.nan_to_num(gate_reg, nan=0.0, posinf=0.0, neginf=0.0)


        # ---------------------------
        # KL
        # ---------------------------
        kl_term = kl.mean() if torch.is_tensor(kl) and kl.dim() > 0 else kl
        if not torch.is_tensor(kl_term):
            kl_term = pred.new_tensor(float(kl_term))

        beta_now = float(beta_kl)
        if kl_scheduler is not None:
            beta_now = float(kl_scheduler(epoch_idx, it, iters_per_epoch, beta_max=float(beta_kl)))
        kl_term_w = float(beta_now) * kl_term

        # ==========================================================
        # well term（sequence版）
        # ==========================================================
        well_mask_list = [m[3] if (isinstance(m, (list, tuple)) and len(m) >= 4) else False for m in metas]
        mask = torch.tensor(well_mask_list, dtype=torch.bool, device=pred.device)
        well_cnt = int(mask.sum().item())

        loss_well = pred.new_tensor(0.0)
        well_term = pred.new_tensor(0.0)

        if float(lambda_well) > 0.0 and well_cnt > 0:
            pm = pred_m[mask].contiguous()         # (Bw,win,3)
            ym = y_m[mask].contiguous()

            base_well_w = torch.tensor([1.0, 1.0, 3.0], device=pm.device, dtype=pm.dtype).view(1, 1, -1)
            if base_well_w.shape[-1] != C:
                base_well_w = pm.new_ones((1, 1, C))
            well_w = base_well_w * chan_mask
            den_well = (well_w.sum() + 1e-12)

            base_well = ((((pm - ym) ** 2) * well_w).sum(dim=(1, 2)) / (float(pm.shape[1]) * den_well)).mean()

            shape_terms = []
            groups = {}
            for i in range(B):
                if well_mask_list[i]:
                    l_, t_, tau_, _ = metas[i]
                    groups.setdefault((int(l_), int(t_)), []).append((int(tau_), i))

            for (_, _), lst in groups.items():
                if len(lst) < 2:
                    continue
                lst.sort(key=lambda z: z[0])
                idxs = [j for (_, j) in lst]
                p_seq = pred_m[idxs]
                y_seq = y_m[idxs]

                dp = p_seq[1:] - p_seq[:-1]
                dy = y_seq[1:] - y_seq[:-1]
                shape_terms.append(F.smooth_l1_loss(dp, dy, reduction='mean', beta=0.5))

            well_shape = torch.stack(shape_terms).mean() if len(shape_terms) > 0 else pred.new_tensor(0.0)

            loss_well = base_well + 0.3 * well_shape
            well_term = float(lambda_well) * loss_well

        # ---------------------------
        # physics（sequence版，继续吃 pred_m/y_m）
        # ---------------------------
        loss_physics = pred.new_tensor(0.0)
        if physics_loss_fn is not None:
            physics_loss_dict = physics_loss_fn(
                pred=pred_m,
                target=y_m,
                kl_divergence=kl_term,
                seismic_input=X,
                denormalize_fn=denorm_t
            )

            misfit = physics_loss_dict.get("physics_misfit", None)
            prior = physics_loss_dict.get("prior_loss", None)
            rp_raw = physics_loss_dict.get("rock_prior", None)
            ploss = physics_loss_dict.get("physics_loss", None)

            raw_ok = (
                    torch.is_tensor(misfit) and torch.is_tensor(prior) and
                    getattr(misfit, "requires_grad", False) and getattr(prior, "requires_grad", False)
            )

            if not raw_ok:
                loss_physics = ploss if torch.is_tensor(ploss) else pred.new_tensor(0.0)
            else:
                if phys_norm is not None:
                    phys_norm.update(misfit, prior)
                    misfit_n, prior_n, _, _ = phys_norm.normalize(misfit, prior)
                else:
                    misfit_n, prior_n = misfit, prior

                w_prior_rel = float(getattr(physics_loss_fn, "prior_weight", 0.0))
                w_rp_rel = float(getattr(physics_loss_fn, "rp_weight", 0.0))
                w_phys = float(getattr(physics_loss_fn, "physics_weight", 0.0))

                rp_n = pred.new_tensor(0.0)
                if rp_raw is not None:
                    rp_n = torch.clamp(rp_raw, 0.0, 10.0)

                physics_core_term = misfit_n + w_prior_rel * prior_n + w_rp_rel * rp_n

                if bool(auto_w_phys) \
                        and (epoch_idx >= int(phys_auto_start_ep)) \
                        and (epoch_idx >= int(warmup_phys)) \
                        and (step % int(phys_auto_every) == 0) and (step > 0):
                    try:
                        phys_model = getattr(physics_loss_fn, "physics_model", None) if (
                                    physics_loss_fn is not None) else None
                        if phys_model is not None:
                            any_trainable = any(getattr(p, "requires_grad", False) for p in phys_model.parameters())
                            if not any_trainable:
                                print(f"[PHYS-AUTO][SKIP] step={step} physics_model frozen -> skip auto weight.",
                                      flush=True)
                                raise RuntimeError("skip_auto_due_to_frozen_phys_model")

                        g_main = torch.autograd.grad(attr_loss, pred, retain_graph=True, allow_unused=True)[0]
                        g_core = torch.autograd.grad(physics_core_term, pred, retain_graph=True, allow_unused=True)[0]

                        g_main_m = 0.0 if g_main is None else float(g_main.detach().abs().mean().cpu().item())
                        g_core_m = 0.0 if g_core is None else float(g_core.detach().abs().mean().cpu().item())

                        g_core_eps = 5e-5
                        g_main_eps = 5e-5
                        if (g_core_m < g_core_eps) or (g_main_m < g_main_eps):
                            print(
                                f"[PHYS-AUTO][SKIP] step={step} g_main={g_main_m:.3e} g_core={g_core_m:.3e} (<eps) -> skip.",
                                flush=True)
                        else:
                            w_star = g_main_m / (float(phys_target_ratio) * g_core_m + 1e-12)
                            w_star = float(max(float(phys_w_min), min(float(phys_w_max), w_star)))

                            w_new = (1.0 - float(phys_ema)) * float(w_phys) + float(phys_ema) * float(w_star)

                            up_ratio = 1.20
                            down_ratio = 0.83
                            w_new = float(min(float(w_new), float(w_phys) * up_ratio))
                            w_new = float(max(float(w_new), float(w_phys) * down_ratio))

                            w_new = float(max(float(phys_w_min), min(float(phys_w_max), w_new)))

                            try:
                                setattr(physics_loss_fn, "physics_weight", w_new)
                            except Exception:
                                pass

                            print(
                                f"[PHYS-AUTO] step={step} g_main={g_main_m:.3e} g_core={g_core_m:.3e} "
                                f"w_star={w_star:.3e} w_phys: {float(w_phys):.3e} -> {w_new:.3e}",
                                flush=True
                            )
                            w_phys = w_new
                    except Exception as e:
                        if "skip_auto_due_to_frozen_phys_model" not in repr(e):
                            print("[PHYS-AUTO][WARN] failed:", repr(e), flush=True)

                loss_physics = float(w_phys) * physics_core_term

            loss_physics = torch.nan_to_num(loss_physics, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------------------------
        # LAT（sequence版）
        # ---------------------------
        lat_raw = pred.new_tensor(0.0)
        lat_term = pred.new_tensor(0.0)
        lat_pairs = 0

        if (float(lambda_lat) > 0.0) and (int(epoch_idx) >= int(warm_lat_ep)):
            edge_w_line = pred.new_ones((B, 1, 1))
            try:
                if X.dim() == 4:
                    win = int(X.shape[-1])
                    xc = X[..., win // 2]
                    gl = xc[:, :, 1:] - xc[:, :, :-1]
                    gl_mag = gl.abs().mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
                    edge_alpha = 8.0
                    edge_w_line = torch.exp(-edge_alpha * gl_mag).clamp(0.15, 1.0)
            except Exception:
                edge_w_line = pred.new_ones((B, 1, 1))

            groups = {}
            for i, m in enumerate(metas):
                if not (isinstance(m, (list, tuple)) and len(m) >= 3):
                    continue
                l_, t_, tau_ = int(m[0]), int(m[1]), int(m[2])
                groups.setdefault((t_, tau_), []).append((l_, i))

            lat_terms = []
            max_gap = int(lat_max_gap) if lat_max_gap is not None else 8
            if max_gap < 1:
                max_gap = 1

            for _, lst in groups.items():
                if len(lst) < 3:
                    continue
                lst.sort(key=lambda x: x[0])

                seg = [lst[0]]
                for (l_i, idx_i) in lst[1:]:
                    l_prev = seg[-1][0]
                    if abs(int(l_i) - int(l_prev)) <= max_gap:
                        seg.append((l_i, idx_i))
                    else:
                        if len(seg) >= 3:
                            idxs = [ii for (_, ii) in seg]
                            p = pred_m[idxs]
                            d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]
                            w_edge = edge_w_line[idxs][1:-1]
                            eps = 1e-3
                            lat_terms.append((w_edge * torch.sqrt(d2 * d2 + eps)).mean())
                            lat_pairs += int(d2.shape[0])
                        seg = [(l_i, idx_i)]

                if len(seg) >= 3:
                    idxs = [ii for (_, ii) in seg]
                    p = pred_m[idxs]
                    d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]
                    w_edge = edge_w_line[idxs][1:-1]
                    eps = 1e-3
                    lat_terms.append((w_edge * torch.sqrt(d2 * d2 + eps)).mean())
                    lat_pairs += int(d2.shape[0])

            if len(lat_terms) > 0:
                lat_raw = torch.stack(lat_terms).mean()
                lat_term = float(lambda_lat) * lat_raw
                lat_term = torch.nan_to_num(lat_term, nan=0.0, posinf=0.0, neginf=0.0)

        # ==========================================================
        # total loss (RAW for backward)
        # ==========================================================
        loss_raw = attr_loss + gate_reg + recon_term + kl_term_w + loss_physics + lat_term + well_term
        loss_raw = torch.nan_to_num(loss_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ==========================================================
        # Scheme A: shift-to-positive ONLY for logging/balancing
        # ==========================================================
        shift_total.update(float(loss_raw.detach().cpu().item()))
        shift_attr.update(float(attr_loss.detach().cpu().item()))
        shift_well.update(float(well_term.detach().cpu().item()))

        loss_for_log = shift_total.apply(loss_raw)
        attr_for_log = shift_attr.apply(attr_loss)
        well_for_log = shift_well.apply(well_term)

        # backward uses RAW
        loss_raw.backward()

        # clip grads
        if (optimizer_phys is not None) and (physics_loss_fn is not None) and hasattr(physics_loss_fn, "physics_model"):
            torch.nn.utils.clip_grad_norm_(physics_loss_fn.physics_model.parameters(), max_norm=1.0)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer_main.step()
        if optimizer_phys is not None:
            optimizer_phys.step()

        # stats（按样本数加权）
        total_samples += B
        total_loss += float(loss_raw.item()) * B
        total_attr += float(attr_loss.item()) * B
        total_recon_raw += float(loss_recon.item()) * B
        total_recon_term += float(recon_term.item()) * B
        total_kl += float(kl_term.item()) * B
        total_kl_w += float(kl_term_w.item()) * B
        total_phys += float(loss_physics.item()) * B
        total_lat += float(lat_term.item()) * B
        total_mse_for_log += float(mse_for_log.item()) * B
        total_lat_pairs += int(lat_pairs)

        total_well_loss += float(loss_well.item()) * B
        total_well_term += float(well_term.item()) * B

        # ✅ NEW: gate regularization stats
        total_gate_reg += float(gate_reg.item()) * B

        total_loss_for_log += float(loss_for_log.item()) * B
        total_attr_for_log += float(attr_for_log.item()) * B
        total_well_term_for_log += float(well_for_log.item()) * B

        den = max(1, total_samples)

        progress_bar.set_postfix({
            "loss": f"{(total_loss_for_log / den):.4f}",
            "attr": f"{(total_attr_for_log / den):.4f}",
            "gate": f"{(total_gate_reg / den):.4f}",  # ✅ NEW
            "recon": f"{(total_recon_term / den):.4f}",
            "β·kl": f"{(total_kl_w / den):.4f}",
            "phys": f"{(total_phys / den):.4f}",
            "lat": f"{(total_lat / den):.4f}",
            "well": f"{(total_well_term_for_log / den):.4f}",
            "rhob_bad": int(rhob_bad),
        })

    tau_now = None
    # multivariate Student-t 版不再用 temp_scale/logvar
    # 这里保留返回字段兼容旧日志
    try:
        tau_now = None
    except Exception:
        pass

    denom = max(1, int(total_samples))
    ret = {
        "loss": total_loss / denom,
        "attr_loss": total_attr / denom,
        "well_term": total_well_term / denom,

        "loss_for_log": total_loss_for_log / denom,
        "attr_loss_for_log": total_attr_for_log / denom,
        "well_term_for_log": total_well_term_for_log / denom,

        "mse_for_log": total_mse_for_log / denom,
        "mse": total_mse_for_log / denom,


        "recon_loss_raw": total_recon_raw / denom,
        "recon_term": total_recon_term / denom,

        "kl": total_kl / denom,
        "kl_w": total_kl_w / denom,

        "physics_loss": total_phys / denom,

        "well_loss": total_well_loss / denom,
        "lambda_well": float(lambda_well),

        "lat": total_lat / denom,
        "lat_pairs": int(total_lat_pairs),

        # ✅ NEW
        "gate_reg": total_gate_reg / denom,

        "_tau": tau_now,
        "_fs_eff": float(fs_eff),
        "_dt_eff": float(dt_eff),
        "_total_samples": int(total_samples),
        "_rhob_bad": bool(rhob_bad),
    }

    ret.setdefault("report_loss", ret["loss_for_log"])
    ret.setdefault("extra_loss", 0.0)
    return ret



@torch.no_grad()
def eval_epoch_vaim(
    model, loader, device,
    physics_loss_fn=None,
    denormalize_fn=None,
    beta_kl: float = 0.0,
    temp_scale=None,   # 保留接口兼容，但联合t版不再使用
    lambda_recon: float = 0.02,
    phys_norm=None,
    y_mean_t=None, y_std_t=None,
    eval_Tmc: int = 8,
    lambda_lat: float = 0.05,
    lambda_well: float = 0.0,
    warm_lat_ep: int = 8,
    lat_max_gap: int = 8,
    dt: float = 0.001,
    epoch_idx: int = 1,
    lambda_vaim: float = 0.10,
    rhob_std_thresh: float = 1e-6,
    rhob_check_every: int = 200,
    desc: str = "Validation",
    shift_pack: dict = None,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import r2_score

    # dt_eff / fs_eff
    ds_local = getattr(loader, "dataset", None)
    time_stride = int(getattr(ds_local, "time_stride", 1)) if ds_local is not None else 1
    if time_stride < 1:
        time_stride = 1
    dt_eff = float(dt) * float(time_stride)
    fs_eff = 1.0 / dt_eff

    # shifters
    if shift_pack is None:
        shift_pack = getattr(train_epoch_vaim, "_shift_pack", None)
    shift_total = shift_pack["total"] if shift_pack is not None else None
    shift_attr  = shift_pack["attr"]  if shift_pack is not None else None
    shift_well  = shift_pack["well"]  if shift_pack is not None else None

    def _finite_or_report(t: torch.Tensor, name: str, batch_tag: str):
        bad = ~torch.isfinite(t)
        if bad.any():
            cnt = int(bad.sum().item())
            print(f"[EVAL][WARN] {batch_tag}:{name} has {cnt} non-finite -> sanitized.", flush=True)
            t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        return t

    # denorm
    _warned_denorm = False
    _Y_mean = None
    _Y_std = None
    if (y_mean_t is not None) and (y_std_t is not None):
        _Y_mean = torch.as_tensor(y_mean_t, device=device, dtype=torch.float32).view(1, 1, -1)
        _Y_std  = torch.as_tensor(y_std_t,  device=device, dtype=torch.float32).view(1, 1, -1)

    def denorm_t(z: torch.Tensor) -> torch.Tensor:
        nonlocal _warned_denorm
        if (_Y_mean is not None) and (_Y_std is not None):
            if z.dim() == 2:
                return z * _Y_std.view(1, -1).to(z.dtype) + _Y_mean.view(1, -1).to(z.dtype)
            return z * _Y_std.to(z.dtype) + _Y_mean.to(z.dtype)
        if denormalize_fn is not None:
            out = denormalize_fn(z)
            if torch.is_tensor(out):
                return out
            if not _warned_denorm:
                print("[EVAL][DENORM][WARN] denormalize_fn returned non-torch; recommend y_mean_t/y_std_t.", flush=True)
                _warned_denorm = True
        if not _warned_denorm:
            print("[EVAL][DENORM][WARN] No y_mean_t/y_std_t; denorm is identity.", flush=True)
            _warned_denorm = True
        return z

    # extract center line seismic
    # returns:
    #   X_centerline_seq : (B,win,A)
    #   X_center         : (B,A)
    def _extract_centerline_and_center(X: torch.Tensor, A_hint: int = None):
        def _slice_channels(x: torch.Tensor) -> torch.Tensor:
            if A_hint is None:
                return x
            A = int(A_hint)
            if A <= 0:
                return x
            if x.dim() >= 2 and x.size(1) > A:
                return x[:, :A, ...].contiguous()
            return x

        B = int(X.shape[0])

        if X.dim() == 2:
            if A_hint is None or int(A_hint) <= 0:
                raise ValueError("X is 2D [B,A*win] but A_hint not provided.")
            A_hint = int(A_hint)
            in_dim = int(X.shape[1])
            if in_dim % A_hint != 0:
                raise ValueError(f"X.shape[1]={in_dim} not divisible by A_hint={A_hint}")
            win = in_dim // A_hint
            X_centerline = X.reshape(B, A_hint, win).contiguous()          # (B,A,win)
            X_center = X_centerline[:, :, win // 2].contiguous()           # (B,A)
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()   # (B,win,A)
            return X_centerline_seq, X_center

        if X.dim() == 3:
            X_centerline = _slice_channels(X.contiguous())                  # (B,A,win)
            X_center = X_centerline[:, :, X_centerline.shape[-1] // 2].contiguous()
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()   # (B,win,A)
            return X_centerline_seq, X_center

        if X.dim() == 4:
            lc = int(X.shape[2])
            X_centerline = X[:, :, lc // 2, :].contiguous()                # (B,C,win)
            X_centerline = _slice_channels(X_centerline)
            X_center = X_centerline[:, :, X_centerline.shape[-1] // 2].contiguous()
            X_centerline_seq = X_centerline.transpose(1, 2).contiguous()   # (B,win,A)
            return X_centerline_seq, X_center

        raise ValueError(f"Unexpected X.dim={X.dim()} shape={tuple(X.shape)}")

    # LAT eval
    def _lat_loss_eval(pred: torch.Tensor, X: torch.Tensor, metas,
                       lambda_lat: float, warm_lat_ep: int, epoch_idx: int,
                       lat_max_gap: int = 8):
        if (lambda_lat <= 0.0) or (int(epoch_idx) < int(warm_lat_ep)):
            return pred.new_tensor(0.0), 0

        B = int(pred.shape[0])
        C = int(pred.shape[-1])

        edge_w_line = pred.new_ones((B, 1, 1))
        try:
            if torch.is_tensor(X) and X.dim() == 4:
                win = int(X.shape[-1])
                xc = X[..., win // 2]
                gl = xc[:, :, 1:] - xc[:, :, :-1]
                gl_mag = gl.abs().mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
                edge_alpha = 8.0
                edge_w_line = torch.exp(-edge_alpha * gl_mag).clamp(0.15, 1.0)
        except Exception:
            edge_w_line = pred.new_ones((B, 1, 1))

        groups = {}
        for i, m in enumerate(metas):
            if not (isinstance(m, (list, tuple)) and len(m) >= 3):
                continue
            l_, t_, tau_ = int(m[0]), int(m[1]), int(m[2])
            groups.setdefault((t_, tau_), []).append((l_, i))

        lat_terms = []
        lat_pairs = 0
        max_gap = int(lat_max_gap) if lat_max_gap is not None else 8
        if max_gap < 1:
            max_gap = 1

        for _, lst in groups.items():
            if len(lst) < 3:
                continue
            lst.sort(key=lambda x: x[0])

            seg = [lst[0]]
            for (l_i, idx_i) in lst[1:]:
                l_prev = seg[-1][0]
                if abs(int(l_i) - int(l_prev)) <= max_gap:
                    seg.append((l_i, idx_i))
                else:
                    if len(seg) >= 3:
                        idxs = [ii for (_, ii) in seg]
                        p = pred[idxs]                                 # (K,win,3)
                        d2 = (p[2:] - 2.0 * p[1:-1] + p[:-2])          # (K-2,win,3)
                        w_edge = edge_w_line[idxs][1:-1]               # (K-2,1,1)
                        eps = 1e-3
                        lat_terms.append((w_edge * torch.sqrt(d2 * d2 + eps)).mean())
                        lat_pairs += int(d2.shape[0])
                    seg = [(l_i, idx_i)]

            if len(seg) >= 3:
                idxs = [ii for (_, ii) in seg]
                p = pred[idxs]
                d2 = (p[2:] - 2.0 * p[1:-1] + p[:-2])
                w_edge = edge_w_line[idxs][1:-1]
                eps = 1e-3
                lat_terms.append((w_edge * torch.sqrt(d2 * d2 + eps)).mean())
                lat_pairs += int(d2.shape[0])

        if len(lat_terms) == 0:
            return pred.new_tensor(0.0), 0

        lat_raw = torch.stack(lat_terms).mean()
        lat_term = float(lambda_lat) * lat_raw
        lat_term = torch.nan_to_num(lat_term, nan=0.0, posinf=0.0, neginf=0.0)
        return lat_term, int(lat_pairs)

    model.eval()

    total_samples = 0
    rhob_bad = False
    _warned_rhob_bad = False

    # RAW sums
    sum_loss = 0.0
    sum_attr = 0.0
    sum_mse_for_log = 0.0
    sum_recon_raw = 0.0
    sum_recon_term = 0.0
    sum_kl = 0.0
    sum_kl_w = 0.0
    sum_phys = 0.0
    sum_lat = 0.0
    sum_lat_pairs = 0
    sum_well_loss = 0.0
    sum_well_term = 0.0
    sum_gate_reg = 0.0   # ✅ NEW

    # FOR_LOG sums
    sum_loss_for_log = 0.0
    sum_attr_for_log = 0.0
    sum_well_term_for_log = 0.0

    all_preds = []
    all_targets = []
    all_metas = []

    chan_weight = torch.tensor([1.0, 1.0, 12.0], device=device).view(1, 1, -1)
    progress_bar = tqdm(loader, desc=desc)

    for it, batch in enumerate(progress_bar):
        if len(batch) == 4:
            X, y, y_bl, metas = batch
        else:
            X, y, metas = batch
            y_bl = None

        nonblock = (device.type == "cuda")
        X = X.to(device, non_blocking=nonblock)
        y = y.to(device, non_blocking=nonblock)                  # (B,win,3)
        y_bl = y_bl.to(device, non_blocking=nonblock) if y_bl is not None else None

        if y_bl is not None:
            y_bl = torch.nan_to_num(y_bl, nan=0.0, posinf=0.0, neginf=0.0)
            y_bl = torch.clamp(y_bl, min=-8.0, max=8.0)

        B = int(X.shape[0])
        total_samples += B

        mu_sum = None

        loss_sum = attr_sum = mse_log_sum = vaim_sum = 0.0
        recon_raw_sum = recon_term_sum = kl_sum = kl_w_sum = 0.0
        phys_sum = lat_sum = 0.0
        lat_pairs_sum = 0
        well_loss_sum = well_term_sum = 0.0
        gate_reg_sum = 0.0   # ✅ NEW

        for k in range(int(eval_Tmc)):
            try:
                out = model(X, return_kl=True)
            except TypeError:
                out = model(X)

            if not isinstance(out, (tuple, list)):
                raise RuntimeError(f"[EVAL] Expected tuple/list model output, got type={type(out)}")

            extra = None
            if len(out) >= 5:
                mu_attr, chol_params, d_rec, kl, extra = out[:5]
            elif len(out) >= 4:
                mu_attr, chol_params, d_rec, kl = out[:4]
            else:
                raise RuntimeError(
                    "[EVAL] Expected model output like "
                    "(mu_attr, chol_params, d_rec, kl[, extra]), "
                    f"but got len={len(out)}"
                )

            batch_tag = f"it{it}/mc{k}"
            mu_attr = _finite_or_report(mu_attr, "pred", batch_tag)
            y_safe  = _finite_or_report(y, "target", batch_tag)
            chol_params = _finite_or_report(chol_params, "chol_params", batch_tag)

            L_attr = build_scale_tril_3x3(chol_params)

            C = int(mu_attr.shape[-1])
            W = int(mu_attr.shape[1])

            # RHOB degenerate check
            if (not rhob_bad) and (k == 0) and ((it < 3) or ((it % int(rhob_check_every) == 0) and (it > 0))):
                try:
                    y_phys = denorm_t(y_safe)
                    if torch.is_tensor(y_phys) and (y_phys.dim() == 3) and (y_phys.size(-1) >= 3):
                        rh_std = float(y_phys[..., 2].detach().float().std().cpu().item())
                        if rh_std < float(rhob_std_thresh):
                            rhob_bad = True
                            if not _warned_rhob_bad:
                                print(
                                    f"[EVAL][RHOB][AUTO] degenerate RHOB: std={rh_std:.3e} < {float(rhob_std_thresh):.3e}. "
                                    f"Mask RHOB in attr/vaim/well/lat/physics.",
                                    flush=True
                                )
                                _warned_rhob_bad = True
                except Exception:
                    pass

            chan_mask = mu_attr.new_ones((1, 1, C))
            if (C >= 3) and rhob_bad:
                chan_mask[..., 2] = 0.0

            pred_m = mu_attr * chan_mask
            y_m    = y_safe * chan_mask
            y_bl_m = (y_bl * chan_mask) if (y_bl is not None and torch.is_tensor(y_bl) and y_bl.shape == y_safe.shape) else y_bl

            w = chan_weight
            if w.shape[-1] != C:
                w = torch.ones((1, 1, C), device=device, dtype=mu_attr.dtype)
            w = w * chan_mask
            den_w = (w.sum() + 1e-12)

            # mse_for_log
            err = (pred_m - y_m)
            mse_for_log = (((err ** 2) * w).sum(dim=(1, 2)) / (float(W) * den_w)).mean()

            # ---------------------------
            # attr_loss: multivariate Student-t NLL
            # ---------------------------
            df_use = float(getattr(model, "student_df", 4.0))
            attr_loss = multivariate_student_t_nll(
                y=y_m,
                mu=pred_m,
                scale_tril=L_attr,
                nu=df_use,
                reduction="mean"
            )

            # ---------------------------
            # ✅ gate entropy regularization (eval logging only)
            # ---------------------------
            gate_reg = mu_attr.new_tensor(0.0)
            if isinstance(extra, dict) and ("gate" in extra):
                gate = extra["gate"]   # [B,win,K]
                if torch.is_tensor(gate):
                    gate_safe = gate.clamp_min(1e-8)
                    gate_entropy = -(gate_safe * torch.log(gate_safe)).sum(dim=-1).mean()
                    gate_reg = -1e-3 * gate_entropy
                    gate_reg = torch.nan_to_num(gate_reg, nan=0.0, posinf=0.0, neginf=0.0)

            # recon
            loss_recon = mu_attr.new_tensor(0.0)
            recon_term = mu_attr.new_tensor(0.0)
            if d_rec is not None and torch.is_tensor(d_rec):
                A = int(d_rec.shape[-1])
                X_centerline_seq, _ = _extract_centerline_and_center(X, A_hint=A)
                if X_centerline_seq.size(-1) != A:
                    X_centerline_seq = X_centerline_seq[..., :A].contiguous()
                loss_recon = _recon_loss(d_rec, X_centerline_seq, aux=None, loss_type="student")
                recon_term = float(lambda_recon) * loss_recon


            # KL
            kl_term = kl.mean() if (torch.is_tensor(kl) and getattr(kl, "dim", lambda: 0)() > 0) else kl
            kl_term = kl_term if torch.is_tensor(kl_term) else mu_attr.new_tensor(float(kl_term))
            kl_term_w = float(beta_kl) * kl_term

            # physics
            loss_physics = mu_attr.new_tensor(0.0)
            if physics_loss_fn is not None:
                physics_loss_dict = physics_loss_fn(
                    pred=pred_m, target=y_m, kl_divergence=kl_term, seismic_input=X, denormalize_fn=denorm_t
                )
                misfit = physics_loss_dict.get("physics_misfit", None)
                prior  = physics_loss_dict.get("prior_loss", None)
                rp_raw = physics_loss_dict.get("rock_prior", None)
                ploss  = physics_loss_dict.get("physics_loss", None)

                raw_ok = (torch.is_tensor(misfit) and torch.is_tensor(prior))
                if not raw_ok:
                    loss_physics = ploss if torch.is_tensor(ploss) else mu_attr.new_tensor(0.0)
                else:
                    if (phys_norm is not None) and getattr(phys_norm, "inited", False):
                        misfit_n, prior_n, _, _ = phys_norm.normalize(misfit, prior)
                    else:
                        misfit_n, prior_n = misfit, prior

                    w_prior_rel = float(getattr(physics_loss_fn, "prior_weight", 0.0))
                    w_rp_rel    = float(getattr(physics_loss_fn, "rp_weight", 0.0))
                    w_phys      = float(getattr(physics_loss_fn, "physics_weight", 0.0))

                    rp_n = mu_attr.new_tensor(0.0)
                    if rp_raw is not None:
                        rp_n = torch.clamp(rp_raw, 0.0, 10.0)

                    physics_core = misfit_n + w_prior_rel * prior_n + w_rp_rel * rp_n
                    loss_physics = w_phys * physics_core

                loss_physics = torch.nan_to_num(loss_physics, nan=0.0, posinf=0.0, neginf=0.0)

            # well
            well_loss = mu_attr.new_tensor(0.0)
            well_term = mu_attr.new_tensor(0.0)
            if float(lambda_well) > 0.0:
                well_mask_list = [m[3] if (isinstance(m, (list, tuple)) and len(m) >= 4) else False for m in metas]
                mask = torch.tensor(well_mask_list, dtype=torch.bool, device=mu_attr.device)
                well_cnt = int(mask.sum().item())
                if well_cnt > 0:
                    pm = pred_m[mask].contiguous()
                    ym = y_m[mask].contiguous()

                    base_well_w = torch.tensor([1.0, 1.0, 3.0], device=pm.device, dtype=pm.dtype).view(1, 1, -1)
                    if base_well_w.shape[-1] != C:
                        base_well_w = pm.new_ones((1, 1, C))
                    well_w = base_well_w * chan_mask
                    den_well = (well_w.sum() + 1e-12)

                    base_well = ((((pm - ym) ** 2) * well_w).sum(dim=(1, 2)) / (float(pm.shape[1]) * den_well)).mean()

                    shape_terms = []
                    groups = {}
                    for i in range(B):
                        if well_mask_list[i]:
                            l_, t_, tau_, _ = metas[i]
                            groups.setdefault((int(l_), int(t_)), []).append((int(tau_), i))
                    for (_, _), lst in groups.items():
                        if len(lst) < 2:
                            continue
                        lst.sort(key=lambda z: z[0])
                        idxs = [j for (_, j) in lst]
                        p_seq = pred_m[idxs]
                        y_seq = y_m[idxs]
                        dp = p_seq[1:] - p_seq[:-1]
                        dy = y_seq[1:] - y_seq[:-1]
                        shape_terms.append(F.smooth_l1_loss(dp, dy, reduction="mean", beta=0.5))
                    well_shape = torch.stack(shape_terms).mean() if len(shape_terms) > 0 else mu_attr.new_tensor(0.0)

                    well_loss = base_well + 0.3 * well_shape
                    well_term = float(lambda_well) * well_loss

            # lat
            lat_term, lat_pairs = _lat_loss_eval(
                pred=pred_m, X=X, metas=metas,
                lambda_lat=float(lambda_lat),
                warm_lat_ep=int(warm_lat_ep),
                epoch_idx=int(epoch_idx),
                lat_max_gap=int(lat_max_gap)
            )

            # total RAW
            loss_raw = attr_loss + gate_reg +  recon_term + kl_term_w + loss_physics + lat_term + well_term
            loss_raw = torch.nan_to_num(loss_raw, nan=0.0, posinf=0.0, neginf=0.0)

            if shift_total is not None:
                loss_for_log = shift_total.apply(loss_raw)
                attr_for_log = shift_attr.apply(attr_loss) if shift_attr is not None else loss_for_log
                well_for_log = shift_well.apply(well_term) if shift_well is not None else loss_for_log
            else:
                loss_for_log = torch.clamp(loss_raw, min=1e-3)
                attr_for_log = torch.clamp(attr_loss, min=1e-3)
                well_for_log = torch.clamp(well_term, min=0.0)

            loss_sum       += float(loss_raw.item())
            attr_sum       += float(attr_loss.item())
            mse_log_sum    += float(mse_for_log.item())
            recon_raw_sum  += float(loss_recon.item())
            recon_term_sum += float(recon_term.item())
            kl_sum         += float(kl_term.item())
            kl_w_sum       += float(kl_term_w.item())
            phys_sum       += float(loss_physics.item())
            lat_sum        += float(lat_term.item())
            lat_pairs_sum  += int(lat_pairs)
            well_loss_sum  += float(well_loss.item())
            well_term_sum  += float(well_term.item())
            gate_reg_sum   += float(gate_reg.item())   # ✅ NEW

            mu_sum = pred_m if (mu_sum is None) else (mu_sum + pred_m)

        invT = 1.0 / float(eval_Tmc)
        mu_mean = mu_sum * invT

        loss_mc       = loss_sum * invT
        attr_mc       = attr_sum * invT
        mse_log_mc    = mse_log_sum * invT
        recon_raw_mc  = recon_raw_sum * invT
        recon_term_mc = recon_term_sum * invT
        kl_mc         = kl_sum * invT
        kl_w_mc       = kl_w_sum * invT
        phys_mc       = phys_sum * invT
        lat_mc        = lat_sum * invT
        lat_pairs_mc  = int(round(lat_pairs_sum * invT))
        well_loss_mc  = well_loss_sum * invT
        well_term_mc  = well_term_sum * invT
        gate_reg_mc   = gate_reg_sum * invT   # ✅ NEW

        loss_raw_t = torch.as_tensor(loss_mc, device=device, dtype=torch.float32)
        attr_raw_t = torch.as_tensor(attr_mc, device=device, dtype=torch.float32)
        well_raw_t = torch.as_tensor(well_term_mc, device=device, dtype=torch.float32)

        if shift_total is not None:
            loss_for_log_mc = float(shift_total.apply(loss_raw_t).item())
            attr_for_log_mc = float(shift_attr.apply(attr_raw_t).item()) if shift_attr is not None else loss_for_log_mc
            well_for_log_mc = float(shift_well.apply(well_raw_t).item()) if shift_well is not None else float(max(0.0, well_term_mc))
        else:
            loss_for_log_mc = float(max(1e-3, loss_mc))
            attr_for_log_mc = float(max(1e-3, attr_mc))
            well_for_log_mc = float(max(0.0, well_term_mc))

        sum_loss        += loss_mc * B
        sum_attr        += attr_mc * B
        sum_mse_for_log += mse_log_mc * B
        sum_recon_raw   += recon_raw_mc * B
        sum_recon_term  += recon_term_mc * B
        sum_kl          += kl_mc * B
        sum_kl_w        += kl_w_mc * B
        sum_phys        += phys_mc * B
        sum_lat         += lat_mc * B
        sum_lat_pairs   += lat_pairs_mc
        sum_well_loss   += well_loss_mc * B
        sum_well_term   += well_term_mc * B
        sum_gate_reg    += gate_reg_mc * B   # ✅ NEW

        sum_loss_for_log      += loss_for_log_mc * B
        sum_attr_for_log      += attr_for_log_mc * B
        sum_well_term_for_log += well_for_log_mc * B

        all_preds.append(mu_mean.detach().cpu())     # (B,win,3)
        all_targets.append(y.detach().cpu())         # (B,win,3)
        all_metas.extend(list(metas))

        progress_bar.set_postfix({
            "val_loss": f"{loss_for_log_mc:.4f}",
            "attr": f"{attr_for_log_mc:.4f}",
            "gate": f"{gate_reg_mc:.4f}",   # ✅ NEW
            "recon": f"{recon_term_mc:.4f}",
            "β·kl": f"{kl_w_mc:.4f}",
            "phys": f"{phys_mc:.4f}",
            "lat": f"{lat_mc:.4f}",
            "well": f"{well_for_log_mc:.4f}",
            "rhob_bad": bool(rhob_bad),
        })

    # R2（flatten sequence）
    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    P = all_preds_t.numpy()
    T = all_targets_t.numpy()

    P2 = P.reshape(-1, P.shape[-1])
    T2 = T.reshape(-1, T.shape[-1])

    mask_np = np.isfinite(P2).all(axis=1) & np.isfinite(T2).all(axis=1)
    kept = int(mask_np.sum())
    if kept >= 8:
        if (P2.shape[1] >= 3) and rhob_bad:
            r2 = float(r2_score(T2[mask_np, :2], P2[mask_np, :2], multioutput="uniform_average"))
        else:
            r2 = float(r2_score(T2[mask_np], P2[mask_np], multioutput="uniform_average"))
    else:
        r2 = float("nan")

    tau_now = None

    denom = max(1, int(total_samples))

    ret = {
        "loss": sum_loss / denom,
        "attr_loss": sum_attr / denom,
        "mse_for_log": sum_mse_for_log / denom,
        "mse": sum_mse_for_log / denom,
        "recon_loss_raw": sum_recon_raw / denom,
        "recon_term": sum_recon_term / denom,
        "kl": sum_kl / denom,
        "kl_w": sum_kl_w / denom,
        "physics_loss": sum_phys / denom,
        "well_loss": sum_well_loss / denom,
        "well_term": sum_well_term / denom,
        "lambda_well": float(lambda_well),
        "lat": sum_lat / denom,
        "lat_pairs": int(sum_lat_pairs),
        "r2": float(r2),

        "loss_for_log": sum_loss_for_log / denom,
        "attr_loss_for_log": sum_attr_for_log / denom,
        "well_term_for_log": sum_well_term_for_log / denom,

        # ✅ NEW
        "gate_reg": sum_gate_reg / denom,

        "predictions": all_preds_t,
        "targets": all_targets_t,
        "metas": all_metas,

        "_tau": tau_now,
        "_eval_Tmc": int(eval_Tmc),
        "_dt_eff": float(dt_eff),
        "_fs_eff": float(fs_eff),
        "_rhob_bad": bool(rhob_bad),
    }

    ret.setdefault("report_loss", ret["loss_for_log"])
    ret.setdefault("extra_loss", 0.0)
    return ret






# ====================== Plots ======================
def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return p

def plot_training_curves(train_history, val_history, out_dir, beta_kl=1.0):
    import os, math
    from pathlib import Path
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_history) + 1)

    def _get(x, keys, default=float("nan")):
        for k in keys:
            if k in x and x[k] is not None:
                try:
                    v = float(x[k])
                    return v
                except Exception:
                    pass
        return float(default)

    def _safe(v):
        try:
            v = float(v)
        except Exception:
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    def safe_div(a, b, eps=1e-12):
        a = _safe(a); b = _safe(b)
        if (not math.isfinite(a)) or (not math.isfinite(b)):
            return float("nan")
        if abs(b) < eps:
            return float("nan")
        return a / b

    # -----------------------
    # 1) Raw metrics (train/val same coordinate)
    # -----------------------
    tr_loss_raw = [_get(x, ["loss"]) for x in train_history]
    va_loss_raw = [_get(x, ["loss"]) for x in val_history]

    tr_mse = [_get(x, ["mse", "mse_for_log", "recon", "recon_mse"]) for x in train_history]
    va_mse = [_get(x, ["mse", "mse_for_log", "recon", "recon_mse"]) for x in val_history]

    tr_phys = [_get(x, ["physics_loss"]) for x in train_history]
    va_phys = [_get(x, ["physics_loss"]) for x in val_history]

    tr_kl = [_get(x, ["kl"]) for x in train_history]

    # well term (the one added into total loss)
    tr_well_raw = [_get(x, ["well_term", "well_loss"]) for x in train_history]
    va_well_raw = [_get(x, ["well_term", "well_loss"]) for x in val_history]

    va_r2 = [_get(x, ["r2"], default=float("nan")) for x in val_history]

    # -----------------------
    # 2) Display metrics (optional, for visualization only)
    #   - prefer *_for_log if exists, else fallback to raw
    # -----------------------
    tr_loss_disp = [_get(x, ["loss_for_log", "report_loss", "loss"]) for x in train_history]
    va_loss_disp = [_get(x, ["loss_for_log", "report_loss", "loss"]) for x in val_history]

    tr_well_disp = [_get(x, ["well_term_for_log", "well_term", "well_loss"]) for x in train_history]
    va_well_disp = [_get(x, ["well_term_for_log", "well_term", "well_loss"]) for x in val_history]

    # -----------------------
    # 3) Ratios (use RAW denom to avoid fake spikes)
    # -----------------------
    denom = tr_loss_raw  # IMPORTANT: raw

    # recon term: prefer recon_term; fallback to mse
    tr_recon_term = [
        _get(x, ["recon_term", "recon_loss_raw", "recon", "mse"], default=float("nan"))
        for x in train_history
    ]

    # KL weight: prefer kl_w if already weighted, else beta_kl * kl
    tr_kl_w = []
    for x in train_history:
        v = _get(x, ["kl_w"], default=float("nan"))
        if math.isfinite(v):
            tr_kl_w.append(v)
        else:
            tr_kl_w.append(beta_kl * _get(x, ["kl"], default=float("nan")))

    recon_r = [safe_div(tr_recon_term[i], denom[i]) for i in range(len(train_history))]
    phys_r  = [safe_div(tr_phys[i],       denom[i]) for i in range(len(train_history))]
    kl_r    = [safe_div(tr_kl_w[i],       denom[i]) for i in range(len(train_history))]
    well_r  = [safe_div(tr_well_raw[i],   denom[i]) for i in range(len(train_history))]

    # -----------------------
    # 4) Plot
    # -----------------------
    plt.figure(figsize=(18, 12))

    # (1) Total Loss
    plt.subplot(2, 4, 1)
    plt.plot(epochs, tr_loss_disp, "b-", label="Train (display)")
    plt.plot(epochs, va_loss_disp, "r-", label="Val (display)")
    plt.title("Total Loss (display)")
    plt.grid(alpha=0.3); plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("Loss")

    # (2) MSE
    plt.subplot(2, 4, 2)
    plt.plot(epochs, tr_mse, "b-", label="Train")
    plt.plot(epochs, va_mse, "r-", label="Val")
    plt.title("MSE (raw)")
    plt.grid(alpha=0.3); plt.legend()
    plt.xlabel("Epoch")

    # (3) Physics Loss
    plt.subplot(2, 4, 3)
    plt.plot(epochs, tr_phys, color="orange", label="Train")
    plt.plot(epochs, va_phys, color="red", label="Val")
    plt.title("Physics Loss (raw)")
    plt.grid(alpha=0.3); plt.legend()
    plt.xlabel("Epoch")

    # (4) KL raw
    plt.subplot(2, 4, 4)
    plt.plot(epochs, tr_kl, "g-")
    plt.title("KL (raw)")
    plt.grid(alpha=0.3)
    plt.xlabel("Epoch")

    # (5) Well Term
    plt.subplot(2, 4, 5)
    plt.plot(epochs, tr_well_disp, color="purple", label="Train (display)")
    plt.plot(epochs, va_well_disp, color="magenta", label="Val (display)")
    plt.title("Well Term (display)")
    plt.grid(alpha=0.3); plt.legend()
    plt.xlabel("Epoch")

    # (6) Val R²
    plt.subplot(2, 4, 6)
    plt.plot(epochs, va_r2, color="orange")
    plt.title("Val R²")
    plt.grid(alpha=0.3)
    plt.xlabel("Epoch")

    # (7) Ratios (RAW denom)
    plt.subplot(2, 4, 7)
    plt.plot(epochs, recon_r, "b-", label="ReconTerm/Loss(raw)")
    plt.plot(epochs, phys_r,  color="orange", label="Physics/Loss(raw)")
    plt.plot(epochs, kl_r,    "g-", label="β·KL/Loss(raw)")
    plt.plot(epochs, well_r,  color="purple", label="WellTerm/Loss(raw)")
    plt.title("Loss Ratios (RAW denom)")
    plt.grid(alpha=0.3); plt.legend()
    plt.xlabel("Epoch")

    # (8) LR + tau
    plt.subplot(2, 4, 8)
    lr_main = [x.get("_lr_main", x.get("_lr", None)) for x in train_history]
    lr_phys = [x.get("_lr_phys", None) for x in train_history]
    has_main = any(v is not None for v in lr_main)
    has_phys = any(v is not None for v in lr_phys)

    if has_main:
        plt.plot(epochs, [v if v is not None else float("nan") for v in lr_main], "-", label="Main LR")
    if has_phys:
        plt.plot(epochs, [v if v is not None else float("nan") for v in lr_phys], "-", label="Physics LR")

    plt.title("Learning Rate")
    plt.grid(alpha=0.3)
    plt.xlabel("Epoch")
    if has_main or has_phys:
        plt.legend()

    ax2 = plt.gca().twinx()
    tau_vp   = [x.get("tau_vp", float("nan")) for x in train_history]
    tau_vs   = [x.get("tau_vs", float("nan")) for x in train_history]
    tau_rhob = [x.get("tau_rhob", float("nan")) for x in train_history]
    ax2.plot(epochs, tau_vp,   ":", label="τ_VP")
    ax2.plot(epochs, tau_vs,   ":", label="τ_VS")
    ax2.plot(epochs, tau_rhob, ":", label="τ_RHOB")
    ax2.set_ylabel("Temp τ")
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves_with_physics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE]", out_path)




def _build_windows_25d(ds, l: int, t: int, tau_list, line_ctx: int = None):
    """
    return:
      Xb: torch.FloatTensor [T_eff, A_sel, line_ctx, win] (标准化后)
      tau_axis: np.int64 [T_eff]  (就是 tau_list)
      ang_idx: np.int64 [A_sel]
      ls_idx:  np.int64 [line_ctx]
    """
    import numpy as np
    import torch

    L, Tn, A_all, N_all = ds.stack.shape
    win = int(ds.win)
    h = win // 2

    # -------- line_ctx ----------
    if line_ctx is None:
        line_ctx = int(getattr(ds, "line_ctx", 1))
    line_ctx = max(1, line_ctx)
    r = line_ctx // 2

    # 预先构造 ls（对所有 tau 一样）
    l0 = max(0, l - r)
    l1 = min(L - 1, l + r)
    ls = list(range(l0, l1 + 1))
    while len(ls) < line_ctx:
        if ls[0] > 0:
            ls = [ls[0] - 1] + ls
        elif ls[-1] < L - 1:
            ls = ls + [ls[-1] + 1]
        else:
            ls = [ls[0]] * line_ctx
            break
    if len(ls) > line_ctx:
        mid = len(ls) // 2
        half = line_ctx // 2
        ls = ls[mid - half: mid - half + line_ctx]
    ls_idx = np.asarray(ls, dtype=np.int64)

    # -------- angles_idx -> array ----------
    ang = getattr(ds, "angles_idx", None)
    if ang is None:
        ang_idx = np.arange(A_all, dtype=np.int64)
    elif isinstance(ang, slice):
        ang_idx = np.arange(A_all, dtype=np.int64)[ang]
    else:
        ang_idx = np.asarray(ang, dtype=np.int64).reshape(-1)
    A_sel = int(ang_idx.size)

    # -------- 标准化参数对齐角度子集 ----------
    X_mean_sel = np.asarray(ds.X_mean, dtype=np.float32).reshape(-1)[ang_idx]  # (A_sel,)
    X_std_sel  = np.asarray(ds.X_std,  dtype=np.float32).reshape(-1)[ang_idx]  # (A_sel,)
    Xm = X_mean_sel.reshape(A_sel, 1, 1)  # (A_sel,1,1)
    Xs = X_std_sel.reshape(A_sel, 1, 1)   # (A_sel,1,1)

    X_list = []
    for tau in tau_list:
        s0 = int(tau - h)
        s1 = int(tau + h + 1)

        # ✅ FIX: 先只用 ls_idx 这个 advanced index，angles 暂时用 ':'
        tmp = ds.stack[ls_idx, t, :, s0:s1]         # (line_ctx, A_all, win)
        tmp = tmp[:, ang_idx, :]                    # (line_ctx, A_sel, win)

        Xctx = np.transpose(tmp, (1, 0, 2)).astype(np.float32)  # (A_sel, line_ctx, win)
        Xn = (Xctx - Xm) / (Xs + 1e-8)

        X_list.append(torch.from_numpy(Xn).float())

    Xb = torch.stack(X_list, 0)  # (T_eff, A_sel, line_ctx, win)
    return Xb, np.asarray(tau_list, dtype=np.int64), ang_idx, ls_idx


def denorm_np(arr_torch, ds):
    arr = arr_torch.detach().cpu().numpy()
    return arr * ds.Y_std[None,:] + ds.Y_mean[None,:]

def r2_1d(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < eps:
        return np.nan
    return 1.0 - ss_res / ss_tot

def plot_parity_and_residuals(y_true, y_pred, out_dir, tag="val", max_points=200_000):
    """
    修复版：自动过滤 NaN/Inf，必要时随机抽样避免卡死；每通道画散点+y=x+拟合线、
    并输出残差直方图和Q-Q。
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from scipy import stats as _stats
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"
    mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.shape[0] == 0:
        print("[WARN] parity: no finite samples after filtering"); return
    # 抽样
    if yt.shape[0] > max_points:
        idx = np.random.default_rng(123).choice(yt.shape[0], size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]
    labs = ["VP","VS","RHOB"]

    # --- Parity
    plt.figure(figsize=(12,4))
    for i in range(3):
        ax = plt.subplot(1,3,i+1)
        ax.scatter(yt[:,i], yp[:,i], s=4, alpha=.35)
        mn = float(min(yt[:,i].min(), yp[:,i].min())); mx = float(max(yt[:,i].max(), yp[:,i].max()))
        ax.plot([mn,mx],[mn,mx], 'k-', lw=1.5)
        k,b,r,_,_ = _stats.linregress(yt[:,i], yp[:,i])
        ax.plot([mn,mx],[k*mn+b, k*mx+b], 'r--', lw=1.2)
        ax.set_title(f"{labs[i]}  R²={r**2:.3f}")
        ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"parity_{tag}.png"), dpi=250, bbox_inches='tight'); plt.close()

    # --- Residuals + Q-Q
    res = yp - yt
    plt.figure(figsize=(12,6))
    for i in range(3):
        ax = plt.subplot(2,3,i+1); ax.hist(res[:,i], bins=50, alpha=.9); ax.set_title(f"{labs[i]} Residual"); ax.grid(alpha=.3)
        ax2 = plt.subplot(2,3,3+i+1); _stats.probplot(res[:,i], dist="norm", plot=ax2); ax2.set_title(f"{labs[i]} Q-Q"); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"residuals_{tag}.png"), dpi=250, bbox_inches='tight'); plt.close()




def enable_dropout_only(m):
    import torch.nn as nn
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            mod.train()


@torch.no_grad()
def mc_predict_loader(model, loader, device, T: int = 20, temp_scale=None,
                      df: float = 4.0,
                      return_scale: bool = False,
                      dropout_only: bool = True):
    """
    MC 预测（中心点兼容版，适配联合 Student-t + sequence 输出）

    返回：
      mean_n:     (N,3) 标准化空间均值（取 win 中心点）
      std_pred_n: (N,3) 标准化空间预测总 std（epi + aleatoric）
    若 return_scale=True，额外返回：
      scale_n:    (N,3) 标准化空间条件后验 std（取联合协方差对角线开方）

    当前模型输出约定：
      mu:          (B, win, 3)
      chol_params: (B, win, 6)
      d_rec:       (B, win, A)
      kl:          scalar / tensor

    说明：
      - 这里只取中心时间位置 t0 = win//2
      - 联合 Student-t 的 aleatoric 方差：
            Var_t = Sigma * df/(df-2)
        这里对每个通道取协方差对角线
    """
    import torch
    import torch.nn as nn

    def _enable_dropout_only(m):
        for mod in m.modules():
            if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                mod.train()

    was_train = model.training
    if dropout_only:
        model.eval()
        _enable_dropout_only(model)
    else:
        model.train()

    all_mean = []
    all_scale2 = []   # E[diag(Sigma)]
    all_var_epi = []  # Var[mu]

    df_use = float(getattr(model, "student_df", df))
    t_var_factor = df_use / (df_use - 2.0) if df_use > 2.0 else float("inf")

    for batch in loader:
        if len(batch) == 4:
            X, y, y_bl, meta = batch
        elif len(batch) == 3:
            X, y, meta = batch
        else:
            raise ValueError(f"mc_predict_loader: unexpected batch size {len(batch)}")

        X = X.to(device, non_blocking=(device.type == "cuda"))

        mu_samples = []
        scale2_samples = []

        for _ in range(int(T)):
            try:
                out = model(X, return_kl=True)
            except TypeError:
                out = model(X)

            if not (isinstance(out, (tuple, list)) and len(out) >= 4):
                raise RuntimeError(
                    "[mc_predict_loader] Expected model output like (mu, chol_params, d_rec, kl), "
                    f"but got type={type(out)} len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
                )

            pred, chol_params, d_rec, kl = out[:4]   # pred: (B,win,3), chol_params: (B,win,6)

            if pred.dim() != 3 or pred.size(-1) != 3:
                raise RuntimeError(f"[mc_predict_loader] pred must be (B,win,3), got {tuple(pred.shape)}")
            if chol_params.dim() != 3 or chol_params.size(-1) != 6:
                raise RuntimeError(f"[mc_predict_loader] chol_params must be (B,win,6), got {tuple(chol_params.shape)}")

            # ---- 取中心时间点 ----
            t0 = int(pred.shape[1] // 2)

            pred_c = pred[:, t0, :]                 # (B,3)
            chol_c = chol_params[:, t0, :]          # (B,6)

            # build L for center point
            chol_c_seq = chol_c.unsqueeze(1)        # (B,1,6)
            L_c = build_scale_tril_3x3(chol_c_seq)  # (B,1,3,3)
            cov_c = scale_tril_to_cov(L_c)          # (B,1,3,3)
            var_diag_c = torch.diagonal(cov_c, dim1=-2, dim2=-1).squeeze(1)  # (B,3)

            mu_samples.append(pred_c.detach().cpu())
            scale2_samples.append(var_diag_c.detach().cpu())

        mu_stack = torch.stack(mu_samples, 0)          # (T,B,3)
        mean_b   = mu_stack.mean(0)                    # (B,3)
        var_epi  = mu_stack.var(0, unbiased=False)     # (B,3)

        scale2_stack = torch.stack(scale2_samples, 0)  # (T,B,3)
        mean_scale2  = scale2_stack.mean(0)            # (B,3)

        all_mean.append(mean_b)
        all_var_epi.append(var_epi)
        all_scale2.append(mean_scale2)

    if was_train:
        model.train()
    else:
        model.eval()

    mean_n = torch.cat(all_mean, 0)               # (N,3)
    var_epi_n = torch.cat(all_var_epi, 0)         # (N,3)
    mean_scale2_n = torch.cat(all_scale2, 0)      # (N,3)

    # Student-t aleatoric 方差：diag(Sigma) * df/(df-2)
    var_ale_t_n = mean_scale2_n * float(t_var_factor)

    # 总预测方差 = epistemic + aleatoric
    var_pred_n = var_epi_n + var_ale_t_n
    std_pred_n = torch.sqrt(torch.clamp(var_pred_n, min=0.0))

    if return_scale:
        scale_n = torch.sqrt(torch.clamp(mean_scale2_n, min=0.0))  # (N,3)
        return mean_n.numpy(), std_pred_n.numpy(), scale_n.numpy()

    return mean_n.numpy(), std_pred_n.numpy()






import torch

def _predict_mu_sd_compat(model, Xb, ds, device,
                          Tmc=30, temp_scale=None, df: float = 4.0,
                          mc_train_mode: bool = True,
                          mode: str = "pred"):
    """
    统一输出：MU, SD （物理单位）shape = (T_eff, 3)

    - 自动兼容 predict_distribution_t(...) 是否支持 dropout_only / mc_train_mode 等参数
    - 自动兼容输出结构：dict / tuple / list
    """
    import numpy as np
    import torch
    import inspect

    # ---- call predict_distribution_t with compat kwargs ----
    fn = getattr(model, "predict_distribution_t", None)
    if fn is None:
        raise AttributeError("model.predict_distribution_t not found")

    sig = None
    try:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()  # signature 拿不到就走 try/except 兜底

    kwargs = {}
    # 常见参数：你代码里用过哪些就放哪些（存在才传）
    if "Tmc" in params:
        kwargs["Tmc"] = int(Tmc)
    if "temp_scale" in params:
        kwargs["temp_scale"] = temp_scale
    if "df" in params:
        kwargs["df"] = float(df)
    if "mc_train_mode" in params:
        kwargs["mc_train_mode"] = bool(mc_train_mode)
    if "mode" in params:
        kwargs["mode"] = str(mode)

    # ✅ 关键：dropout_only 只有在对方支持时才传
    # （你现在的 predict_distribution_t 不支持，所以不会传，自然不报错）
    if "dropout_only" in params:
        kwargs["dropout_only"] = True

    # ---- manage train/eval for MC dropout ----
    was_training = model.training
    if mc_train_mode:
        model.train()
    else:
        model.eval()

    out = None
    try:
        out = fn(Xb, **kwargs) if len(params) > 0 else fn(Xb)  # params 为空就别乱传
    except TypeError:
        # signature/params 判断失败时的兜底：逐步降级
        try:
            out = fn(Xb, Tmc=int(Tmc), temp_scale=temp_scale, df=float(df))
        except TypeError:
            out = fn(Xb)
    finally:
        model.train(was_training)

    # ---- parse output -> (mu, sd) in normalized or physical ----
    def _to_np(x):
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        return np.asarray(x)

    mu = sd = None

    if isinstance(out, dict):
        for k in ["mu", "mean", "pred_mu", "m"]:
            if k in out:
                mu = out[k]; break
        for k in ["sd", "std", "sigma", "pred_sd", "s"]:
            if k in out:
                sd = out[k]; break

        # 有些会给 var
        if sd is None:
            for k in ["var", "variance", "pred_var"]:
                if k in out:
                    v = out[k]
                    sd = torch.sqrt(v) if torch.is_tensor(v) else np.sqrt(_to_np(v))
                    break

    elif isinstance(out, (tuple, list)):
        # 常见 (mu, sd) / (mu, var) / (mu, sd, extra...)
        if len(out) >= 2:
            mu, sd = out[0], out[1]

            # 如果 sd 看起来像 var（全非负且量级很大），可按需改；这里不强判，保守只处理明显 var 的情况
            # 你也可以删掉下面这段
            try:
                sd_np = _to_np(sd)
                if (sd_np >= 0).all() and np.nanmax(sd_np) > 1e3 and np.nanmax(sd_np) > 10 * np.nanmax(_to_np(mu)**2 + 1e-12):
                    sd = np.sqrt(sd_np)
            except Exception:
                pass
        else:
            raise ValueError(f"predict_distribution_t returned list/tuple of len={len(out)}; cannot parse")

    else:
        raise ValueError(f"Unexpected predict_distribution_t output type: {type(out)}")

    if mu is None or sd is None:
        raise ValueError(f"Cannot parse (mu, sd) from predict_distribution_t output: {type(out)}")

    mu = _to_np(mu)
    sd = _to_np(sd)

    # ---- ensure shape (T,3) ----
    mu = mu.reshape(mu.shape[0], -1)
    sd = sd.reshape(sd.shape[0], -1)
    if mu.shape[1] != 3:
        # 兜底：如果多维，取前3
        mu = mu[:, :3]
        sd = sd[:, :3]

    # ---- denormalize to physical ----
    # 你的 ds.denormalize_y 支持 torch / numpy，这里用 torch 更稳
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    mu_phys = ds.denormalize_y(mu_t).detach().cpu().numpy()
    # SD 的反归一：乘以 Y_std（不加均值）
    Ys = torch.as_tensor(ds.Y_std, dtype=sd_t.dtype, device=sd_t.device).view(1, 3)
    sd_phys = (sd_t * Ys).detach().cpu().numpy()

    return mu_phys.astype(np.float32), sd_phys.astype(np.float32)


@torch.no_grad()
def plot_profile_at(model, ds, device, l, t, out_dir,
                    tag="profile", Tmc=30, temp_scale=None,
                    space="depth",
                    fdom: float = 45.0, dt: float = 0.001, df: float = 4.0):
    import os, numpy as np, torch, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _r2_1d(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        n = min(y_true.size, y_pred.size)
        if n <= 1:
            return np.nan
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < eps:
            return np.nan
        return 1.0 - ss_res / ss_tot

    df_use = float(getattr(model, "student_df", df))

    # ---------------- main ----------------
    L, Tn, A_all, _ = ds.stack.shape
    assert 0 <= l < L and 0 <= t < Tn, "plot_profile_at: 井位越界"
    h = ds.win // 2

    line_ctx = int(getattr(ds, "line_ctx", 1))
    line_ctx = max(1, line_ctx)
    r = line_ctx // 2

    tau_stride = int(getattr(ds, "time_stride", 1))
    tau_stride = max(1, tau_stride)

    # True
    Y_mod = ds.mod[l, t, ds.props_idx, :].astype(np.float32)
    if Y_mod.ndim == 2 and Y_mod.shape[0] == 3:
        Y_mod = Y_mod.T
    N_mod = Y_mod.shape[0]
    if N_mod <= 2 * h + 1:
        print(f"[plot_profile_at] N_mod too small (N={N_mod}, win={ds.win}) at (l={l}, t={t}), skip.")
        return

    tau_list = list(range(h, N_mod - h, tau_stride))
    if len(tau_list) == 0:
        print(f"[plot_profile_at] no window for (l={l}, t={t}), skip.")
        return

    # lines context indices
    l0 = max(0, l - r)
    l1 = min(L - 1, l + r)
    ls = list(range(l0, l1 + 1))
    while len(ls) < line_ctx:
        if ls[0] > 0:
            ls = [ls[0] - 1] + ls
        elif ls[-1] < L - 1:
            ls = ls + [ls[-1] + 1]
        else:
            ls = [ls[0]] * line_ctx
            break
    if len(ls) > line_ctx:
        mid = len(ls) // 2
        half = line_ctx // 2
        ls = ls[mid - half: mid - half + line_ctx]
    ls_idx = np.asarray(ls, dtype=np.int64)

    # angles_idx
    ang = getattr(ds, "angles_idx", None)
    if ang is None:
        ang_idx = np.arange(A_all, dtype=np.int64)
    elif isinstance(ang, slice):
        ang_idx = np.arange(A_all, dtype=np.int64)[ang]
    else:
        ang_idx = np.asarray(ang, dtype=np.int64).reshape(-1)

    # ---------------- normalization stats ----------------
    X_mean = np.asarray(ds.X_mean, dtype=np.float32)
    X_std = np.asarray(ds.X_std, dtype=np.float32)

    use_angle_time = (X_mean.ndim == 2)

    X_list = []
    for tau in tau_list:
        s0, s1 = tau - h, tau + h + 1
        tmp = ds.stack[ls_idx, t, :, s0:s1]                     # [line_ctx, A_all, win]
        tmp = tmp[:, ang_idx, :]                                # [line_ctx, A_sel, win]
        Xctx = np.transpose(tmp, (1, 0, 2)).astype(np.float32)  # [A_sel, line_ctx, win]

        if use_angle_time:
            Xm_sel = X_mean[ang_idx, :].astype(np.float32)[:, None, :]
            Xs_sel = X_std[ang_idx, :].astype(np.float32)[:, None, :]
            Xn = (Xctx - Xm_sel) / (Xs_sel + 1e-8)
        else:
            Xm_sel = X_mean.reshape(-1)[ang_idx].astype(np.float32)[:, None, None]
            Xs_sel = X_std.reshape(-1)[ang_idx].astype(np.float32)[:, None, None]
            Xn = (Xctx - Xm_sel) / (Xs_sel + 1e-8)

        X_list.append(torch.from_numpy(Xn).float())

    Xb = torch.stack(X_list, 0).to(device)

    # ---------------- add_line_diff 与训练保持一致 ----------------
    if bool(getattr(ds, "add_line_diff", False)):
        A = int(len(ang_idx))
        if Xb.dim() == 4 and Xb.size(1) == A:
            hl = Xb.size(2) // 2
            Xc = Xb[:, :, hl, :].unsqueeze(2)
            Xd = Xb - Xc
            Xb = torch.cat([Xb, Xd], dim=1)

    # ---------------- MC prediction ----------------
    was_training = bool(model.training)
    model.train()
    try:
        mu_samples = []
        scale_samples = []

        for _ in range(int(Tmc)):
            out = model(Xb, return_kl=True)
            if not (isinstance(out, (tuple, list)) and len(out) >= 4):
                raise RuntimeError(
                    "[plot_profile_at] Expected model output like (mu_attr, chol_params, d_rec, kl), "
                    f"but got type={type(out)} len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
                )

            if len(out) >= 5:
                mu_attr, chol_params, d_rec, kl, extra = out[:5]
            else:
                mu_attr, chol_params, d_rec, kl = out[:4]

            if mu_attr.dim() != 3 or mu_attr.size(-1) != 3:
                raise RuntimeError(f"[plot_profile_at] mu_attr must be (B,win,3), got {tuple(mu_attr.shape)}")
            if chol_params.dim() != 3 or chol_params.size(-1) != 6:
                raise RuntimeError(f"[plot_profile_at] chol_params must be (B,win,6), got {tuple(chol_params.shape)}")

            t0 = int(mu_attr.shape[1] // 2)

            mu_c = mu_attr[:, t0, :]
            chol_c = chol_params[:, t0, :].unsqueeze(1)
            L_c = build_scale_tril_3x3(chol_c)
            cov_c = scale_tril_to_cov(L_c)
            var_c = torch.diagonal(cov_c, dim1=-2, dim2=-1).squeeze(1)
            sd_c = torch.sqrt(torch.clamp(var_c, min=1e-12))

            mu_samples.append(mu_c.detach().cpu())
            scale_samples.append(sd_c.detach().cpu())

        mu_stack = torch.stack(mu_samples, 0)
        scale_stack = torch.stack(scale_samples, 0)

        mu_mean_n = mu_stack.mean(0)
        var_epi_n = mu_stack.var(0, unbiased=False)

        t_var_factor = (df_use / (df_use - 2.0)) if df_use > 2.0 else 1.0
        mean_var_ale_n = (scale_stack ** 2).mean(0) * t_var_factor
        std_n = torch.sqrt(torch.clamp(var_epi_n + mean_var_ale_n, min=1e-12))

        Y_mean_t = torch.as_tensor(ds.Y_mean, dtype=mu_mean_n.dtype, device=mu_mean_n.device).view(1, 3)
        Y_std_t = torch.as_tensor(ds.Y_std, dtype=mu_mean_n.dtype, device=mu_mean_n.device).view(1, 3)

        MU = (mu_mean_n * Y_std_t + Y_mean_t).cpu().numpy()
        SD = (std_n * Y_std_t).cpu().numpy()

    finally:
        model.train(was_training) if was_training else model.eval()

    # ---------------- align ----------------
    y_true_eff = Y_mod[np.asarray(tau_list, dtype=np.int64), :]
    n = min(len(tau_list), y_true_eff.shape[0], MU.shape[0], SD.shape[0])
    if n <= 1:
        print(f"[plot_profile_at] effective length too small at (l={l}, t={t}), skip.")
        return

    y_axis_mod = np.arange(N_mod)
    y_axis_pred = np.asarray(tau_list[:n], dtype=np.int64)
    y_true_eff = y_true_eff[:n]
    MU = MU[:n]
    SD = SD[:n]

    # 仍然算 R²，但不显示在图上
    r2_list = [_r2_1d(y_true_eff[:, i], MU[:, i]) for i in range(3)]
    r2_mean = np.nanmean(r2_list) if np.isfinite(np.nanmean(r2_list)) else np.nan

    labs = [
        r"$V_p$",
        r"$V_s$",
        r"$\rho$"
    ]

    # ================== 这里开始是你要的新版排版 ==================
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.2, 12.0),   # 更窄，更像你第二张图
        sharey=True
    )

    # 只显示 18~100
    y_min_show, y_max_show = 18, 100

    for i, ax in enumerate(axes):
        ax.plot(Y_mod[:, i], y_axis_mod, color="k", lw=1.0, label="True")
        ax.plot(MU[:, i], y_axis_pred, color="r", lw=1.1, label="Pred μ")
        ax.fill_betweenx(
            y_axis_pred,
            MU[:, i] - SD[:, i],
            MU[:, i] + SD[:, i],
            color="0.8", alpha=0.45, linewidth=0,
            label="±1σ" if i == 0 else None
        )

        ax.grid(alpha=0.25)
        ax.set_ylim(y_max_show, y_min_show)   # 反转 + 限定 18~100
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # 去掉顶部标题
        ax.set_title("")

        # 去掉底部 xlabel
        ax.set_xlabel("")

        # 把 VP / VS / RHOB 放到底部
        ax.text(
            0.5, -0.035, labs[i],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11
        )

    axes[0].set_ylabel("Time (s)" if space == "time" else "Depth index (tau)", fontsize=10)


    # 图例放右上角，和你原图接近
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", frameon=True, fontsize=9)

    # 更像第二张图的紧凑排版
    plt.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.09, wspace=0.10)

    fn = os.path.join(out_dir, f"{tag}_l{l}_t{t}_clean.png")
    plt.savefig(fn, dpi=220, bbox_inches="tight")
    plt.close()
    print("[SAVE]", fn)



# ===== Metrics: R2 / RMSE（按通道+总体统计，另存 CSV） =====
def compute_metrics_and_save(y_true_d, y_pred_d, out_dir, tag="val"):
    """
    y_true_d / y_pred_d: 物理空间 (N,3)
    产物：
      - metrics_{tag}.json
      - metrics_{tag}.csv
      - 控制台摘要

    新增：
      - per-channel: MAE, Corr
      - overall: MAE, Corr
      - top-level: mean_mae, std_mae, mean_corr, std_corr
      - 兼容数组风格导出: mae_d / rmse_d / r2_d / corr_d
    """
    import json, os, csv
    import numpy as np
    from sklearn.metrics import r2_score

    os.makedirs(out_dir, exist_ok=True)
    labs = ["VP", "VS", "RHOB"]

    y_true_d = np.asarray(y_true_d, dtype=np.float64)
    y_pred_d = np.asarray(y_pred_d, dtype=np.float64)

    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _corr(a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.size < 2 or b.size < 2:
            return float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # -------------------------
    # per-channel
    # -------------------------
    r2s, rmses, maes, corrs = [], [], [], []
    for i in range(3):
        yt = y_true_d[:, i]
        yp = y_pred_d[:, i]

        r2s.append(float(r2_score(yt, yp)))
        rmses.append(_rmse(yt, yp))
        maes.append(_mae(yt, yp))
        corrs.append(_corr(yt, yp))

    # -------------------------
    # overall（把三通道拼一起）
    # -------------------------
    yt_all = y_true_d.reshape(-1)
    yp_all = y_pred_d.reshape(-1)

    r2_overall = float(r2_score(yt_all, yp_all))
    rmse_overall = _rmse(yt_all, yp_all)
    mae_overall = _mae(yt_all, yp_all)
    corr_overall = _corr(yt_all, yp_all)

    # -------------------------
    # JSON
    # -------------------------
    js = {
        "per_channel": {
            labs[i]: {
                "R2": float(r2s[i]),
                "RMSE": float(rmses[i]),
                "MAE": float(maes[i]),
                "Corr": float(corrs[i]) if np.isfinite(corrs[i]) else None,
            }
            for i in range(3)
        },
        "overall": {
            "R2": float(r2_overall),
            "RMSE": float(rmse_overall),
            "MAE": float(mae_overall),
            "Corr": float(corr_overall) if np.isfinite(corr_overall) else None,
        },

        # 便于你后面论文表直接取
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_corr": float(np.nanmean(corrs)),
        "std_corr": float(np.nanstd(corrs)),

        # 兼容你 baseline 那种数组风格
        "mae_d": [float(x) for x in maes],
        "rmse_d": [float(x) for x in rmses],
        "r2_d": [float(x) for x in r2s],
        "corr_d": [float(x) if np.isfinite(x) else None for x in corrs],
    }

    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)

    # -------------------------
    # CSV
    # -------------------------
    with open(os.path.join(out_dir, f"metrics_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Channel", "R2", "RMSE", "MAE", "Corr"])
        for i in range(3):
            w.writerow([
                labs[i],
                f"{r2s[i]:.4f}",
                f"{rmses[i]:.4f}",
                f"{maes[i]:.4f}",
                f"{corrs[i]:.4f}" if np.isfinite(corrs[i]) else "nan",
            ])

        w.writerow([
            "Overall",
            f"{r2_overall:.4f}",
            f"{rmse_overall:.4f}",
            f"{mae_overall:.4f}",
            f"{corr_overall:.4f}" if np.isfinite(corr_overall) else "nan",
        ])
        w.writerow(["Mean RMSE", "", f"{np.mean(rmses):.4f}", "", ""])
        w.writerow(["Std RMSE", "", f"{np.std(rmses):.4f}", "", ""])
        w.writerow(["Mean MAE", "", "", f"{np.mean(maes):.4f}", ""])
        w.writerow(["Std MAE", "", "", f"{np.std(maes):.4f}", ""])
        w.writerow(["Mean Corr", "", "", "", f"{np.nanmean(corrs):.4f}"])
        w.writerow(["Std Corr", "", "", "", f"{np.nanstd(corrs):.4f}"])

    print(
        f"[METRICS/{tag}] "
        f"R² per-channel={['%.3f' % v for v in r2s]} overall={r2_overall:.3f} | "
        f"RMSE per-channel={['%.4f' % v for v in rmses]} overall={rmse_overall:.4f} | "
        f"MAE per-channel={['%.4f' % v for v in maes]} overall={mae_overall:.4f} | "
        f"Corr per-channel={[('%.3f' % v) if np.isfinite(v) else 'nan' for v in corrs]} "
        f"overall={corr_overall:.3f}"
    )








def _ricker(length=101, dt=0.001, fdom=45.0):
    import numpy as np
    t = np.arange(length)*dt - (length*dt)/2
    w = (1 - 2*(np.pi*fdom*t)**2) * np.exp(-(np.pi*fdom*t)**2)
    return w.astype(np.float32)
@torch.no_grad()
def predict_distribution_t(model, Xb, ds, device, Tmc: int = 30, temp_scale=None, df: float = 4.0):
    """
    Xb: torch Tensor (支持 2.5D: [B,A,line_ctx,win] 或 1D: [B,in_dim])
    return:
      MU_phys, SD_phys   (物理单位, SD 是 t predictive std 的 epi+ale 合成)
    """

    df_use = float(getattr(model, "student_df", df))
    was_training = model.training
    model.train()  # 采样模式

    mus = []
    ales = []

    for _ in range(int(Tmc)):
        try:
            out = model(Xb, return_kl=False)
        except TypeError:
            out = model(Xb)

        pred, logvar, _ = _unpack_forward(out, device)

        if (logvar is not None) and (temp_scale is not None):
            logvar = _apply_temp_to_logvar(logvar, temp_scale)

        mus.append(pred.detach().cpu())

        if logvar is not None:
            scale = torch.exp(0.5 * logvar.detach().cpu())  # t 的 scale
            std_t = scale * float((df_use / (df_use - 2.0)) ** 0.5)  # ✅ predictive std
            ales.append(std_t)

    model.train(was_training)

    mus_t = torch.stack(mus, 0)      # (Tmc, B, 3)
    mu_n  = mus_t.mean(0).numpy()
    epi_n = mus_t.std(0).numpy()

    if len(ales) > 0:
        ale_n = torch.stack(ales, 0).mean(0).numpy()
        std_n = np.sqrt(epi_n**2 + ale_n**2)
    else:
        std_n = epi_n

    Y_mean = np.asarray(ds.Y_mean, dtype=np.float32).reshape(1, -1)
    Y_std  = np.asarray(ds.Y_std,  dtype=np.float32).reshape(1, -1)

    MU_phys = mu_n * Y_std + Y_mean
    SD_phys = std_n * Y_std
    return MU_phys, SD_phys


def _maybe_add_line_diff(Xb: torch.Tensor, ds) -> torch.Tensor:
    """
    若 ds.add_line_diff=True 且 Xb 是 [B, A, line_ctx, win]，则拼成 [B, 2A, line_ctx, win]:
      Xd = X - X_center
      cat([X, Xd], dim=1)
    """
    import torch
    if (not torch.is_tensor(Xb)) or Xb.dim() != 4:
        return Xb
    if not bool(getattr(ds, "add_line_diff", False)):
        return Xb

    A = int(len(getattr(ds, "angles_idx", []))) if getattr(ds, "angles_idx", None) is not None else int(Xb.size(1))
    # A 的更稳取法：以当前 Xb 的通道数作为 raw A
    A = int(Xb.size(1))

    # 已经是 2A 就不动
    if Xb.size(1) == 2 * A:
        return Xb

    hl = Xb.size(2) // 2
    Xc = Xb[:, :, hl, :].unsqueeze(2)    # [B, A, 1, win]
    Xd = Xb - Xc                         # [B, A, line_ctx, win]
    return torch.cat([Xb, Xd], dim=1)    # [B, 2A, line_ctx, win]

@torch.no_grad()



@torch.no_grad()
def fit_sigma_temperature(model, loader, device, ds, conf_levels=(0.5,0.68,0.8,0.9,0.95), T=30):
    """
    在验证集上拟合一个标量 τ，使得采用 σ' = τ·σ 的覆盖率更接近名义置信度。
    返回: tau(float)
    """
    from scipy.stats import norm
    was_training = model.training
    model.train()
    mus = []
    for _ in range(T):
        step = []
        for X, _, _ in loader:
            X = X.to(device)
            out = model(X, return_kl=False)
            mu = out[0] if isinstance(out, tuple) else out
            step.append(mu.detach().cpu())
        mus.append(torch.cat(step, 0))
    model.train(was_training)

    mu_n = torch.stack(mus,0).mean(0).numpy()
    epi  = torch.stack(mus,0).std(0).numpy()
    # 反归一化
    y_true = []
    for _, y, _ in loader: y_true.append(y)
    y_true = torch.cat(y_true,0).numpy() * ds.Y_std[None,:] + ds.Y_mean[None,:]
    mu_d   = mu_n * ds.Y_std[None,:] + ds.Y_mean[None,:]
    std_d  = epi  * ds.Y_std[None,:]

    # 用简单的一维 τ 最小化平方误差
    def coverage(tau):
        covs=[]
        for a in conf_levels:
            z = norm.ppf((1.0+a)/2.0)
            inside = ((y_true >= mu_d - z*(tau*std_d)) & (y_true <= mu_d + z*(tau*std_d))).mean()
            covs.append(inside)
        return np.array(covs)

    import numpy as np
    taus = np.linspace(0.6, 2.0, 50)
    best_tau, best_err = 1.0, 1e9
    target = np.array(conf_levels)
    for t in taus:
        e = np.mean((coverage(t) - target)**2)
        if e < best_err:
            best_err, best_tau = e, float(t)
    return best_tau

@torch.no_grad()




@torch.no_grad()
def _gather_val_truth_denorm(val_loader, ds_all):
    import numpy as np
    ys = []
    for batch in val_loader:
        if len(batch) == 4:
            _, y, _, _ = batch
        elif len(batch) == 3:
            _, y, _ = batch
        else:
            raise ValueError(f"_gather_val_truth_denorm: unexpected batch size {len(batch)}")
        ys.append(ds_all.denormalize_y(y).cpu().numpy())
    return np.concatenate(ys, axis=0)



@torch.no_grad()
def _gather_val_flags(val_loader):
    """
    收集所有样本是否属于井附近的标记，兼容 3/4 元组。
    """
    flags = []
    for batch in val_loader:
        if len(batch) == 4:
            _, _, _, meta = batch
        elif len(batch) == 3:
            _, _, meta = batch
        else:
            raise ValueError(f"_gather_val_flags: unexpected batch size {len(batch)}")

        for m in meta:
            # meta = (l, t, tau, is_well)
            if isinstance(m, (list, tuple)) and len(m) >= 4:
                flags.append(bool(m[3]))
            else:
                flags.append(False)
    return np.array(flags, dtype=bool)


def _ricker_vec(L=101, dt=0.001, fdom=45.0):
    import numpy as np
    t = np.arange(L)*dt - (L*dt)/2
    w = (1 - 2*(np.pi*fdom*t)**2) * np.exp(-(np.pi*fdom*t)**2)
    return w.astype(np.float32)


@torch.no_grad()
def well_vertical_resolution_check(model, ds, device, l, t, out_dir,
                                   Tmc: int = 30,
                                   fdom: float = 45.0,
                                   dt: float = 0.001,
                                   tag: str = "well_check",
                                   temp_scale=None,
                                   bl_mode: str = "ricker",     # "ricker" / "soft_bandpass"
                                   bl_taper: float = 0.25,
                                   bl_bw_frac: float = 0.90,
                                   spec_norm: bool = False,
                                   df: float = 4.0):
    """
    ✅ 2.5D 口径：输入构造为 [B, A, line_ctx, win]
    ✅ 中心点兼容版：
      - 当前模型输出 (mu_attr, chol_params, d_rec, kl)
      - 取中心时间点 t0 = win//2
      - 由联合协方差对角线得到 ±1σ
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import rfft, rfftfreq
    from scipy.signal import fftconvolve, correlate
    from scipy.stats import pearsonr
    import torch

    os.makedirs(out_dir, exist_ok=True)
    df_use = float(getattr(model, "student_df", df))
    # ---------- 小工具 ----------
    def _ricker_vec(L=101, dt=0.001, fdom=45.0):
        tt = (np.arange(L, dtype=np.float32) - L // 2) * dt
        x = np.pi * fdom * tt
        w = (1.0 - 2.0 * x ** 2) * np.exp(-x ** 2)
        return w.astype(np.float32)

    def _normalize_wavelet_energy(w: np.ndarray, eps=1e-12):
        e = float(np.sqrt(np.sum(w * w)) + eps)
        return (w / e).astype(np.float32)

    def _fft_soft_bandpass_1d(y: np.ndarray, dt: float, f_low: float, f_high: float, taper: float = 0.25):
        y = np.asarray(y, dtype=np.float32)
        n = y.size
        if n < 4:
            return y.copy()
        m = float(y.mean())
        x = y - m

        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, d=dt)

        bw = max(1e-6, float(f_high - f_low))
        tw = float(taper) * bw

        w = np.zeros_like(f, dtype=np.float32)

        core = (f >= (f_low + tw)) & (f <= (f_high - tw))
        w[core] = 1.0

        left = (f >= f_low) & (f < (f_low + tw))
        if np.any(left):
            u = (f[left] - f_low) / max(tw, 1e-6)
            w[left] = 0.5 - 0.5 * np.cos(np.pi * u)

        right = (f > (f_high - tw)) & (f <= f_high)
        if np.any(right):
            u = (f_high - f[right]) / max(tw, 1e-6)
            w[right] = 0.5 - 0.5 * np.cos(np.pi * u)

        Xf = X * w
        xr = np.fft.irfft(Xf, n=n).astype(np.float32)
        return xr + m

    def _bandlimit_true(y: np.ndarray, dt: float, fdom: float, mode: str):
        y = np.asarray(y, dtype=np.float32)

        mode = str(mode).lower()
        if mode == "soft_bandpass":
            nyq = 0.5 / float(dt)
            bw = max(1e-6, float(bl_bw_frac) * float(fdom))
            f_low = max(0.0, float(fdom) - 0.5 * bw)
            f_high = min(nyq * 0.999, float(fdom) + 0.5 * bw)
            return _fft_soft_bandpass_1d(y, dt=dt, f_low=f_low, f_high=f_high, taper=float(bl_taper))

        w = _ricker_vec(L=101, dt=dt, fdom=fdom)
        w = _normalize_wavelet_energy(w)

        m = float(y.mean())
        x = y - m

        pad = len(w) // 2
        if pad > 0:
            xpad = np.pad(x, (pad, pad), mode="reflect")
        else:
            xpad = x

        ypad = fftconvolve(xpad, w, mode="same").astype(np.float32)
        if pad > 0:
            y_bl = ypad[pad:-pad]
        else:
            y_bl = ypad
        return y_bl + m

    def _spectrum(y: np.ndarray, dt_use: float):
        y = np.asarray(y, dtype=np.float32)
        y = y - y.mean()
        Yf = np.abs(rfft(y))
        f = rfftfreq(y.size, d=dt_use)

        p = Yf + 1e-12
        centroid = float((f * p).sum() / p.sum())

        th = Yf.max() * (10 ** (-6 / 20))
        above = np.where(Yf >= th)[0]
        if above.size >= 2:
            bw_6db = float(f[above[-1]] - f[above[0]])
        else:
            bw_6db = 0.0
        return f, Yf, centroid, bw_6db

    def _ncc_peak(x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float32) - np.float32(x.mean())
        y = y.astype(np.float32) - np.float32(y.mean())
        if x.std() < 1e-8 or y.std() < 1e-8:
            return 0.0, 0
        xz = x / (x.std() + 1e-8)
        yz = y / (y.std() + 1e-8)
        c = correlate(xz, yz, mode="full")
        lags = np.arange(-len(y) + 1, len(x))
        k = int(np.argmax(c))
        return float(c[k] / len(x)), int(lags[k])

    # ---------- 边界 & 参数 ----------
    Ls, Ts, A_all, N_stack = ds.stack.shape
    assert 0 <= l < Ls and 0 <= t < Ts, "井位越界"
    h = ds.win // 2

    line_ctx = int(getattr(ds, "line_ctx", 1))
    if line_ctx < 1:
        line_ctx = 1
    r = line_ctx // 2

    tau_stride = int(getattr(ds, "time_stride", 1))
    if tau_stride < 1:
        tau_stride = 1

    # ✅ 频谱用的 dt（考虑 stride）
    dt_eff = float(dt) * float(tau_stride)

    # ---------- angles_idx -> array ----------
    ang = getattr(ds, "angles_idx", None)
    if ang is None:
        ang_idx = None
        A_sel = A_all
        X_mean_all = np.asarray(ds.X_mean, dtype=np.float32)
        X_std_all = np.asarray(ds.X_std, dtype=np.float32)
    elif isinstance(ang, slice):
        ang_idx = np.arange(A_all, dtype=np.int64)[ang]
        A_sel = int(len(ang_idx))
        X_mean_all = np.asarray(ds.X_mean, dtype=np.float32)
        X_std_all = np.asarray(ds.X_std, dtype=np.float32)
    else:
        ang_idx = np.asarray(ang, dtype=np.int64).reshape(-1)
        A_sel = int(len(ang_idx))
        X_mean_all = np.asarray(ds.X_mean, dtype=np.float32)
        X_std_all = np.asarray(ds.X_std, dtype=np.float32)

    use_angle_time = (X_mean_all.ndim == 2)

    # ---------- 真值（Mod） ----------
    Y = ds.mod[l, t, ds.props_idx, :].astype(np.float32)
    if Y.ndim == 2 and Y.shape[0] == 3:
        Y = Y.T
    elif Y.ndim == 2 and Y.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected Mod shape at well: {Y.shape}, expect (3,N) or (N,3)")
    N = int(Y.shape[0])
    assert N > 2 * h + 1, "井剖面长度太短，不足以支撑当前 win。"

    # ---------- Band-limited True（全长生成，再采样） ----------
    Y_bl = np.empty_like(Y)
    for i in range(3):
        Y_bl[:, i] = _bandlimit_true(Y[:, i], dt=float(dt), fdom=float(fdom), mode=str(bl_mode))

    # ---------- 2.5D：line 索引（固定不随 tau 变） ----------
    l0 = max(0, l - r)
    l1 = min(Ls - 1, l + r)
    ls = list(range(l0, l1 + 1))
    while len(ls) < line_ctx:
        if ls[0] > 0:
            ls = [ls[0] - 1] + ls
        elif ls[-1] < Ls - 1:
            ls = ls + [ls[-1] + 1]
        else:
            ls = [ls[0]] * line_ctx
            break
    if len(ls) > line_ctx:
        mid = len(ls) // 2
        half = line_ctx // 2
        ls = ls[mid - half: mid - half + line_ctx]
    ls_idx = np.asarray(ls, dtype=np.int64)

    # ---------- tau_list（与训练一致） ----------
    tau_list = list(range(h, N - h, tau_stride))
    if len(tau_list) == 0:
        print(f"[WARN] no tau windows at (l={l}, t={t}) with win={ds.win}, stride={tau_stride}")
        return

    # ---------- 构造 2.5D 输入 Xb ----------
    X_list = []
    for tau in tau_list:
        s0 = tau - h
        s1 = tau + h + 1

        tmp = ds.stack[ls_idx, t, :, s0:s1]      # [line_ctx, A_all, win]
        if ang_idx is not None:
            tmp = tmp[:, ang_idx, :]             # [line_ctx, A_sel, win]

        Xctx = np.transpose(tmp, (1, 0, 2)).astype(np.float32)  # [A_sel, line_ctx, win]

        if use_angle_time:
            if ang_idx is None:
                Xm = X_mean_all[:, :].astype(np.float32)[:, None, :]   # [A_sel,1,win]
                Xs = X_std_all[:, :].astype(np.float32)[:, None, :]
            else:
                Xm = X_mean_all[ang_idx, :].astype(np.float32)[:, None, :]
                Xs = X_std_all[ang_idx, :].astype(np.float32)[:, None, :]
        else:
            if ang_idx is None:
                Xm = X_mean_all.reshape(-1).astype(np.float32)[:, None, None]
                Xs = X_std_all.reshape(-1).astype(np.float32)[:, None, None]
            else:
                Xm = X_mean_all.reshape(-1)[ang_idx].astype(np.float32)[:, None, None]
                Xs = X_std_all.reshape(-1)[ang_idx].astype(np.float32)[:, None, None]

        Xn = (Xctx - Xm[:Xctx.shape[0]]) / (Xs[:Xctx.shape[0]] + 1e-8)
        X_list.append(torch.from_numpy(Xn).float())

    Xb = torch.stack(X_list, 0).to(device)  # [T_eff, A_sel, line_ctx, win]

    # ---------- add_line_diff 与训练一致 ----------
    if bool(getattr(ds, "add_line_diff", False)):
        A = int(A_sel)
        if Xb.dim() == 4 and Xb.size(1) == A:
            hl = Xb.size(2) // 2
            Xc = Xb[:, :, hl, :].unsqueeze(2)  # [T_eff, A, 1, win]
            Xd = Xb - Xc
            Xb = torch.cat([Xb, Xd], dim=1)    # [T_eff, 2A, line_ctx, win]
        else:
            print(f"[WELL-CHECK][WARN] add_line_diff=True but Xb.shape={tuple(Xb.shape)} not [B,A,lc,win]; skip diff.")

    T_eff = int(Xb.shape[0])
    depth_axis = np.asarray(tau_list, dtype=np.int64)

    # ---------- MC 预测（μ/σ），并反归一化 ----------
    was_training = bool(model.training)
    model.train()

    mu_samples = []
    scale_samples = []

    try:
        for _ in range(int(Tmc)):
            out = model(Xb, return_kl=False) if "return_kl" in model.forward.__code__.co_varnames else model(Xb)
            if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                raise RuntimeError(
                    "[well_vertical_resolution_check] Expected model output like "
                    "(mu_attr, chol_params, d_rec[, kl])"
                )

            mu_attr = out[0]
            chol_params = out[1]

            if mu_attr.dim() != 3 or mu_attr.size(-1) != 3:
                raise RuntimeError(f"[well_vertical_resolution_check] mu_attr bad shape={tuple(mu_attr.shape)}")
            if chol_params.dim() != 3 or chol_params.size(-1) != 6:
                raise RuntimeError(f"[well_vertical_resolution_check] chol_params bad shape={tuple(chol_params.shape)}")

            t0 = int(mu_attr.shape[1] // 2)

            mu_c_n = mu_attr[:, t0, :]                       # [T_eff,3]
            chol_c = chol_params[:, t0, :].unsqueeze(1)      # [T_eff,1,6]

            L_c = build_scale_tril_3x3(chol_c)               # [T_eff,1,3,3]
            cov_c = scale_tril_to_cov(L_c)                   # [T_eff,1,3,3]
            var_c_n = torch.diagonal(cov_c, dim1=-2, dim2=-1).squeeze(1)  # [T_eff,3]
            sd_c_n = torch.sqrt(torch.clamp(var_c_n, min=1e-12))

            mu_samples.append(mu_c_n.detach().cpu())
            scale_samples.append(sd_c_n.detach().cpu())

    finally:
        model.train(was_training) if was_training else model.eval()

    mu_stack = torch.stack(mu_samples, axis=0)           # [Tmc,T_eff,3]
    scale_stack = torch.stack(scale_samples, axis=0)     # [Tmc,T_eff,3]

    mu_mean_n = mu_stack.mean(axis=0)                    # [T_eff,3]
    var_epi_n = mu_stack.var(axis=0, unbiased=False)     # [T_eff,3]
    mean_var_ale_n = (scale_stack ** 2).mean(axis=0) * (float(df) / (float(df) - 2.0) if float(df) > 2.0 else 1.0)
    std_n = torch.sqrt(torch.clamp(var_epi_n + mean_var_ale_n, min=1e-12))  # [T_eff,3]

    Y_mean_t = torch.as_tensor(ds.Y_mean, dtype=mu_mean_n.dtype).view(1, 3)
    Y_std_t = torch.as_tensor(ds.Y_std, dtype=mu_mean_n.dtype).view(1, 3)

    MU = (mu_mean_n * Y_std_t + Y_mean_t).cpu().numpy()
    SD = (std_n * Y_std_t).cpu().numpy()

    # ---------- True/BL True：采样到 tau_list 对齐 ----------
    Yc = Y[np.asarray(tau_list, dtype=np.int64), :]
    Y_blc = Y_bl[np.asarray(tau_list, dtype=np.int64), :]

    n = min(T_eff, Yc.shape[0], Y_blc.shape[0], MU.shape[0], SD.shape[0], depth_axis.shape[0])
    depth_axis = depth_axis[:n]
    MU = MU[:n]
    SD = SD[:n]
    Yc = Yc[:n]
    Y_blc = Y_blc[:n]

    labs = [r"$V_p$",r"$V_s$",r"$\rho$"]

    # ========== NCC plot only (paper/GitHub retained diagnostic) ==========
    # ========== 3) NCC ==========
    plt.figure(figsize=(12, 4))
    for i in range(3):
        ncc, lag = _ncc_peak(MU[:, i], Yc[:, i])
        lag_raw = int(lag * tau_stride)

        x = MU[:, i] - MU[:, i].mean()
        yv = Yc[:, i] - Yc[:, i].mean()
        x /= (x.std() + 1e-8)
        yv /= (yv.std() + 1e-8)
        c = correlate(x, yv, mode="full") / len(x)
        lags = np.arange(-len(yv) + 1, len(x))

        ax = plt.subplot(1, 3, i + 1)
        ax.plot(lags, c)
        ax.axvline(lag, color='r', ls='--', label=f"lag={lag} (raw≈{lag_raw})")
        ax.set_title(f"NCC — {labs[i]} peak={ncc:.3f} lag={lag} (raw≈{lag_raw})")
        ax.grid(alpha=.3)
        ax.legend()
    plt.tight_layout()
    f3 = os.path.join(out_dir, f"{tag}_ncc_l{l}_t{t}.png")
    plt.savefig(f3, dpi=220, bbox_inches="tight")
    plt.close()

    # ========== 4) 数值指标 ==========
    mets = []
    for i in range(3):
        rmse = float(np.sqrt(np.mean((MU[:, i] - Yc[:, i]) ** 2)))
        r, _ = pearsonr(MU[:, i], Yc[:, i])
        ncc, lag = _ncc_peak(MU[:, i], Yc[:, i])
        lag_raw = int(lag * tau_stride)
        _, _, cP, bwP = _spectrum(MU[:, i], dt_eff)
        mets.append((labs[i], rmse, r, ncc, lag, lag_raw, cP, bwP))

    txt = os.path.join(out_dir, f"{tag}_metrics_l{l}_t{t}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Channel, RMSE, Pearson_r, NCC_peak, NCC_lag[subsample], NCC_lag[raw], "
                "SpectralCentroid[Hz], BW_-6dB[Hz]\n")
        for (name, rmse, r, ncc, lag, lag_raw, cP, bwP) in mets:
            f.write(f"{name}, {rmse:.6f}, {r:.6f}, {ncc:.6f}, {lag}, {lag_raw}, {cP:.2f}, {bwP:.2f}\n")

    print("[SAVE]", f3)
    print("[SAVE]", txt)



import os, random
from collections import defaultdict

def _group_indices_by_lt(ds, indices):
    g = defaultdict(list)
    for idx in indices:
        l, t, tau = ds.samples[int(idx)]
        g[(int(l), int(t))].append(int(idx))
    for k in g:
        g[k].sort(key=lambda ii: int(ds.samples[ii][2]))
    return g

def build_section_indices_for_trace(ds_all, t_fixed: int):
    """
    固定 trace=t_fixed，收集 (line=l, tau) 的样本索引，构成 line–time 网格。

    返回：
      l_list:   sorted unique lines
      tau_list: sorted unique taus
      valid_gidx: 按 (li, tj) 扫描顺序排列的 global idx（稳定！）
      valid_pos:  与 valid_gidx 对齐的 (li, tj)
      grid_idx:   [L, Nt] full-grid 映射（缺失=-1），便于检查/可视化
      missing:    缺失点数量（missing>0 基本必然块状）
    """
    import numpy as np

    if not hasattr(ds_all, "samples"):
        raise RuntimeError("[B2] ds_all.samples 不存在。")

    samples = np.asarray(ds_all.samples)  # [N,3] -> (l,t,tau)
    l_arr   = samples[:, 0].astype(np.int64)
    t_arr   = samples[:, 1].astype(np.int64)
    tau_arr = samples[:, 2].astype(np.int64)

    mask = (t_arr == int(t_fixed))
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return [], [], [], [], None, 0

    # 轴
    l_list   = np.unique(l_arr[idxs]);   l_list.sort()
    tau_list = np.unique(tau_arr[idxs]); tau_list.sort()

    L, Nt = int(l_list.size), int(tau_list.size)
    l2i   = {int(v): i for i, v in enumerate(l_list.tolist())}
    tau2i = {int(v): j for j, v in enumerate(tau_list.tolist())}

    # full-grid 映射：缺失=-1
    grid_idx = -np.ones((L, Nt), dtype=np.int64)

    dup = 0
    for gi in idxs:
        li = l2i[int(l_arr[gi])]
        tj = tau2i[int(tau_arr[gi])]
        if grid_idx[li, tj] >= 0:
            dup += 1
            continue
        grid_idx[li, tj] = int(gi)

    missing = int(np.sum(grid_idx < 0))
    if missing > 0:
        print(f"[B2][WARN] trace t={t_fixed}: missing={missing}/{L*Nt} (缺点越多越块状)")
    if dup > 0:
        print(f"[B2][WARN] trace t={t_fixed}: duplicated (l,tau) ignored dup={dup}")

    # ✅ 稳定顺序：按 li->tj 扫描（你永远知道 k 对应哪个格子）
    valid_pos = np.argwhere(grid_idx >= 0)                # [M,2] (li,tj)
    valid_gidx = grid_idx[grid_idx >= 0].astype(np.int64) # [M]

    return (
        l_list.tolist(),
        tau_list.tolist(),
        valid_gidx.tolist(),
        [tuple(map(int, x)) for x in valid_pos],
        grid_idx,
        missing
    )



def mc_predict_section_mean(model, ds_all, device, t_fixed: int,
                            Tmc=30, temp_scale=None,
                            batch_size=2048, collate_fn=None,
                            mc_train_mode=True):
    """
    固定 trace=t_fixed，对整张 line–time 剖面做 MC 预测，输出物理空间均值剖面（full-grid 装配）。
    返回 dict:
      mean: [L, Nt, 3] (VP/VS/RHOB) 物理单位
      l_list, tau_list
      missing: 缺失点数量（full-grid 缺点会导致块状）
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    # -----------------------------
    # B2-FIG9-① full-grid 索引构造
    # -----------------------------
    if not hasattr(ds_all, "samples"):
        raise RuntimeError("[FIG9] ds_all.samples 不存在，无法构造 2D 剖面索引。")

    samples = np.asarray(ds_all.samples)
    l_arr   = samples[:, 0].astype(int)
    t_arr   = samples[:, 1].astype(int)
    tau_arr = samples[:, 2].astype(int)

    mask = (t_arr == int(t_fixed))
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None

    l_list   = np.unique(l_arr[idxs]);   l_list.sort()
    tau_list = np.unique(tau_arr[idxs]); tau_list.sort()

    L, Nt = len(l_list), len(tau_list)
    l2i   = {int(v): i for i, v in enumerate(l_list)}
    tau2i = {int(v): j for j, v in enumerate(tau_list)}

    grid_idx = -np.ones((L, Nt), dtype=np.int64)  # full-grid: 缺点=-1
    dup = 0
    for gi in idxs:
        ii = l2i[int(l_arr[gi])]
        jj = tau2i[int(tau_arr[gi])]
        if grid_idx[ii, jj] >= 0:
            dup += 1
            continue
        grid_idx[ii, jj] = int(gi)

    missing = int(np.sum(grid_idx < 0))
    if missing > 0:
        print(f"[FIG9][WARN] section not full grid: missing={missing}/{L*Nt} -> 图可能块状/断裂")
    if dup > 0:
        print(f"[FIG9][WARN] duplicated (l,tau) ignored: dup={dup}")

    # 只跑 valid 点（顺序固定）
    valid_pos = np.argwhere(grid_idx >= 0)             # [M,2] (li, tj)
    valid_gidx = grid_idx[grid_idx >= 0].astype(int)   # [M]
    M = int(valid_gidx.shape[0])
    if M == 0:
        return None

    sub = Subset(ds_all, valid_gidx.tolist())
    loader = DataLoader(
        sub,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn
    )

    # -----------------------------
    # B2-FIG9-② MC 模式：建议 train() 做 MC（dropout/BNN更像MC）
    # -----------------------------
    prev_train = model.training
    if bool(mc_train_mode):
        model.train()
    else:
        model.eval()

    mean_n, std_n = mc_predict_loader(model, loader, device, T=int(Tmc), temp_scale=temp_scale)

    # 恢复模型状态
    model.train(prev_train)

    # 还原到物理空间（均值）
    mean_d = mean_n * ds_all.Y_std[None, :] + ds_all.Y_mean[None, :]

    # -----------------------------
    # B2-FIG9-③ pack 回 full-grid
    # -----------------------------
    C = mean_d.shape[1]
    sec_mean = np.full((L, Nt, C), np.nan, dtype=np.float32)

    # mean_d 的顺序 == loader 的顺序 == valid_gidx 的顺序
    for k, (li, tj) in enumerate(valid_pos):
        sec_mean[int(li), int(tj), :] = mean_d[k, :]

    return {"mean": sec_mean, "l_list": l_list.tolist(), "tau_list": tau_list.tolist(), "missing": missing}

def mc_predict_section_mean_fullgrid(
    model, ds_all, device, t_fixed: int,
    l_list_full, tau_list_full,
    Tmc=30, temp_scale=None,
    batch_size=2048, collate_fn=None,
    mc_train_mode=True,
    # ✅ NEW: 控制 2.5D line_ctx 的推理使用方式（不改网络结构）
    ctx_mode="none",        # "none" | "center_repeat" | "mid3_keep"
    keep_k=3,               # mid3_keep 保留中间 k 条（推荐 3）
):
    """
    dataset 侧补齐 full-grid：对给定 (l_list_full × tau_list_full) 上的所有点真正跑模型
    返回 dict: mean[L,Nt,3], l_list, tau_list, missing=0

    ctx_mode:
      - "none": 原样使用 X 的 line_ctx
      - "center_repeat": 只用中心线（中心线复制到所有 line_ctx）
      - "mid3_keep": 只保留中间 k 条真实横向，其余用中心线填充（shape 不变）
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    # -----------------------------
    # 1) 构造 full-grid samples
    # -----------------------------
    l_list_full = list(map(int, l_list_full))
    tau_list_full = list(map(int, tau_list_full))
    samples_full = [(int(l), int(t_fixed), int(tau)) for l in l_list_full for tau in tau_list_full]

    # -----------------------------
    # 2) full-grid dataset wrapper
    # -----------------------------
    ds_full = FullGridDatasetForSection(ds_all, samples_full)

    loader = DataLoader(
        ds_full,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(getattr(device, "type", "") == "cuda"),
        drop_last=False,
        collate_fn=collate_fn
    )

    # -----------------------------
    # 3) MC 推理（逐 batch，支持 ctx_mode）
    #    我们不直接用 mc_predict_loader（因为要改 X）
    # -----------------------------
    prev_train = model.training
    model.train() if bool(mc_train_mode) else model.eval()

    preds_T = []  # list of [M_batch, 3] per MC, then concat
    with torch.no_grad():
        # 先把所有 batch 的 X 缓存到 device 侧（避免每次 MC 重读/重拷贝）
        # 但如果显存不够，可以改成不缓存（按 MC 循环读 batch）
        cached_batches = []
        for batch in loader:
            # 兼容 batch 结构：可能是 (X, meta) / (X,y,meta) / (X,y,y_bl,meta)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    X, y, y_bl, metas = batch
                elif len(batch) == 3:
                    X, y, metas = batch
                    y_bl = None
                elif len(batch) == 2:
                    X, metas = batch
                    y = None
                    y_bl = None
                else:
                    # 兜底：把第一个当 X，最后一个当 metas
                    X = batch[0]
                    metas = batch[-1]
                    y = None
                    y_bl = None
            else:
                raise RuntimeError("[FIG9][FULLGRID] unexpected batch type")

            X = X.to(device, non_blocking=True)

            # ===== ctx_mode 处理：X shape [B, A, line_ctx, win] =====
            if ctx_mode in ("center_repeat", "mid3_keep"):
                if X.ndim >= 4:
                    lc = int(X.shape[2])
                    mid = lc // 2
                    X_mid = X[:, :, mid:mid + 1, :]          # [B,A,1,win]
                    X_new = X_mid.repeat(1, 1, lc, 1)        # 全部先填中心线

                    if ctx_mode == "mid3_keep":
                        k = int(keep_k)
                        if k < 1:
                            k = 1
                        if k % 2 == 0:
                            k += 1
                        if k > lc:
                            k = lc if (lc % 2 == 1) else max(1, lc - 1)

                        half = k // 2
                        lo = max(0, mid - half)
                        hi = min(lc, mid + half + 1)
                        X_new[:, :, lo:hi, :] = X[:, :, lo:hi, :]  # 中间 k 条真实横向放回去

                    X = X_new
                else:
                    # shape 不符合 2.5D，就忽略
                    pass

            cached_batches.append((X, y, y_bl, metas))

        # MC loop
        T = int(Tmc)
        all_mc = []

        for k in range(T):
            out_list = []
            for (X, y, y_bl, metas) in cached_batches:
                out = model(X, return_kl=True)

                if not (isinstance(out, (tuple, list)) and len(out) >= 4):
                    raise RuntimeError(
                        "[FIG9][FULLGRID] Expected model output like (mu_attr, chol_params, d_rec, kl)"
                    )

                extra = None
                if isinstance(out, (tuple, list)):
                    if len(out) >= 5:
                        mu_attr, chol_params, d_rec, kl, extra = out[:5]
                    elif len(out) >= 4:
                        mu_attr, chol_params, d_rec, kl = out[:4]
                    else:
                        raise RuntimeError(f"Unexpected model output len={len(out)}")
                else:
                    raise RuntimeError("Model output must be tuple/list")

                # ✅ 中心点兼容：mu_attr [B,win,3] -> [B,3]
                if mu_attr.dim() == 3 and mu_attr.size(-1) == 3:
                    t0 = int(mu_attr.shape[1] // 2)
                    mu = mu_attr[:, t0, :]   # [B,3]
                elif mu_attr.dim() == 2 and mu_attr.size(-1) == 3:
                    mu = mu_attr
                else:
                    raise RuntimeError(f"[FIG9][FULLGRID] unexpected mu_attr shape={tuple(mu_attr.shape)}")

                # denorm to phys
                Ym = torch.as_tensor(ds_all.Y_mean, device=mu.device, dtype=mu.dtype).view(1, -1)
                Ys = torch.as_tensor(ds_all.Y_std,  device=mu.device, dtype=mu.dtype).view(1, -1)
                mu_d = mu * Ys + Ym   # [B,3]

                out_list.append(mu_d.detach().float().cpu().numpy())

            mc_k = np.concatenate(out_list, axis=0).astype(np.float32)  # [M,3]
            all_mc.append(mc_k)

        all_mc = np.stack(all_mc, axis=0).astype(np.float32)  # [T,M,3]
        mean_d = np.mean(all_mc, axis=0)                      # [M,3]
        # std_d = np.std(all_mc, axis=0)  # 如需可返回

    model.train(prev_train)

    # -----------------------------
    # 4) reshape 回 [L, Nt, 3]
    # -----------------------------
    L = len(l_list_full)
    Nt = len(tau_list_full)
    if mean_d.shape[0] != L * Nt:
        raise RuntimeError(f"[FIG9][FULLGRID] assemble mismatch: mean_d M={mean_d.shape[0]} != L*Nt={L*Nt}")

    sec_mean = mean_d.reshape(L, Nt, 3).astype(np.float32)

    return {
        "mean": sec_mean,
        "l_list": list(map(int, l_list_full)),
        "tau_list": list(map(int, tau_list_full)),
        "missing": 0
    }


def plot_section_mean_fig9_fullcrop(
    sec,
    out_png,
    dt=0.001,
    prop_names=(r"$V_p$", r"$V_s$", r"$\rho$"),
    prop_units=("m/s", "m/s", "g/cc"),
    fontsize=12,
    interpolation="bilinear",
    dx=10.0,
    x0=11300.0,
    dpi=300,
):
    """
    Fig9 predicted section:
      x axis: Distance (m)
      y axis: Time (ms)

    sec["mean"] shape: [L, Nt, 3]

    Layout:
      3 rows × 1 column
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if sec is None or (not isinstance(sec, dict)) or ("mean" not in sec):
        print("[FIG9-FULL][WARN] sec invalid or no 'mean'.")
        return False

    M = np.asarray(sec["mean"], dtype=np.float32)   # [L, Nt, 3]
    l_list = sec.get("l_list", None)
    tau_list = sec.get("tau_list", None)

    if l_list is None or tau_list is None:
        print("[FIG9-FULL][WARN] sec has no l_list/tau_list.")
        return False

    if M.ndim != 3 or M.shape[2] < 3:
        print(f"[FIG9-FULL][WARN] sec['mean'] shape bad: {M.shape}, expect [L,Nt,3].")
        return False

    l_arr = np.asarray(l_list, dtype=np.float32)
    tau_arr = np.asarray(tau_list, dtype=np.float32)

    # =========================
    # x 轴：Distance (m)
    # =========================
    x_vals = float(x0) + l_arr * float(dx)
    x_left = float(x_vals[0])
    x_right = float(x_vals[-1])
    xlabel = "Distance (m)"

    # =========================
    # y 轴：Time (ms)
    # =========================
    t_ms = tau_arr * float(dt) * 1000.0
    y_bottom = float(t_ms[-1])
    y_top = float(t_ms[0])
    ylabel = "Time (ms)"

    extent = [x_left, x_right, y_bottom, y_top]

    def robust_limits(img2d, qlo=2.0, qhi=98.0):
        x = img2d[np.isfinite(img2d)]
        if x.size < 10:
            if np.isfinite(img2d).any():
                return float(np.nanmin(img2d)), float(np.nanmax(img2d))
            return 0.0, 1.0

        vmin = float(np.percentile(x, qlo))
        vmax = float(np.percentile(x, qhi))

        if vmax <= vmin:
            vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))

        return vmin, vmax

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize,
        "xtick.labelsize": max(fontsize - 1, 1),
        "ytick.labelsize": max(fontsize - 1, 1),
    })

    # ==========================================================
    # ✅ 改成 3 行 1 列竖向排列
    # ==========================================================
    fig, axes = plt.subplots(
        3, 1,
        figsize=(8.5, 12.5),
        constrained_layout=True
    )

    title_list = [
        r"$V_p$",
        r"$V_s$",
        r"$\rho$",
    ]

    for i, ax in enumerate(axes):
        img = M[:, :, i].T   # [Nt, L]
        vmin, vmax = robust_limits(img, 2.0, 98.0)

        im = ax.imshow(
            img,
            aspect="auto",
            extent=extent,
            origin="upper",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title_list[i], pad=8)
        ax.set_ylabel(ylabel)

        # 只在最下面一幅图显示横坐标
        if i == 2:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
        if prop_units is not None and i < len(prop_units):
            cb.set_label(prop_units[i], rotation=90, labelpad=6)

        ax.grid(False)

    # =========================
    # 保存
    # =========================
    out_png = os.path.abspath(str(out_png))
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(out_png):
        print(f"[SAVE] {out_png} size={os.path.getsize(out_png) / 1024:.1f} KB")
        return True
    else:
        print(f"[SAVE-FAIL] {out_png}")
        return False





def run_eval_and_plots(model, ds_all, val_loader, device, args,
                       temp_scale=None,
                       train_indices=None,
                       val_indices=None,
                       cont_loader=None):
    """
    Minimal evaluation/export routine for GitHub release.

    This function keeps only the figures used in the manuscript:
      - parity_val.png and residuals_val.png
      - val_only_well_profile_l*_t*_clean.png
      - res_check_final_ncc_l*_t*.png
      - jointGlobal_CONT_GLOBAL_sample_N*_3Dscatter/3DKDE/corner.png
      - fig10_ALIGN_tidx*_trace_t*_muOnly_blinear.png
      - fig9_fullcrop_pred_mean.png
      - fig9_fullcrop_residual_t*.png

    JSON/NPZ metric and residual files are still saved.
    """
    import os
    import json
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Subset
    from scipy.stats import gaussian_kde

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    def _chk_file(fn):
        if os.path.exists(fn):
            print(f"[SAVE-CHECK] {fn} ({os.path.getsize(fn) / 1024:.1f} KB)", flush=True)
        else:
            print(f"[SAVE-CHECK][WARN] missing: {fn}", flush=True)

    def _take_center_from_seq(y):
        if torch.is_tensor(y):
            if y.dim() == 3:
                return y[:, y.shape[1] // 2, :]
            if y.dim() == 2 and y.shape[-1] in (3, 6):
                return y
            if y.dim() == 1 and y.shape[0] in (3, 6):
                return y
            raise RuntimeError(f"_take_center_from_seq: unexpected tensor shape={tuple(y.shape)}")
        y = np.asarray(y)
        if y.ndim == 3:
            return y[:, y.shape[1] // 2, :]
        if y.ndim == 2 and y.shape[-1] in (3, 6):
            return y
        if y.ndim == 1 and y.shape[0] in (3, 6):
            return y
        raise RuntimeError(f"_take_center_from_seq: unexpected ndarray shape={tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 1) Validation predictions, metrics, parity, and residual diagnostics
    # ------------------------------------------------------------------
    metas_all = []
    for batch in val_loader:
        metas_all.extend(list(batch[-1]))

    val_trues = np.asarray(_gather_val_truth_denorm(val_loader, ds_all))
    if val_trues.ndim == 3 and val_trues.shape[-1] == 3:
        val_trues = val_trues[:, val_trues.shape[1] // 2, :]
    elif val_trues.ndim != 2 or val_trues.shape[1] != 3:
        raise RuntimeError(f"[run_eval_and_plots] unexpected val_trues shape: {val_trues.shape}")

    mean_n, std_pred_n, scale_n = mc_predict_loader(
        model, val_loader, device,
        T=50, temp_scale=temp_scale,
        df=4.0,
        return_scale=True,
        dropout_only=True,
    )
    mean_n = np.asarray(mean_n)
    std_pred_n = np.asarray(std_pred_n)
    scale_n = np.asarray(scale_n)
    if mean_n.ndim != 2 or mean_n.shape[1] != 3:
        raise RuntimeError(f"[run_eval_and_plots] mean_n shape must be (N,3), got {mean_n.shape}")
    if val_trues.shape != mean_n.shape:
        raise RuntimeError(f"[run_eval_and_plots] val_trues shape mismatch: {val_trues.shape} vs {mean_n.shape}")

    mean_d = mean_n * ds_all.Y_std[None, :] + ds_all.Y_mean[None, :]
    std_pred_d = std_pred_n * ds_all.Y_std[None, :]

    # Simple post-hoc standard-deviation calibration used by the original script.
    resid = val_trues - mean_d
    alpha = np.sqrt((resid ** 2).mean(axis=0) / ((std_pred_d ** 2).mean(axis=0) + 1e-8))
    std_pred_d = std_pred_d * alpha[None, :]
    print("[CALIB] std_pred scale alpha =", alpha, flush=True)

    compute_metrics_and_save(val_trues, mean_d, out_dir, tag="val")
    plot_parity_and_residuals(val_trues, mean_d, out_dir, tag="val")

    # ------------------------------------------------------------------
    # 2) Pick the validation well and export the retained validation-well plots
    # ------------------------------------------------------------------
    def _pick_val_well_lt(ds_all, metas_all, val_indices=None, prefer_lt=None):
        val_well_lts = []
        if (val_indices is not None) and hasattr(ds_all, "samples") and hasattr(ds_all, "is_well_sample"):
            vi = np.asarray(val_indices, dtype=np.int64).reshape(-1)
            vi = vi[(vi >= 0) & (vi < len(ds_all))]
            if vi.size > 0:
                mask = np.asarray(ds_all.is_well_sample, dtype=bool)[vi]
                if mask.any():
                    lt = np.asarray(ds_all.samples, dtype=int)[vi[mask], :2]
                    val_well_lts = [(int(l), int(t)) for (l, t) in lt]
        if len(val_well_lts) == 0:
            for m in metas_all:
                if isinstance(m, (list, tuple)) and len(m) >= 4 and bool(m[3]):
                    val_well_lts.append((int(m[0]), int(m[1])))
        val_well_lts = sorted(set(val_well_lts), key=lambda x: (x[1], x[0]))
        if not val_well_lts:
            return [], None
        if prefer_lt is not None:
            prefer_lt = (int(prefer_lt[0]), int(prefer_lt[1]))
            if prefer_lt in val_well_lts:
                return val_well_lts, prefer_lt
        return val_well_lts, val_well_lts[0]

    preferred_val_lt = getattr(args, "_pseudo_val_lt", None)
    val_well_lts, val_lt = _pick_val_well_lt(ds_all, metas_all, val_indices=val_indices, prefer_lt=preferred_val_lt)
    if val_lt is None:
        print("[VAL][WARN] no validation well sample found. Skip well/section plots.", flush=True)
        return

    l0_val, t0_val = int(val_lt[0]), int(val_lt[1])
    print(f"[VAL] using validation well (l={l0_val}, t={t0_val})", flush=True)

    try:
        well_vertical_resolution_check(
            model, ds_all, device,
            l0_val, t0_val, out_dir,
            Tmc=50,
            fdom=args.dominant_freq,
            dt=args.dt,
            tag="res_check_final",
            temp_scale=temp_scale,
        )
    except Exception as e:
        print("[RES-CHECK][WARN] failed:", repr(e), flush=True)

    try:
        plot_profile_at(
            model, ds_all, device,
            l=l0_val, t=t0_val, out_dir=out_dir,
            tag="val_only_well_profile",
            Tmc=30,
            fdom=getattr(args, "dominant_freq", 45.0),
            dt=getattr(args, "dt", 0.001),
            temp_scale=temp_scale,
            space="depth",
        )
    except Exception as e:
        print("[PROFILE][WARN] failed:", repr(e), flush=True)

    # ------------------------------------------------------------------
    # 3) Global joint distribution: keep only sample-mode 3D scatter/KDE/corner
    # ------------------------------------------------------------------
    def _posterior_stats(samples):
        samples = np.asarray(samples, dtype=np.float64)
        return samples.mean(axis=0), samples.std(axis=0, ddof=1), np.cov(samples.T), np.corrcoef(samples.T)

    def _save_posterior_stats_txt(samples, out_txt):
        mu, std, cov, corr = _posterior_stats(samples)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("Posterior mean:\n")
            f.write(f"VP   = {mu[0]:.6f}\nVS   = {mu[1]:.6f}\nRHOB = {mu[2]:.6f}\n\n")
            f.write("Posterior std:\n")
            f.write(f"VP   = {std[0]:.6f}\nVS   = {std[1]:.6f}\nRHOB = {std[2]:.6f}\n\n")
            f.write("Covariance matrix:\n")
            f.write(np.array2string(cov, precision=6, suppress_small=False))
            f.write("\n\nCorrelation matrix:\n")
            f.write(np.array2string(corr, precision=6, suppress_small=False))

    def plot_joint_3d_scatter(samples, out_png, title="Global Joint Cloud"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        samples = np.asarray(samples, dtype=np.float32)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=6, alpha=0.22)
        ax.set_xlabel(r"$V_p$")
        ax.set_ylabel(r"$V_s$")
        ax.set_zlabel(r"$\rho$")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print("[SAVE]", out_png, flush=True)

    def plot_joint_3d_kde(samples, out_png, title="Global 3D Joint Density"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        samples = np.asarray(samples, dtype=np.float32)
        xyz = np.vstack([samples[:, 0], samples[:, 1], samples[:, 2]])
        density = gaussian_kde(xyz)(xyz)
        order = np.argsort(density)
        s = samples[order]
        density = density[order]
        fig = plt.figure(figsize=(8.8, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.08, 0.08, 0.84, 0.84])
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=density, s=8, alpha=0.35)
        ax.set_xlabel(r"$V_p$", labelpad=10)
        ax.set_ylabel(r"$V_s$", labelpad=10)
        ax.set_zlabel("")
        ax.text2D(1.02, 0.53, r"$\rho$", transform=fig.transFigure,
                  rotation=90, va="center", ha="center", fontsize=10)
        ax.set_title(title, pad=14)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)
        print("[SAVE]", out_png, flush=True)

    def plot_joint_corner(samples, out_png):
        try:
            import corner
        except Exception as e:
            print("[GLOBAL-JOINT][WARN] corner import failed:", repr(e), flush=True)
            return
        fig = corner.corner(
            np.asarray(samples, dtype=np.float32),
            labels=[r"$V_p$", r"$V_s$", r"$\rho$"],
            show_titles=True,
            title_fmt=".3f",
            bins=40,
            smooth=1.0,
        )
        fig.subplots_adjust(top=0.97)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print("[SAVE]", out_png, flush=True)

    def collect_global_joint_cloud(model, loader, device, max_points=50000, mode="sample", df=4.0,
                                   mc_train_mode=False, sample_max_scale=2.5,
                                   sample_max_mahalanobis_scale=5.0):
        df_use = float(getattr(model, "student_df", df))
        prev_train = model.training
        model.train() if mc_train_mode else model.eval()
        pts = []
        if hasattr(loader.dataset, "dataset"):
            ds0 = loader.dataset.dataset
        else:
            ds0 = loader.dataset

        def _denorm_t(z):
            y_mean = torch.as_tensor(ds0.Y_mean, device=z.device, dtype=z.dtype).view(1, -1)
            y_std = torch.as_tensor(ds0.Y_std, device=z.device, dtype=z.dtype).view(1, -1)
            return z * y_std + y_mean

        with torch.no_grad():
            collected = 0
            for batch in loader:
                if len(batch) == 4:
                    X, y, y_bl, metas = batch
                else:
                    X, y, metas = batch
                X = X.to(device, non_blocking=(device.type == "cuda"))
                out = model(X, return_kl=True)
                if len(out) >= 5:
                    mu_attr, chol_params, d_rec, kl, extra = out[:5]
                else:
                    mu_attr, chol_params, d_rec, kl = out[:4]
                mu_c = _take_center_from_seq(mu_attr)
                chol_c = _take_center_from_seq(chol_params)
                if mode == "sample":
                    L_c = build_scale_tril_3x3(chol_c.unsqueeze(1)).squeeze(1)
                    z = sample_multivariate_student_t(
                        mu=mu_c,
                        scale_tril=L_c,
                        nu=df_use,
                        n_samples=1,
                        max_scale=float(sample_max_scale),
                        max_mahalanobis_scale=float(sample_max_mahalanobis_scale),
                    ).squeeze(0)
                else:
                    z = mu_c
                pts.append(_denorm_t(z).detach().cpu().numpy())
                collected += int(z.shape[0])
                if collected >= int(max_points):
                    break
        model.train(prev_train)
        if not pts:
            raise RuntimeError("[GLOBAL-JOINT] no points collected")
        cloud = np.concatenate(pts, axis=0).astype(np.float32)
        return cloud[:int(max_points)]

    if cont_loader is not None:
        try:
            max_points = int(getattr(args, "joint_global_n", 50000))
            cloud = collect_global_joint_cloud(
                model=model,
                loader=cont_loader,
                device=device,
                max_points=max_points,
                mode="sample",
                df=float(getattr(model, "student_df", getattr(args, "student_df", 6.0))),
                mc_train_mode=False,
                sample_max_scale=2.5,
                sample_max_mahalanobis_scale=5.0,
            )
            base = os.path.join(out_dir, f"jointGlobal_CONT_GLOBAL_sample_N{int(cloud.shape[0])}")
            np.save(base + "_samples.npy", cloud)
            _save_posterior_stats_txt(cloud, base + "_stats.txt")
            plot_joint_3d_scatter(cloud, base + "_3Dscatter.png", title="Global Joint Cloud (sample)")
            try:
                plot_joint_3d_kde(cloud, base + "_3DKDE.png", title="Global 3D Joint Density (sample)")
            except Exception as e:
                print("[GLOBAL-JOINT][WARN] 3D KDE failed:", repr(e), flush=True)
            plot_joint_corner(cloud, base + "_corner.png")
        except Exception as e:
            print("[GLOBAL-JOINT][WARN] export failed:", repr(e), flush=True)

    # ------------------------------------------------------------------
    # 4) Fig10 ALIGN: keep only muOnly_blinear
    # ------------------------------------------------------------------
    def build_section_indices_for_trace_local(ds, t0: int):
        samples = np.asarray(ds.samples)
        l_arr = samples[:, 0].astype(int)
        t_arr = samples[:, 1].astype(int)
        tau_arr = samples[:, 2].astype(int)
        idxs = np.where(t_arr == int(t0))[0]
        if idxs.size == 0:
            return None, [], []
        l_list = np.unique(l_arr[idxs]); l_list.sort()
        tau_list = np.unique(tau_arr[idxs]); tau_list.sort()
        grid_idx = -np.ones((len(l_list), len(tau_list)), dtype=np.int64)
        l2i = {int(v): i for i, v in enumerate(l_list)}
        tau2i = {int(v): i for i, v in enumerate(tau_list)}
        for gi in idxs:
            grid_idx[l2i[int(l_arr[gi])], tau2i[int(tau_arr[gi])]] = int(gi)
        return grid_idx, l_list.tolist(), tau_list.tolist()

    @torch.no_grad()
    def predict_section_percentiles_fullgrid_for_trace(model, ds, device, t0: int,
                                                       Tmc: int = 80, batch_size: int = 2048,
                                                       temp_scale=None, mc_train_mode: bool = True):
        grid_idx, l_list, tau_list = build_section_indices_for_trace_local(ds, t0)
        if grid_idx is None:
            raise RuntimeError(f"[FIG10] empty section for trace t0={t0}")
        valid_pos = np.argwhere(grid_idx >= 0)
        valid_gi = grid_idx[grid_idx >= 0].astype(int)
        L, Nt = grid_idx.shape
        M = int(valid_gi.shape[0])
        loader = DataLoader(
            Subset(ds, valid_gi.tolist()),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=_collate,
        )
        prev_train = model.training
        model.train() if mc_train_mode else model.eval()
        mc_samples = np.zeros((int(Tmc), M, 3), dtype=np.float32)

        def _denorm(mu_t):
            y_mean = torch.as_tensor(ds.Y_mean, device=mu_t.device, dtype=mu_t.dtype).view(1, -1)
            y_std = torch.as_tensor(ds.Y_std, device=mu_t.device, dtype=mu_t.dtype).view(1, -1)
            return mu_t * y_std + y_mean

        try:
            for k in range(int(Tmc)):
                offset = 0
                for batch in loader:
                    if len(batch) == 4:
                        X, y, y_bl, metas = batch
                    else:
                        X, y, metas = batch
                    X = X.to(device, non_blocking=True)
                    out = model(X, return_kl=True)
                    if len(out) >= 5:
                        mu_attr, chol_params, d_rec, kl, extra = out[:5]
                    else:
                        mu_attr, chol_params, d_rec, kl = out[:4]
                    mu_c = mu_attr[:, mu_attr.shape[1] // 2, :]
                    out_np = _denorm(mu_c).detach().float().cpu().numpy()
                    bsz = out_np.shape[0]
                    mc_samples[k, offset:offset + bsz, :] = out_np
                    offset += bsz
        finally:
            model.train(prev_train)

        q5 = np.percentile(mc_samples, 5, axis=0)
        q50 = np.percentile(mc_samples, 50, axis=0)
        q95 = np.percentile(mc_samples, 95, axis=0)
        p5 = np.full((L, Nt, 3), np.nan, dtype=np.float32)
        p50 = np.full((L, Nt, 3), np.nan, dtype=np.float32)
        p95 = np.full((L, Nt, 3), np.nan, dtype=np.float32)
        for m, (ii, jj) in enumerate(valid_pos):
            p5[ii, jj, :] = q5[m]
            p50[ii, jj, :] = q50[m]
            p95[ii, jj, :] = q95[m]
        return {"t0": int(t0), "l_list": l_list, "tau_list": tau_list, "p5": p5, "p50": p50, "p95": p95,
                "missing": int(np.sum(grid_idx < 0))}

    def plot_section_percentiles_3x3(sec, out_png, dt=0.001, interpolation="bilinear"):
        p5, p50, p95 = sec["p5"], sec["p50"], sec["p95"]
        l_list = sec["l_list"]
        tau_arr = np.asarray(sec["tau_list"], dtype=np.float32)
        t_ms = tau_arr * float(dt) * 1000.0
        extent = [l_list[0], l_list[-1], float(t_ms[-1]), float(t_ms[0])]
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), constrained_layout=True)
        panels = [(p5, "P5"), (p50, "P50"), (p95, "P95")]
        prop_names = [r"$V_p$", r"$V_s$", r"$\rho$"]
        for c, name in enumerate(prop_names):
            stack = np.stack([p5[:, :, c], p50[:, :, c], p95[:, :, c]], axis=0)
            vmin, vmax = (np.nanmin(stack), np.nanmax(stack)) if np.isfinite(stack).any() else (0.0, 1.0)
            for j, (arr, title) in enumerate(panels):
                ax = axes[c, j]
                im = ax.imshow(arr[:, :, c].T, aspect="auto", extent=extent,
                               vmin=vmin, vmax=vmax, interpolation=interpolation)
                ax.set_title(f"{name} {title}")
                ax.set_xlabel("Line")
                ax.set_ylabel("Time (ms)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print("[SAVE]", out_png, flush=True)

    try:
        t_align = 0 if int(ds_all.T) == 1 else int(t0_val)
        sec_mu = predict_section_percentiles_fullgrid_for_trace(
            model, ds_all, device,
            t0=t_align,
            Tmc=int(getattr(args, "fig10_tmc", 80)),
            batch_size=int(getattr(args, "fig10_batch", 2048)),
            temp_scale=temp_scale,
            mc_train_mode=True,
        )
        fn = os.path.join(out_dir, f"fig10_ALIGN_tidx{int(t_align)}_trace_t{int(t_align):03d}_muOnly_blinear.png")
        plot_section_percentiles_3x3(sec_mu, fn, dt=getattr(args, "dt", 0.001), interpolation="bilinear")
        _chk_file(fn)
    except Exception as e:
        print("[FIG10][WARN] ALIGN muOnly_blinear export failed:", repr(e), flush=True)

    # ------------------------------------------------------------------
    # 5) Fig9 full-crop prediction mean and residual section
    # ------------------------------------------------------------------
    try:
        t_fixed = 0
        if not (0 <= t_fixed < ds_all.T):
            raise RuntimeError(f"[FIG9-FULL] t_fixed={t_fixed} out of range [0,{ds_all.T - 1}]")
        l_list_full_idx = list(range(int(ds_all.L)))
        samples0 = np.asarray(ds_all.samples, dtype=int)
        m_t = (samples0[:, 1] == int(t_fixed))
        if int(np.sum(m_t)) == 0:
            raise RuntimeError(f"[FIG9-FULL] no samples for dataset trace t_fixed={t_fixed}")
        tau_list_full = np.unique(samples0[m_t, 2]).astype(int)
        tau_list_full.sort()
        sec = mc_predict_section_mean_fullgrid(
            model, ds_all, device,
            t_fixed=t_fixed,
            l_list_full=l_list_full_idx,
            tau_list_full=tau_list_full,
            Tmc=30,
            temp_scale=temp_scale,
            batch_size=2048,
            collate_fn=_collate,
            mc_train_mode=True,
            ctx_mode="mid3_keep",
            keep_k=3,
        )
        if sec is not None:
            fn = os.path.join(out_dir, "fig9_fullcrop_pred_mean.png")
            plot_section_mean_fig9_fullcrop(
                sec, fn,
                dt=getattr(args, "dt", 0.001),
                prop_names=(r"$V_p$", r"$V_s$", r"$\rho$"),
                prop_units=("m/s", "m/s", "g/cc"),
                fontsize=12,
                interpolation="bilinear",
                dx=10.0,
                x0=11300.0,
                dpi=300,
            )
            _chk_file(fn)

            res_npz = os.path.join(out_dir, f"fig9_fullcrop_residual_t{int(t_fixed)}.npz")
            res_sec, res_metrics = build_residual_section_like_prediction(
                sec, ds_all,
                t0=int(t_fixed),
                pred_key="mean",
                use_bl=False,
                residual_sign="pred_minus_true",
                save_npz=res_npz,
            )
            print("[FIG9-FULL-RES] metrics =", res_metrics, flush=True)
            fn_res = os.path.join(out_dir, f"fig9_fullcrop_residual_t{int(t_fixed)}.png")
            plot_residual_section_fig9_marmousi_3x1(
                res_sec, fn_res,
                dt=getattr(args, "dt", 0.001),
                x0=float(getattr(args, "fig_x0", 11300.0)),
                dx=float(getattr(args, "fig_dx", 10.0)),
                prop_names=(r"$V_p$", r"$V_s$", r"$\rho$"),
                prop_units=("m/s", "m/s", "g/cc"),
                fontsize=13,
                interpolation=getattr(args, "interp", "bilinear"),
                tmin_ms=getattr(args, "fig_tmin_ms", None),
                tmax_ms=getattr(args, "fig_tmax_ms", None),
                cmap="RdBu_r",
                dpi=300,
                qabs=98.0,
            )
            _chk_file(fn_res)
    except Exception as e:
        print("[FIG9-FULL][WARN] full-crop export failed:", repr(e), flush=True)





def _warmup_weight(ep: int, warmup_epochs: int, final_weight: float, mode: str = "linear") -> float:
    """返回当前 epoch 的 warmup 权重 (线性/余弦)"""
    if warmup_epochs is None or warmup_epochs <= 0:
        return float(final_weight)
    if ep <= warmup_epochs:
        if mode == "cos":
            # 余弦升温，更平滑
            return float(final_weight) * 0.5 * (1.0 - math.cos(math.pi * ep / float(warmup_epochs)))
        # 线性升温
        return float(final_weight) * (ep / float(warmup_epochs))
    return float(final_weight)


def build_residual_section_like_prediction(
    sec,
    ds,
    t0=None,
    pred_key="mean",
    use_bl: bool = False,
    residual_sign: str = "pred_minus_true",
    save_npz: str | None = None,
):
    """
    Build residual section between predicted section and true elastic model.

    Parameters
    ----------
    sec : dict
        Prediction section. It should contain either:
          - sec["mean"] with shape [L, Nt, 3], or
          - sec["p50"] with shape [L, Nt, 3].
        It must contain sec["l_list"] and sec["tau_list"].
    ds : VolumeWindowDataset
        Dataset with ds.mod or ds.mod_bl, whose shape is [L, T, 3, N].
    t0 : int
        Fixed trace index. If None, use sec["t0"].
    pred_key : str
        Prediction array key. For CNN mean section use "mean"; for Fig10 use "p50".
    use_bl : bool
        If True, residual is computed against ds.mod_bl; otherwise against ds.mod.
    residual_sign : str
        "pred_minus_true" or "true_minus_pred".
    save_npz : str | None
        If provided, save pred/true/residual and axes into an npz file.

    Returns
    -------
    res_sec : dict
        {"mean": residual, "pred": pred, "true": true, "l_list":..., "tau_list":..., "t0":...}
    metrics : dict
        Per-channel residual statistics in physical units.
    """
    import os
    import json
    import numpy as np

    if sec is None or not isinstance(sec, dict):
        raise RuntimeError("[RESIDUAL] invalid sec")

    if pred_key in sec:
        pred = np.asarray(sec[pred_key], dtype=np.float32)
    elif "mean" in sec:
        pred = np.asarray(sec["mean"], dtype=np.float32)
    elif "p50" in sec:
        pred = np.asarray(sec["p50"], dtype=np.float32)
    else:
        raise RuntimeError("[RESIDUAL] sec must contain 'mean' or 'p50'")

    if pred.ndim != 3 or pred.shape[-1] < 3:
        raise RuntimeError(f"[RESIDUAL] bad pred shape: {pred.shape}; expect [L,Nt,3]")

    l_list = np.asarray(sec.get("l_list", np.arange(pred.shape[0])), dtype=np.int64).reshape(-1)
    tau_list = np.asarray(sec.get("tau_list", np.arange(pred.shape[1])), dtype=np.int64).reshape(-1)

    if len(l_list) != pred.shape[0] or len(tau_list) != pred.shape[1]:
        raise RuntimeError(
            f"[RESIDUAL] axis length mismatch: pred={pred.shape}, "
            f"len(l_list)={len(l_list)}, len(tau_list)={len(tau_list)}"
        )

    if t0 is None:
        if "t0" not in sec:
            raise RuntimeError("[RESIDUAL] t0 is None and sec has no 't0'")
        t0 = int(sec["t0"])
    else:
        t0 = int(t0)

    MOD = ds.mod_bl if (use_bl and hasattr(ds, "mod_bl") and ds.mod_bl is not None) else ds.mod
    if MOD is None:
        raise RuntimeError("[RESIDUAL] ds.mod is None")

    L_full, T_full, C_full, N_full = MOD.shape
    if not (0 <= t0 < T_full):
        raise RuntimeError(f"[RESIDUAL] t0 out of range: t0={t0}, T={T_full}")
    if C_full < 3:
        raise RuntimeError(f"[RESIDUAL] expect 3 props in MOD, got C={C_full}")

    # l_list here should be dataset line indices. If a plotted distance axis is passed by mistake,
    # this explicit check avoids silently indexing wrong locations.
    if np.nanmedian(l_list.astype(float)) > max(5000, L_full * 2):
        raise RuntimeError(
            "[RESIDUAL] l_list looks like Distance(m), not dataset line index. "
            "Use the original section from prediction, before converting x-axis to distance."
        )

    l_idx = l_list[(l_list >= 0) & (l_list < L_full)]
    tau_idx = tau_list[(tau_list >= 0) & (tau_list < N_full)]
    if len(l_idx) != len(l_list) or len(tau_idx) != len(tau_list):
        raise RuntimeError(
            f"[RESIDUAL] clipped axes changed length: l {len(l_list)}->{len(l_idx)}, "
            f"tau {len(tau_list)}->{len(tau_idx)}"
        )

    true_sec = MOD[l_idx, t0, :, :]              # [L, 3, N]
    true_sec = true_sec[:, :, tau_idx]           # [L, 3, Nt]
    true_sec = np.transpose(true_sec, (0, 2, 1)).astype(np.float32)  # [L, Nt, 3]

    pred3 = pred[:, :, :3].astype(np.float32)
    if true_sec.shape != pred3.shape:
        raise RuntimeError(f"[RESIDUAL] pred/true shape mismatch: pred={pred3.shape}, true={true_sec.shape}")

    if residual_sign == "pred_minus_true":
        residual = pred3 - true_sec
    elif residual_sign == "true_minus_pred":
        residual = true_sec - pred3
    else:
        raise ValueError("residual_sign must be 'pred_minus_true' or 'true_minus_pred'")

    labs = ["VP", "VS", "RHOB"]
    metrics = {"sign": residual_sign, "per_channel": {}}
    for i, lab in enumerate(labs):
        r = residual[:, :, i]
        r = r[np.isfinite(r)]
        if r.size == 0:
            metrics["per_channel"][lab] = {"MAE": None, "RMSE": None, "Mean": None, "Std": None, "P95Abs": None}
        else:
            metrics["per_channel"][lab] = {
                "MAE": float(np.mean(np.abs(r))),
                "RMSE": float(np.sqrt(np.mean(r ** 2))),
                "Mean": float(np.mean(r)),
                "Std": float(np.std(r)),
                "P95Abs": float(np.percentile(np.abs(r), 95)),
            }

    res_sec = {
        "t0": int(t0),
        "l_list": l_list.tolist(),
        "tau_list": tau_list.tolist(),
        "mean": residual.astype(np.float32),
        "pred": pred3.astype(np.float32),
        "true": true_sec.astype(np.float32),
        "residual_sign": residual_sign,
        "metrics": metrics,
    }

    if save_npz is not None:
        out_dir = os.path.dirname(os.path.abspath(str(save_npz)))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            save_npz,
            residual=res_sec["mean"],
            pred=res_sec["pred"],
            true=res_sec["true"],
            l_list=np.asarray(l_list),
            tau_list=np.asarray(tau_list),
            t0=np.asarray([int(t0)], dtype=np.int32),
        )
        json_path = str(save_npz).replace(".npz", "_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {save_npz}", flush=True)
        print(f"[SAVE] {json_path}", flush=True)

    return res_sec, metrics


def plot_residual_section_fig9_marmousi_3x1(
    res_sec,
    out_png,
    dt=0.001,
    x0=11300.0,
    dx=10.0,
    prop_names=(r"$V_p$", r"$V_s$", r"$\rho$"),
    prop_units=("m/s", "m/s", "g/cc"),
    fontsize=13,
    interpolation="bilinear",
    tmin_ms=None,
    tmax_ms=None,
    cmap="RdBu_r",
    dpi=300,
    qabs=98.0,
):
    """
    Plot signed residual section with the same 3×1 Fig9-style layout.
    res_sec["mean"] should be residual with shape [L, Nt, 3].
    """
    import os
    import numpy as np
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    if res_sec is None or (not isinstance(res_sec, dict)) or ("mean" not in res_sec):
        print("[RES-PLOT][WARN] invalid residual section.", flush=True)
        return False

    R = np.asarray(res_sec["mean"], dtype=np.float32)
    l_list = np.asarray(res_sec.get("l_list", np.arange(R.shape[0])), dtype=np.float32)
    tau_list = np.asarray(res_sec.get("tau_list", np.arange(R.shape[1])), dtype=np.float32)

    if R.ndim != 3 or R.shape[2] < 3:
        print(f"[RES-PLOT][WARN] bad residual shape={R.shape}", flush=True)
        return False
    if len(l_list) < 2 or len(tau_list) < 2:
        print("[RES-PLOT][WARN] residual section too small.", flush=True)
        return False

    t_ms_all = tau_list * float(dt) * 1000.0
    keep = np.ones_like(t_ms_all, dtype=bool)
    if tmin_ms is not None:
        keep &= (t_ms_all >= float(tmin_ms))
    if tmax_ms is not None:
        keep &= (t_ms_all <= float(tmax_ms))
    if int(keep.sum()) >= 2:
        R_use = R[:, keep, :]
        t_ms = t_ms_all[keep]
    else:
        R_use = R
        t_ms = t_ms_all

    if np.nanmedian(l_list) > 5000:
        x_vals = l_list
    else:
        x_vals = float(x0) + l_list * float(dx)
    extent = [float(x_vals[0]), float(x_vals[-1]), float(t_ms[-1]), float(t_ms[0])]

    def symmetric_limits(img2d, q=98.0):
        x = img2d[np.isfinite(img2d)]
        if x.size < 10:
            vmax = float(np.nanmax(np.abs(img2d))) if np.isfinite(img2d).any() else 1.0
        else:
            vmax = float(np.nanpercentile(np.abs(x), q))
        if (not np.isfinite(vmax)) or vmax <= 0:
            vmax = 1.0
        return -vmax, vmax

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 3,
        "axes.labelsize": fontsize + 1,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    out_png = os.path.abspath(str(out_png))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.2), constrained_layout=True)
    sign_text = res_sec.get("residual_sign", "pred_minus_true")
    title_suffix = "Prediction - True" if sign_text == "pred_minus_true" else "True - Prediction"

    for i, ax in enumerate(axes):
        img = R_use[:, :, i].T
        vmin, vmax = symmetric_limits(img, q=qabs)
        im = ax.imshow(
            img,
            aspect="auto",
            extent=extent,
            origin="upper",
            cmap=cmap,
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{prop_names[i]} residual ({title_suffix})", pad=8)
        ax.set_ylabel("Time (ms)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Distance (m)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.018)
        cb.set_label(prop_units[i], rotation=90, labelpad=7)

    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(out_png):
        print(f"[SAVE] {out_png}", flush=True)
        return True
    print(f"[RES-PLOT][WARN] save failed: {out_png}", flush=True)
    return False

# ====================== Main ======================
def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--mod", required=True)
    ap.add_argument("--angles", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--props",  nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--win", type=int, default=9)
    ap.add_argument("--tstride", type=int, default=1)
    ap.add_argument("--mode", type=str, default="wellhood", choices=["full","spatial","wellhood"])
    ap.add_argument("--line_stride", type=int, default=1)
    ap.add_argument("--trace_stride", type=int, default=1)
    ap.add_argument("--phase_line", type=int, default=0)
    ap.add_argument("--phase_trace", type=int, default=0)

    ap.add_argument("--wells_xlsx", type=str, default="")
    ap.add_argument("--well_radius", type=int, default=4)
    ap.add_argument("--well_loss_weight", type=float, default=5.0)
    ap.add_argument("--far_keep_ratio", type=float, default=0.08)
    ap.add_argument("--well_min_per_batch", type=int, default=8)
    ap.add_argument("--student_df", type=float, default=6.0)
    ap.add_argument("--physics_loss_weight", type=float, default=0.1)
    ap.add_argument("--dominant_freq", type=float, default=45.0)
    ap.add_argument("--contrast_eps", type=float, default=0.01)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    # ---- LR (兼容老参数) ----
    ap.add_argument("--lr_main", type=float, default=None,
                    help="主网络学习率；若不填则用 --lr")
    ap.add_argument("--lr_phys", type=float, default=None,
                    help="物理核学习率；若不填则取 lr_main 的一半")

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--beta_kl", type=float, default=1e-5)
    ap.add_argument("--bayes", type=str, default="reparam", choices=["reparam","flipout"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="bnn_ckpt_physics")

    # wells 初始参数（供 auto calibrate 搜索中心值）
    ap.add_argument("--wells_line_origin", type=float, default=2064.0)
    ap.add_argument("--wells_trace_origin", type=float, default=1677.0)
    ap.add_argument("--wells_line_scale", type=float, default=451.0/(2334.0-2064.0))
    ap.add_argument("--wells_trace_scale", type=float, default=351.0/(1941.0-1677.0))
    ap.add_argument("--wells_line_offset", type=float, default=0.0)
    ap.add_argument("--wells_trace_offset", type=float, default=0.0)

    ap.add_argument("--phys_K", type=int, default=3)
    ap.add_argument("--phys_fmin", type=float, default=10.0)
    ap.add_argument("--phys_fmax", type=float, default=70.0)
    ap.add_argument("--phys_center_avg", type=int, default=3)
    ap.add_argument("--phys_eps", type=float, default=0.02)
    ap.add_argument("--phys_standardize", action="store_true", default=True)

    ap.add_argument("--loow_names", type=str, default="",
                    help="留井验证：用 Excel 的井名选择，多口以分号分隔，如 'WellA;WellB'")
    ap.add_argument("--loow_rows", type=str, default="",
                    help="留井验证：用 Excel 的行号(1-based)选择，多行以分号分隔，如 '2;7;12'")

    ap.add_argument("--lambda_lat", type=float, default=0.02,
                        help="lateral continuity loss weight (line direction)")
    ap.add_argument("--warm_lat_ep", type=int, default=5,
                        help="start epoch for lateral continuity loss")
    ap.add_argument("--lat_max_gap", type=int, default=12)

    ap.add_argument("--line_block", type=int, default=16)  # ✅ 8 -> 16
    ap.add_argument("--line_quota", type=float, default=0.7)  # ✅ 0.5 -> 0.7

    ap.add_argument("--line_ctx", type=int, default=7, help="2.5D: number of neighboring lines (odd: 1/3/5/7)")

    ap.add_argument("--use_posenc", action="store_true", default=True)
    ap.add_argument("--pe_dim", type=int, default=16)
    ap.add_argument("--use_multiscale", action="store_true", default=True)
    ap.add_argument("--ms_scales", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--use_dilated", action="store_true", default=True)
    ap.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--recon", type=str, default="student", choices=["student", "gauss"])
    ap.add_argument("--dt", type=float, default=0.001,
                    help="时间采样间隔(秒)，用于频谱损失/带限卷积，fs=1/dt")
    ap.add_argument(
        "--lambda_vaim",
        type=float,
        default=0.10,  # 和你之前硬编码 0.07 保持一致
        help="weight of VAIM band-limited prior loss"
    )
    ap.add_argument("--warm_vaim_ep", type=int, default=0,
                    help="VAIM warmup epochs (keep lambda_vaim=0 before this epoch)")
    ap.add_argument("--vaim_ramp_len", type=int, default=5,
                    help="ramp length epochs to reach args.lambda_vaim")
    ap.add_argument("--n_comp", type=int, default=3)
    # ---- 低频先验相关参数（额外约束 A 的低频部分）----
    ap.add_argument("--lf_cut", type=float, default=8.0,
                    help="低频先验截止频率(Hz)，默认 8Hz，可按区域低频趋势调整")
    ap.add_argument("--lf_weight", type=float, default=0.10,
                    help="低频先验 loss 权重，默认 0.10，可在 0.05~0.2 间微调")
    # 物理带限先验 + 岩石物理先验的权重
    ap.add_argument(
        "--prior_weight", type=float, default=0.03,
        help="weight for band-limited / low-frequency prior in PhysicsInformedLossLearnable"
    )
    ap.add_argument(
        "--rp_weight", type=float, default=0.02,
        help="weight for rock-physics prior (Vp/Vs range & RHOB floor) in PhysicsInformedLossLearnable"
    )

    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    args = ap.parse_args()
    # 兼容老参：如果没传 lr_main / lr_phys，就从 lr 推导
    if getattr(args, "lr_main", None) is None:
        args.lr_main = args.lr  # 用老的 --lr
    if getattr(args, "lr_phys", None) is None:
        args.lr_phys = max(1e-6, float(args.lr_main) * 0.5)

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    stack4d, key_s = load_4d(args.stack)
    mod4d,   key_m = load_4d(args.mod)

    # 快速体检（含 RHOB）
    quick_stats("Stack(angle0)", stack4d[:,:,0,:])
    quick_stats("Stack(angle1)", stack4d[:,:,1,:])
    quick_stats("Stack(angle2)", stack4d[:,:,2,:])
    quick_stats("Mod(VP)",  mod4d[:,:,0,:])
    quick_stats("Mod(VS)",  mod4d[:,:,1,:])
    quick_stats("Mod(RHOB)",mod4d[:,:,2,:])

    os.makedirs(args.out, exist_ok=True)

    # ---------- A) 读取井表 + 自动标定 +（可选）按井名/行号选择 LOOW（直接替换整段） ----------
    wells = None
    loow_pairs_vol = []  # [(l_idx, t_idx), ...] 体坐标（用于留井验证：可扩展为“井邻域”而非单点）

    def _expand_area_pairs(core_pairs, L, T, R):
        """把 LOOW 井中心点扩展成 (l,t) 邻域集合（去重+clip）"""
        R = int(max(0, R))
        out = set()
        for (l0, t0) in core_pairs:
            l0, t0 = int(l0), int(t0)
            for l in range(l0 - R, l0 + R + 1):
                if not (0 <= l < L):
                    continue
                for t in range(t0 - R, t0 + R + 1):
                    if 0 <= t < T:
                        out.add((int(l), int(t)))
        return sorted(list(out))

    if getattr(args, "wells_xlsx", None):
        # 1) 读工程线/道，做自动标定，得到用于训练/采样的 wells（体坐标）
        wells_raw_all = load_wells_xlsx(args.wells_xlsx, base=1, swap_lt=True)
        L, T = int(stack4d.shape[0]), int(stack4d.shape[1])

        wells, best_params = auto_calibrate_wells(
            wells_raw_all, L, T,
            line_origin=args.wells_line_origin, trace_origin=args.wells_trace_origin,
            line_scale=args.wells_line_scale, trace_scale=args.wells_trace_scale,
            line_offset=args.wells_line_offset, trace_offset=args.wells_trace_offset,
            out_dir=args.out, verbose=True
        )
        print("[AUTO-WELLS] Using calibrated params:", best_params)

        # 2) （可选）从 Excel 第一列“井名”筛选留井；或用行号筛选
        try:
            import pandas as pd
            df = pd.read_excel(args.wells_xlsx, header=None, engine="openpyxl")
        except Exception:
            import pandas as pd
            df = pd.read_excel(args.wells_xlsx, header=None)

        # 约定：第一列=井名，第二列=线号(工程坐标)，第三列=道号(工程坐标)
        df = df.iloc[:, :3].copy()
        df.columns = ["name", "line_eng", "trace_eng"]

        pick_df = None
        loow_names = getattr(args, "loow_names", None)
        loow_rows  = getattr(args, "loow_rows", None)

        if loow_names:
            want = [x.strip() for x in str(loow_names).split(";") if x.strip()]
            pick_df = df[df["name"].astype(str).isin(want)].copy()
            if len(pick_df) == 0:
                print(f"[LOOW][WARN] 井名未匹配到：{want}")

        elif loow_rows:
            try:
                rows = [int(x) for x in str(loow_rows).split(";") if str(x).strip()]
            except Exception:
                rows = []
            rows_1based_in = [r for r in rows if 1 <= r <= len(df)]
            if len(rows_1based_in) < len(rows):
                print("[LOOW][WARN] 有行号越界，已忽略。")
            if rows_1based_in:
                pick_df = df.iloc[[r - 1 for r in rows_1based_in]].copy()

        # 3) 把选中的 LOOW 井（工程线/道）用“最佳参数”映射到体索引
        if pick_df is not None and len(pick_df) > 0:
            wells_raw_loow = pick_df[["line_eng", "trace_eng"]].to_numpy()

            # 注意：训练 wells_raw_all 用 swap_lt=True 读进来（可能发生过交换）
            # LOOW 这里列是 (line_eng, trace_eng)，需要与 best_params 的 swap 逻辑一致：
            swap_for_loow = not bool(best_params.get("swap", False))

            loow_vol = map_wells_to_volume(
                wells_raw_loow, L, T,
                swap=swap_for_loow,
                one_based=bool(best_params.get("one_based", True)),
                line_origin=best_params.get("line_origin", args.wells_line_origin),
                trace_origin=best_params.get("trace_origin", args.wells_trace_origin),
                line_scale=best_params.get("line_scale", args.wells_line_scale),
                trace_scale=best_params.get("trace_scale", args.wells_trace_scale),
                line_offset=best_params.get("line_offset", args.wells_line_offset),
                trace_offset=best_params.get("trace_offset", args.wells_trace_offset),
                round_mode=best_params.get("round_mode", "round"),
                clip=True, verbose=True
            )

            loow_core = [(int(l), int(t)) for (l, t) in loow_vol]
            for nm, (le, te), (lv, tv) in zip(
                    pick_df["name"].to_numpy(),
                    wells_raw_loow,
                    loow_core):
                print(f"[DEBUG] LOOW {nm}: eng(line,trace)=({le},{te}) -> vol(l,t)=({lv},{tv})")

            # ✅ 建议：LOOW 留“井邻域”而不是只留中心点，防止空间泄露
            holdout_R = int(getattr(args, "loow_holdout_radius", args.well_radius))
            loow_pairs_vol = _expand_area_pairs(loow_core, L, T, holdout_R)

            print(f"[LOOW] chosen core wells -> volume indices: {loow_core}")
            print(f"[LOOW] holdout AREA size={len(loow_pairs_vol)} (R={holdout_R})")

        else:
            if loow_names or loow_rows:
                print("[LOOW][WARN] 未从 Excel 选到 LOOW 井；将采用常规分层验证。")

    # ==========================================================
    # ✅ Marmousi2 synthetic：没有真实井表时自动创建 5 个伪井
    #    - Well1 / Well2 / Well4 / Well5 参与训练
    #    - Well3 作为 LOOW 验证井
    #    注意：Marmousi2 转换后的 T 通常为 1，因此 t_idx 固定为 0。
    # ==========================================================
    if (wells is None) or (len(wells) == 0):
        L, T = int(stack4d.shape[0]), int(stack4d.shape[1])

        # 选择 5 个横向位置，避开边界，并覆盖左、中、右区域
        # 对 L=560，大致对应 line = 89, 179, 280, 380, 470
        pseudo_line_fracs = [0.16, 0.32, 0.50, 0.68, 0.84]

        pseudo_wells = np.array([
            [
                int(round(frac * (L - 1))),
                0 if T == 1 else int(round(0.50 * (T - 1)))
            ]
            for frac in pseudo_line_fracs
        ], dtype=np.int32)

        # clip 防止极端尺寸越界
        pseudo_wells[:, 0] = np.clip(pseudo_wells[:, 0], 0, L - 1)
        pseudo_wells[:, 1] = np.clip(pseudo_wells[:, 1], 0, T - 1)

        # 5 个都标记为 well；
        # split 时 Well3 被放入验证集，Well1/Well2/Well4/Well5 留在训练集
        wells = pseudo_wells

        pseudo_train_core = [
            (int(pseudo_wells[0, 0]), int(pseudo_wells[0, 1])),  # Well1
            (int(pseudo_wells[1, 0]), int(pseudo_wells[1, 1])),  # Well2
            (int(pseudo_wells[3, 0]), int(pseudo_wells[3, 1])),  # Well4
            (int(pseudo_wells[4, 0]), int(pseudo_wells[4, 1])),  # Well5
        ]

        pseudo_val_core = [
            (int(pseudo_wells[2, 0]), int(pseudo_wells[2, 1]))   # Well3
        ]

        # 留井验证区域：至少覆盖 line_ctx 的半宽，避免 2.5D patch 泄漏
        holdout_R = max(
            int(getattr(args, "well_radius", 0)),
            int(getattr(args, "line_ctx", 1)) // 2 + 1
        )
        loow_pairs_vol = _expand_area_pairs(pseudo_val_core, L, T, holdout_R)

        # 传给最终绘图函数，优先选择这个验证伪井
        args._pseudo_val_lt = pseudo_val_core[0]

        print(
            f"[PSEUDO-WELL] No wells_xlsx provided. Use 5 pseudo wells: "
            f"{[(int(l), int(t)) for l, t in pseudo_wells]}"
        )
        print(f"[PSEUDO-WELL] train pseudo wells = {pseudo_train_core}")
        print(f"[PSEUDO-WELL] val pseudo well    = {pseudo_val_core[0]}")
        print(f"[PSEUDO-WELL] LOOW holdout area size={len(loow_pairs_vol)} (R={holdout_R})")

    # ==========================================================
    # ✅ Helper: 2.5D 邻线泄漏防护（正确带宽：±hl）
    # ==========================================================
    def apply_25d_guard(ds, train_indices, val_indices):
        """
        2.5D 防泄漏：剔除 train 中会在输入 patch 里“看到” val 的样本
        条件：同 trace(t) 且 |l_train - l_val| <= hl
        """
        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)

        hl = int(getattr(ds, "line_ctx", 1)) // 2
        if hl <= 0 or train_indices.size == 0 or val_indices.size == 0:
            return train_indices

        samples = np.asarray(ds.samples).astype(int)
        L = int(getattr(ds, "L", getattr(ds, "stack", samples) .shape[0] if hasattr(ds, "stack") else samples[:, 0].max() + 1))

        lt_val = samples[val_indices, :2].astype(int)
        val_lt_set = set((int(l), int(t)) for (l, t) in lt_val)

        forbid_lt_set = set()
        for (lv, tv) in val_lt_set:
            lv = int(lv); tv = int(tv)
            for l2 in range(lv - hl, lv + hl + 1):
                if 0 <= l2 < L:
                    forbid_lt_set.add((int(l2), int(tv)))

        lt_train = samples[train_indices, :2].astype(int)
        keep = np.array([(int(l), int(t)) not in forbid_lt_set for (l, t) in lt_train], dtype=bool)

        removed = int((~keep).sum())
        if removed > 0:
            print(f"[SPLIT][25D-GUARD] removed {removed} train samples (hl={hl}) to avoid 2.5D neighbor leakage.")
        return train_indices[keep]

    # ==========================================================
    # ✅ Helper: 强自检（同 (l,t) + 2.5D 邻线）
    # ==========================================================
    def check_no_leak(ds, train_indices, val_indices):
        samples = np.asarray(ds.samples).astype(int)
        hl = int(getattr(ds, "line_ctx", 1)) // 2

        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)

        lt_train = samples[train_indices, :2]
        lt_val = samples[val_indices, :2]

        set_train = set(map(tuple, lt_train.tolist()))
        set_val = set(map(tuple, lt_val.tolist()))

        inter = set_train.intersection(set_val)
        if inter:
            raise RuntimeError(f"[LEAK] train/val share same (l,t)! examples={list(inter)[:5]}")

        if hl > 0:
            val_by_t = {}
            for (l, t) in set_val:
                val_by_t.setdefault(int(t), []).append(int(l))

            bad = []
            for (l, t) in set_train:
                t = int(t); l = int(l)
                if t in val_by_t:
                    ls = np.asarray(val_by_t[t], dtype=int)
                    if ls.size > 0 and np.min(np.abs(ls - l)) <= hl:
                        bad.append((l, t))
                        if len(bad) >= 10:
                            break
            if bad:
                raise RuntimeError(f"[LEAK-25D] train has neighbors within ±hl of val on same trace! examples={bad[:10]}")

        print("[CHK] split leak check passed ✅")

    # ==========================================================
    # ✅ Helper: 更全面的 split 泄漏检查（只用 samples 索引，不碰 __getitem__）
    # ==========================================================
    def check_split_leakage(ds, train_indices, val_indices, line_ctx=1, loow_lt_set=None, max_report=10):
        samples = np.asarray(ds.samples).astype(int)

        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)

        lt_train = samples[train_indices, :2].astype(int)
        lt_val = samples[val_indices, :2].astype(int)

        set_train = set(map(tuple, lt_train.tolist()))
        set_val = set(map(tuple, lt_val.tolist()))

        inter = set_train.intersection(set_val)
        if inter:
            ex = list(inter)[:max_report]
            raise RuntimeError(f"[LEAK][(l,t)] train/val share same (l,t)! examples={ex}")

        if loow_lt_set is not None and len(loow_lt_set) > 0:
            loow_lt_set = set((int(l), int(t)) for (l, t) in loow_lt_set)
            bad_val = [lt for lt in set_val if lt not in loow_lt_set]
            if bad_val:
                raise RuntimeError(f"[LOOW][LEAK] val contains non-loow (l,t)! examples={bad_val[:max_report]}")
            bad_train = [lt for lt in set_train if lt in loow_lt_set]
            if bad_train:
                raise RuntimeError(f"[LOOW][LEAK] train still contains loow (l,t)! examples={bad_train[:max_report]}")

        hl = int(line_ctx) // 2
        if hl > 0:
            val_by_t = {}
            for (l, t) in set_val:
                val_by_t.setdefault(int(t), []).append(int(l))

            bad25 = []
            for (l, t) in set_train:
                t = int(t); l = int(l)
                if t in val_by_t:
                    ls = np.asarray(val_by_t[t], dtype=int)
                    if ls.size > 0 and np.min(np.abs(ls - l)) <= hl:
                        bad25.append((l, t))
                        if len(bad25) >= max_report:
                            break
            if bad25:
                raise RuntimeError(f"[LEAK-25D] train overlaps val neighbor band (±hl) on same trace! examples={bad25}")

        print(f"[CHK-SPLIT] ok ✅  train_samples={len(train_indices)} val_samples={len(val_indices)} "
              f"train_unique_lt={len(set_train)} val_unique_lt={len(set_val)} hl={hl}")

    # ==========================================================
    # ✅ Helper: 标签通道健康检查（TRAIN/VAL 都建议跑）
    # ==========================================================
    def check_mod_channel_health(ds, indices, name="TRAIN", props_idx=(0, 1, 2), max_samples=50000, seed=0):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            print(f"[HEALTH:{name}] empty indices, skip.")
            return

        rng = np.random.default_rng(int(seed))
        take = min(int(max_samples), int(indices.size))
        pick = rng.choice(indices, size=take, replace=False)

        samples = np.asarray(ds.samples).astype(int)
        ys = []
        ybls = []

        has_bl = hasattr(ds, "mod_bl") and (ds.mod_bl is not None)

        for gidx in pick:
            l, t, tau = samples[int(gidx)]
            y = ds.mod[int(l), int(t), list(props_idx), int(tau)].astype(np.float64)
            ys.append(y)
            if has_bl:
                yb = ds.mod_bl[int(l), int(t), list(props_idx), int(tau)].astype(np.float64)
                ybls.append(yb)

        Y = np.stack(ys, 0)
        finite = np.isfinite(Y).all(axis=1)
        fin_rate = float(finite.mean()) * 100.0

        Yf = Y[finite] if finite.any() else Y
        mean = Yf.mean(axis=0)
        std = Yf.std(axis=0)
        mn = Yf.min(axis=0)
        mx = Yf.max(axis=0)

        print(f"[HEALTH:{name}] N={take} finite={fin_rate:.2f}%")
        print(f"[HEALTH:{name}] mean={mean} std={std}")
        print(f"[HEALTH:{name}] min ={mn} max={mx}")

        for i, s in enumerate(std):
            if s < 1e-6:
                print(f"[HEALTH:{name}][WARN] channel[{i}] std≈0 (likely constant/empty).")

        if has_bl and len(ybls) > 0:
            YB = np.stack(ybls, 0)
            finite_b = np.isfinite(YB).all(axis=1)
            fin_rate_b = float(finite_b.mean()) * 100.0
            YBf = YB[finite_b] if finite_b.any() else YB
            mean_b = YBf.mean(axis=0)
            std_b = YBf.std(axis=0)
            print(f"[HEALTH:{name}][BL] finite={fin_rate_b:.2f}% mean={mean_b} std={std_b}")

    # ==========================================================
    # 构建数据集（defer_norm=True：split 后用 train-only 统计）
    # ==========================================================
    ds_kwargs = dict(
        angles_idx=tuple(args.angles), props_idx=tuple(args.props),
        win=args.win, time_stride=args.tstride, mode=args.mode,
        line_stride=args.line_stride, trace_stride=args.trace_stride,
        phase_line=args.phase_line, phase_trace=args.phase_trace,
        wells=wells, well_radius=args.well_radius, far_keep_ratio=args.far_keep_ratio,
        line_ctx=args.line_ctx,
        add_line_diff=True,
    )

    ds_all = VolumeWindowDataset(
        stack4d, mod4d,
        **ds_kwargs,
        defer_norm=True,
        norm_seed=args.seed,
        norm_max_samples=20000,
        norm_scheme="angle_time"
    )
    _need_norm_recompute = True

    n_all = len(ds_all)
    n_well = int(np.asarray(ds_all.is_well_sample).sum())
    print(f"[REPORT] samples={n_all:,}, well_labeled={n_well:,} ({100.0 * n_well / max(1, n_all):.2f}%)")

    # -------- Split: 优先 LOOW（留井/留井邻域），否则按 (l,t) 分组 --------
    all_idx = np.arange(len(ds_all), dtype=np.int64)
    is_well = np.asarray(ds_all.is_well_sample).astype(bool)
    rng = np.random.default_rng(args.seed)

    samples_arr = np.asarray(ds_all.samples).astype(int)  # [N,3] = (l,t,tau)
    lt_arr = samples_arr[:, :2].astype(int)               # [N,2] = (l,t)

    if "loow_pairs_vol" not in locals():
        loow_pairs_vol = []

    loow_lt_set = set((int(l), int(t)) for (l, t) in loow_pairs_vol)
    loow_indices = np.array([], dtype=np.int64)

    if len(loow_lt_set) > 0:
        loow_mask = np.array([(int(l), int(t)) in loow_lt_set for (l, t) in lt_arr], dtype=bool)
        loow_indices = np.where(loow_mask)[0].astype(np.int64)
        print(f"[LOOW] hold-out samples = {len(loow_indices)}  from wells(area) = {sorted(list(loow_lt_set))[:10]}")

    if len(loow_indices) > 0:
        val_indices = np.unique(loow_indices).astype(np.int64)
        train_indices = np.setdiff1d(all_idx, val_indices, assume_unique=False).astype(np.int64)

        train_indices = apply_25d_guard(ds_all, train_indices, val_indices)
        print(f"[SPLIT][LOOW] train={len(train_indices)}  val={len(val_indices)} (VAL=only LOOW area)")
    else:
        uniq_lt, inv = np.unique(lt_arr, axis=0, return_inverse=True)
        gid = np.arange(len(uniq_lt), dtype=np.int64)
        rng.shuffle(gid)

        n_val_g = max(1, int(len(gid) * float(args.val_ratio)))
        val_g = set(gid[:n_val_g].tolist())

        val_mask = np.array([g in val_g for g in inv], dtype=bool)
        val_indices = np.where(val_mask)[0].astype(np.int64)
        train_indices = np.where(~val_mask)[0].astype(np.int64)

        train_indices = apply_25d_guard(ds_all, train_indices, val_indices)
        print(f"[SPLIT][GROUP-(l,t)] train={len(train_indices)}  val={len(val_indices)}  "
              f"(groups={len(uniq_lt)}, val_groups={len(val_g)})")

    # ✅ 强自检：同 (l,t) + 2.5D 邻域
    check_no_leak(ds_all, train_indices, val_indices)

    # ==========================================================
    # ✅ Train-only norm（用 Dataset 自带 recompute_norm_stats，确保 _norm_ready=True）
    #   - 只用 train_indices
    #   - 排除 val 的 (l,t) + 2.5D 邻域（同 trace & |Δl|<=hl）
    # ==========================================================
    if _need_norm_recompute:
        samples = np.asarray(ds_all.samples).astype(int)
        lt_val = samples[np.asarray(val_indices, dtype=np.int64), :2].astype(int)
        exclude_lt = set((int(l), int(t)) for (l, t) in lt_val)

        hl = int(getattr(ds_all, "line_ctx", 1)) // 2
        if hl > 0 and len(exclude_lt) > 0:
            ex2 = set()
            for (lv, tv) in exclude_lt:
                for l2 in range(int(lv) - hl, int(lv) + hl + 1):
                    if 0 <= l2 < ds_all.L:
                        ex2.add((int(l2), int(tv)))
            exclude_lt |= ex2

        ds_all.recompute_norm_stats(
            train_indices=train_indices,
            exclude_lt=exclude_lt,
            max_samples=20000,
            seed=args.seed,
            verbose=True
        )

    # ✅ norm 后再做更全面 split 检查（最稳）
    check_split_leakage(
        ds_all,
        train_indices=train_indices,
        val_indices=val_indices,
        line_ctx=getattr(args, "line_ctx", 1),
        loow_lt_set=loow_lt_set if len(loow_lt_set) > 0 else None,
    )

    # === 3) 构造子集 ===
    train_set = torch.utils.data.Subset(ds_all, train_indices.tolist())
    val_set = torch.utils.data.Subset(ds_all, val_indices.tolist())

    print(f"[SPLIT] train={len(train_set)} (wells={int(is_well[train_indices].sum())}), "
          f"val={len(val_set)} (wells={int(is_well[val_indices].sum())})")

    # 标签通道健康检查（强烈建议）
    check_mod_channel_health(ds_all, train_indices, name="TRAIN", props_idx=tuple(args.props), seed=args.seed)
    check_mod_channel_health(ds_all, val_indices, name="VAL", props_idx=tuple(args.props), seed=args.seed)

    # === 4) Loader（保持你原来的采样器不动）===
    cuda_on = (torch.device(args.device).type == "cuda")

    def compute_trace_grad_weights(ds, subset_indices, k=1.5, eps=1e-6):
        """
        根据纵向一阶差分的平均幅度估计难度；返回与 subset_indices 对齐的一维权重
        """
        import numpy as np
        L, Tn, C, N = ds.stack.shape
        grads = []
        for idx in subset_indices:
            l, t, tau = ds.samples[int(idx)]
            tau_l = max(0, int(tau) - 1)
            tau_r = min(N - 1, int(tau) + 1)

            y_l = ds.mod[int(l), int(t), ds.props_idx, tau_l]
            y_r = ds.mod[int(l), int(t), ds.props_idx, tau_r]
            g = np.abs(y_r - y_l).mean()
            grads.append(g)

        g = np.asarray(grads, dtype=np.float32)
        g = (g - g.min()) / (g.max() - g.min() + eps)
        w = 1.0 + k * g
        return w.astype(np.float32)

    from collections import defaultdict

    def build_trace_index_table(ds, subset_indices):
        table = defaultdict(list)
        for gidx in subset_indices:
            l, t, tau = ds.samples[int(gidx)]
            table[(int(l), int(t))].append(int(gidx))
        for k in list(table.keys()):
            table[k].sort(key=lambda gi: int(ds.samples[gi][2]))
        return dict(table)

    def build_line_index_table(ds, subset_indices):
        """
        key = (t, tau)  -> indices sorted by l
        """
        table = defaultdict(list)
        for gidx in subset_indices:
            l, t, tau = ds.samples[int(gidx)]
            table[(int(t), int(tau))].append(int(gidx))
        for k in table.keys():
            table[k].sort(key=lambda gi: int(ds.samples[gi][0]))
        return dict(table)

    # ✅ 训练子集索引 subset_idx（只做一次）
    subset_idx = np.asarray(train_set.indices, dtype=np.int64)

    # ✅ 非井样本权重表（只做一次）
    w_train = compute_trace_grad_weights(ds_all, subset_idx, k=1.5)
    weight_table = {int(gidx): float(w) for gidx, w in zip(subset_idx, w_train)}

    # ✅ trace_table / line_table（只做一次）
    trace_table = build_trace_index_table(ds_all, subset_idx)
    print(f"[TRACE-TABLE] traces={len(trace_table)} (train subset)")
    line_table = build_line_index_table(ds_all, subset_idx)
    print(
        f"[LINE-TABLE] keys={len(line_table)}  example_key={next(iter(line_table.keys())) if len(line_table) > 0 else None}")

    # ==========================
    # ✅ train_loader：使用 trace block sampling
    # ==========================
    train_loader = DataLoader(
        ds_all,
        batch_sampler=WellBalancedBatchSampler(
            subset=train_set,
            is_well_full=ds_all.is_well_sample,
            batch_size=args.batch,
            min_well=16,
            seed=args.seed,
            drop_last=False,
            nonwell_weight_table=weight_table,

            trace_index_table=trace_table,
            trace_block=12,
            prefer_consecutive=True,

            line_index_table=line_table,
            line_block=args.line_block,
            line_quota=args.line_quota,
            prefer_line_consecutive=True,

            samples_full=ds_all.samples,
            patch_quota=getattr(args, "patch_quota", 0.5),
            patch_line_block=getattr(args, "patch_line_block", 16),
            patch_tau_block=getattr(args, "patch_tau_block", 8),
            patch_tau_stride=getattr(ds_all, "time_stride", 1),
            patch_min_keep=getattr(args, "patch_min_keep", 32),

            debug_first=3,
            debug_every=200,
        ),
        num_workers=0,
        pin_memory=cuda_on,
        collate_fn=_collate,
    )

    # ✅ val_loader：只用 val_set，不 shuffle
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda_on,
        drop_last=False,
        collate_fn=_collate
    )

    # =========================
    # ✅ NEW: continuity eval loader (dense nonwell samples)
    # =========================
    nonwell_all = np.where(~is_well)[0]
    rng = np.random.default_rng(args.seed + 999)
    rng.shuffle(nonwell_all)

    cont_N = min(len(nonwell_all), 200000)   # 20万点一般就很稳
    cont_indices = nonwell_all[:cont_N]
    cont_set = torch.utils.data.Subset(ds_all, cont_indices.tolist())

    cont_loader = DataLoader(
        cont_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda_on,
        drop_last=False,
        collate_fn=_collate
    )
    print(f"[CONT-LOADER] nonwell samples for continuity = {len(cont_set)}")


    # ---- device ----
    device = torch.device(args.device)
    # ---- 可学习物理（用于 physics loss），需要真实角度(°) ----
    # 若你知道真实角度，可以直接在这里改成实际角度；否则用兜底值
    angle_degrees = [5, 15, 25] if len(args.angles) == 3 else [5.0] * len(args.angles)
    print(f"[PHYS] angles(deg)={angle_degrees}")

    # 可学习物理核（包含 freq_logits, alpha, gamma, delta 等可训练参数）
    physics_model = LearnableMultiFreqPhysics(
        angle_degs=angle_degrees,
        K=args.phys_K,
        fmin=args.phys_fmin,
        fmax=args.phys_fmax,
        ricker_len=100,
        dt=args.dt,
        center_avg=args.phys_center_avg,
        eps=args.contrast_eps,
    ).to(device)  # ★ 整块物理模型搬到 device 上

    # 物理一致性损失（带限先验 + 岩石物理先验）
    physics_loss_fn = PhysicsInformedLossLearnable(
        physics_model=physics_model,
        physics_weight=getattr(args, "physics_loss_weight", 0.05),
        prior_weight=getattr(args, "prior_weight", 0.03),
        rp_weight=getattr(args, "rp_weight", 0.02),
        x_mean=ds_all.X_mean,
        x_std=ds_all.X_std,
        standardize=True,
    )

    # 构造完之后（在进入训练循环前）再用命令行参数覆盖一次权重
    physics_loss_fn.physics_weight = float(args.physics_loss_weight)
    base_phys_w = physics_loss_fn.physics_weight

    print(f"[INIT] physics_weight={base_phys_w:.4e}, prior_weight={physics_loss_fn.prior_weight:.4e}")

    # ---- 主网络：Advanced_BNN 作为逆问题网 + ForwardNet 作为 VAIM 正演网 ----
    # ✅ 2.5D 后：in_dim 不再是 A*win，因为 X 不再 flatten

    num_props = len(args.props)  # 3 (VP, VS, RHOB)

    # ✅ 你需要在 argparse 里有 args.line_ctx（比如 7），没有就先默认 1
    line_ctx = getattr(args, "line_ctx", 1)
    A = len(args.angles)
    add_line_diff = bool(getattr(ds_all, "add_line_diff", False))  # ✅ 以 dataset 为准
    in_ch = (2 * A) if add_line_diff else A

    print(f"[NET][SYNC] ds_all.add_line_diff={add_line_diff} -> encoder_in_ch={in_ch} (A={A})")

    cnn_encoder = CNNEncoder2D25D(
        in_ch,  # ✅ 第一个参数用位置传，避免关键字不匹配
        line_ctx=line_ctx,
        win=args.win,
        feat_dim=args.hidden
    ).to(device)
    setattr(args, "add_line_diff", add_line_diff)
    # 2) 小 BNN 头：把 CNN feature -> 属性（不变）
    attr_head = SmallBNNHeadSeqMVT(
        in_dim=args.hidden,
        hidden=args.hidden,
        win=args.win,
        out_dim=num_props,   # 这里还是 3
        bayes=args.bayes,
        n_comp=getattr(args, "n_comp", 3),
    ).to(device)

    # 3) 把 encoder + head 打成一个 AttrBNN，接口兼容 inv_net
    class AttrBNN(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

            # 记录 encoder 期望的输入通道数（Conv2d 的 in_channels）
            try:
                self._enc_in_ch = int(self.encoder.conv1.in_channels)
            except Exception:
                self._enc_in_ch = None

        def forward(self, X, return_kl: bool = True):
            # 防呆：4D 输入时检查通道数是否匹配
            if (self._enc_in_ch is not None) and torch.is_tensor(X) and (X.dim() == 4):
                xin = int(X.size(1))
                if xin != self._enc_in_ch:
                    raise RuntimeError(
                        f"[AttrBNN][SHAPE] Encoder expects in_ch={self._enc_in_ch}, but got X.shape={tuple(X.shape)}. "
                        f"Hint: if you enabled dataset add_line_diff, encoder n_angles should be 2*A; "
                        f"if not, set add_line_diff=False or rebuild the encoder."
                    )

            h = self.encoder(X)  # [B,C,line_ctx,win] -> [B,feat_dim]

            out = self.head(h, return_kl=return_kl)

            # mixture-mean 版：head 可能返回 4 项
            if isinstance(out, (tuple, list)) and len(out) >= 4:
                mu, chol_params, kl, extra = out[:4]
                return mu, chol_params, kl, extra
            elif isinstance(out, (tuple, list)) and len(out) >= 3:
                mu, chol_params, kl = out[:3]
                return mu, chol_params, kl
            else:
                raise RuntimeError(
                    f"[AttrBNN] Unexpected head output type={type(out)} "
                    f"len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
                )

    inv_net = AttrBNN(cnn_encoder, attr_head).to(device)

    # 4) 用物理版 BNN-VAIM wrapper：d_rec 直接来自 physics_core（不变）
    model = BNNVAIMPhysics(
        attr_net=inv_net,
        physics_model=physics_model,
        denormalize_fn=ds_all.denormalize_y,
        n_angles=len(args.angles),
        win=args.win,
    ).to(device)

    if not hasattr(args, "student_df"):
        args.student_df = 6.0

    model.student_df = float(args.student_df)
    print(f"[MVT] unified student_df = {model.student_df:.3f}")
    # --- 温度缩放模块（针对属性通道） ---
    temp_scale = PerChannelTemp(C=num_props).to(device)
    temp_scale.tau_raw.requires_grad_(False)  # 先冻结，后面 warmup 之后再解冻

    # ====== 温度缩放：给 VP/VS 更大的初始 τ，RHOB 稍微小一点 ======
    # 你的 PerChannelTemp 里面是用 softplus(tau_raw) 做的正数约束，
    # 所以这里用 softplus 的反函数来设定初值：
    with torch.no_grad():
        # 目标温度：VP、VS 取 3.0，RHOB 取 1.5（可以以后再调）
        base_tau = torch.tensor([3.0, 3.0, 1.5], device=device)
        # softplus^{-1}(y) = log(exp(y) - 1)
        temp_scale.tau_raw.copy_(torch.log(torch.exp(base_tau) - 1.0))

    print("[TEMP] init tau per-channel =",
          torch.nn.functional.softplus(temp_scale.tau_raw).detach().cpu().numpy())


    # ---- 两个优化器：主网(= inv_net+fwd_net+temp_scale) + 可学习物理 ----
    if not hasattr(args, "lr_phys"):
        args.lr_phys = args.lr_main * 0.3  # 如果命令行没给，就默认 0.3× 主学习率

    optimizer_main = torch.optim.AdamW(
        list(inv_net.parameters()) + list(temp_scale.parameters()),
        lr=args.lr_main, weight_decay=args.weight_decay
    )

    optimizer_phys = torch.optim.AdamW(
        physics_model.parameters(),
        lr=args.lr_phys,
        weight_decay=0.0
    )

    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=args.epochs, eta_min=args.lr_main * 0.01
    )

    # best / early stopping state
    # ----------------------------
    best_val_mse = float("inf")
    best_val_r2 = float("-inf")
    best_epoch = -1
    bad_epochs = 0

    # 判定阈值（避免浮点抖动导致“伪提升”）
    mse_eps = getattr(args, "mse_eps", 1e-6)
    r2_eps = getattr(args, "r2_eps", 1e-4)
    patience = getattr(args, "patience", 20)  # 没有就默认 20

    train_hist = []
    val_hist = []

    print(f"开始训练：device={device}  hidden={args.hidden}")
    print(f"物理损失权重={args.physics_loss_weight}  β_kl={args.beta_kl}")
    print(f"[PHYS] angles(deg)={angle_degrees}")

    # === 3.2 物理核 warm-up 冻结 / 解冻 ===
    warmup_phys = int(getattr(args, "warmup_phys", 10))  # 前 warmup_phys 个 epoch 不启用 physics、不更新物理核
    phys_ramp_len = int(getattr(args, "phys_ramp_len", 5))  # 解冻后 ramp 到 base_phys_w

    # 冻结 physics_core（如果存在）
    if physics_model is not None:
        for p in physics_model.parameters():
            p.requires_grad_(False)
    print(f"[PHYS] freeze physics_core for first {warmup_phys} epochs.")

    # ===== EMA 自动平衡器：只创建一次，跨 epoch 累积 =====
    # ✅ 若你已经把 train 收敛成“只保留 physics/lat/vaim/kl/recon”，
    #    这里 names 就别再放 seq/freq/lowfreq/rho_hf 这类项，避免混乱。
    loss_balancer = EMALossBalancer(
        names=["recon", "kl"],  # ✅ 对齐：你要保留的 EMA 项（只平衡 recon / kl）
        decay=0.99,
        target=1.0,
        clamp_min=0.05,
        clamp_max=20.0,
        warmup_steps=100,
        device=device
    )
    global_step = 0
    print("[EMA] Loss balancer enabled (EMA normalization).")

    phys_norm = PhysEMANormalizer(decay=0.99, device=device)
    loss_shifter = LossShiftEMA(decay=0.99, device=device)

    # ---- 基础权重 ----
    base_phys_w = float(getattr(args, "physics_loss_weight", 0.0))
    base_lambda = float(getattr(args, "well_loss_weight", 0.0))

    # well schedule（沿用你原来的）
    well_warm = int(getattr(args, "well_warm", 3))
    well_cool_start = int(getattr(args, "well_cool_start", 10))
    well_min = float(getattr(args, "well_min", 0.4 * base_lambda))

    # lat / dt / recon / eval_Tmc（用于对齐 train/eval）
    lambda_lat = float(getattr(args, "lambda_lat", 0.02))
    warm_lat_ep = int(getattr(args, "warm_lat_ep", 5))
    lat_max_gap = int(getattr(args, "lat_max_gap", 8))
    lambda_recon = float(getattr(args, "lambda_recon", 0.02))
    eval_Tmc = int(getattr(args, "eval_Tmc", 8))
    dt = float(getattr(args, "dt", 0.001))

    # 温度缩放提前解冻一点，让它参与更多训练
    warmup_temp = max(3, int(args.epochs * 0.3))

    # ====== 循环前：解包出 core_inv（你原逻辑保留，不动）======
    if isinstance(model, torch.nn.DataParallel):
        _net = model.module
    else:
        _net = model
    core_inv = getattr(_net, "inv_net", _net)

    # 循环前先挑一个井位作为默认检查井（你原逻辑保留）
    if hasattr(ds_all, "_wells_set") and len(ds_all._wells_set) > 0:
        l0, t0 = next(iter(ds_all._wells_set))
    else:
        l0, t0 = ds_all.stack.shape[0] // 2, ds_all.stack.shape[1] // 2

    # ================== 训练主循环 ==================
    for ep in range(1, args.epochs + 1):

        # ===============================
        # ✅ VAIM 权重：warmup + ramp（你原逻辑保留）
        # ===============================
        base_vaim = float(getattr(args, "lambda_vaim", 0.0))
        warm_vaim_ep = int(getattr(args, "warm_vaim_ep", 0))
        vaim_ramp_len = int(getattr(args, "vaim_ramp_len", 5))

        if base_vaim <= 0.0:
            cur_lambda_vaim = 0.0
        elif ep <= warm_vaim_ep:
            cur_lambda_vaim = 0.0
        else:
            prog = (ep - warm_vaim_ep) / float(max(1, vaim_ramp_len))
            prog = max(0.0, min(1.0, prog))
            cur_lambda_vaim = base_vaim * prog

        # === 解冻 τ ===
        if (temp_scale is not None) and (ep == warmup_temp):
            temp_scale.tau_raw.requires_grad_(True)
            try:
                tau_now = torch.nn.functional.softplus(temp_scale.tau_raw).detach().cpu().numpy()
            except Exception:
                tau_now = None
            print(f"[TEMP] per-channel tau unfrozen at epoch {ep}, tau = {tau_now}")

        # === 到达 warmup_phys+1 时，解冻物理核 + 重置 phys_norm ===
        if (physics_model is not None) and (ep == warmup_phys + 1):
            for p in physics_model.parameters():
                p.requires_grad_(True)
            print(f"[PHYS] unfreeze physics_core at epoch {ep}.")

            if phys_norm is not None:
                phys_norm.inited = False
                phys_norm._ema_misfit = None
                phys_norm._ema_prior = None
                print("[PHYS] reset PhysEMANormalizer at physics enable epoch.")

        # --- physics 权重：warmup=0；解冻后 ramp 到 base_phys_w ---
        if physics_loss_fn is not None:
            if ep <= warmup_phys:
                # 例如给一个小权重，而不是 0
                physics_loss_fn.physics_weight = base_phys_w * 0.2
            else:
                prog = (ep - warmup_phys) / float(max(1, phys_ramp_len))  # ep=warmup+1 -> 1/ramp
                prog = max(0.0, min(1.0, prog))
                physics_loss_fn.physics_weight = base_phys_w * prog

        # --- 井项：升温 → 线性降到 well_min ---
        if ep <= well_warm:
            cur_lambda = base_lambda * (ep / float(max(1, well_warm)))
        elif ep >= well_cool_start:
            span = max(1, args.epochs - well_cool_start)
            frac = min(1.0, (ep - well_cool_start) / float(span))
            cur_lambda = base_lambda - (base_lambda - well_min) * frac
        else:
            cur_lambda = base_lambda

        # === 本 epoch 是否更新物理核 ===
        w_phys_now = float(getattr(physics_loss_fn, "physics_weight", 0.0)) if physics_loss_fn is not None else 0.0
        if (ep <= warmup_phys) or (physics_model is None) or (w_phys_now <= 0.0):
            optimizer_phys_ep = None
        else:
            optimizer_phys_ep = optimizer_phys

        # --------- 1) 训练（对齐你的对齐版 train：physics/lat/vaim/kl/recon + loss_balancer）---------
        tr = train_epoch_vaim(
            model, train_loader,
            optimizer_main, optimizer_phys_ep,
            args.beta_kl, device,
            physics_loss_fn=physics_loss_fn,
            phys_norm=phys_norm,
            lambda_well=cur_lambda,  # ✅ 对齐：本 epoch 的井权重
            denormalize_fn=ds_all.denormalize_y,
            kl_scheduler=cyclical_beta,
            epoch_idx=ep,
            temp_scale=temp_scale,
            auto_w_phys=False,
            # ✅ 对齐关键超参
            lambda_lat=lambda_lat,
            warm_lat_ep=warm_lat_ep,
            lat_max_gap=lat_max_gap,
            dt=dt,

            fdom_vaim=float(getattr(args, "dominant_freq", 45.0)),
            lambda_vaim=cur_lambda_vaim,
            lambda_recon=lambda_recon,

            # ✅ 只平衡你保留的辅项（recon/kl）
            loss_balancer=loss_balancer,
            global_step_start=global_step,

            # debug
            warmup_phys=warmup_phys,
            phys_debug_epochs=(warmup_phys, warmup_phys + 1, warmup_phys + 2),
            print_ema_every=200,
            grad_debug_every=200
        )
        global_step += len(train_loader)

        # --------- 2) 验证（对齐你的对齐版 eval：传 epoch_idx + lat/vaim/dt 等）---------
        va = eval_epoch_vaim(
            model, val_loader, device,
            physics_loss_fn=physics_loss_fn,
            denormalize_fn=ds_all.denormalize_y,
            beta_kl=args.beta_kl,
            temp_scale=temp_scale,
            lambda_recon=lambda_recon,
            phys_norm=phys_norm,
            y_mean_t=ds_all.Y_mean,
            y_std_t=ds_all.Y_std,
            eval_Tmc=eval_Tmc,
            epoch_idx=ep,
            lambda_lat=lambda_lat,
            warm_lat_ep=warm_lat_ep,
            lat_max_gap=lat_max_gap,
            dt=dt,
            lambda_vaim=cur_lambda_vaim,

            # ✅ 关键：把 well 打开
            lambda_well=float(cur_lambda),
            desc="Validation",
        )

        # ==========================================================
        # CONT dense non-well validation
        # mode=well 且 far_keep_ratio=0.0 时，cont_loader 可能为空，需要跳过
        # ==========================================================
        do_cont_eval = (
                cont_loader is not None
                and hasattr(cont_loader, "dataset")
                and len(cont_loader.dataset) > 0
        )

        if do_cont_eval:
            va_cont = eval_epoch_vaim(
                model, cont_loader, device,
                physics_loss_fn=physics_loss_fn,
                denormalize_fn=ds_all.denormalize_y,
                beta_kl=args.beta_kl,
                temp_scale=temp_scale,
                lambda_recon=lambda_recon,
                phys_norm=phys_norm,
                y_mean_t=ds_all.Y_mean,
                y_std_t=ds_all.Y_std,
                lambda_well=0.0,

                eval_Tmc=eval_Tmc,
                epoch_idx=ep,
                lambda_lat=lambda_lat,
                warm_lat_ep=warm_lat_ep,
                lat_max_gap=lat_max_gap,
                dt=dt,
                lambda_vaim=cur_lambda_vaim,
                desc="CONT"
            )
            print("[DBG] va_cont keys =", sorted(list(va_cont.keys()))[:50])

        else:
            print("[CONT][SKIP] cont_loader is empty. This is expected when mode='well' and far_keep_ratio=0.0.",
                  flush=True)

            va_cont = {
                "loss": float("nan"),
                "cont_trace_pairs": 0,
                "cont_line_pairs": 0,
                "cont_trace_ncc": float("nan"),
                "cont_line_ncc": float("nan"),
                "cont_trace_l1": float("nan"),
                "cont_line_l1": float("nan"),
            }
        tp = va_cont.get("cont_trace_pairs", va_cont.get("cont_trace", {}).get("pairs", 0))
        lp = va_cont.get("cont_line_pairs", va_cont.get("cont_line", {}).get("pairs", 0))

        tn = va_cont.get("cont_trace_ncc", va_cont.get("cont_trace", {}).get("mean_ncc", float("nan")))
        tl = va_cont.get("cont_trace_l1", va_cont.get("cont_trace", {}).get("mean_l1", float("nan")))
        ln = va_cont.get("cont_line_ncc", va_cont.get("cont_line", {}).get("mean_ncc", float("nan")))
        ll = va_cont.get("cont_line_l1", va_cont.get("cont_line", {}).get("mean_l1", float("nan")))

        print(
            f"[CONT@NONWELL] trace_pairs={tp} line_pairs={lp} | ncc(trace/line)={tn:.4f}/{ln:.4f} l1(trace/line)={tl:.3f}/{ll:.3f}")

        # --------- 3) LR 调度 ---------
        scheduler_main.step()
        # 物理核的 lr 只有在开始更新之后才衰减
        if ep > warmup_phys and ep >= 10 and (ep % 2 == 0):
            for pg in optimizer_phys.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, args.lr_phys * 0.01)

        lr_main = optimizer_main.param_groups[0]["lr"]
        lr_phys = optimizer_phys.param_groups[0]["lr"]
        tr["_lr_main"] = lr_main
        tr["_lr_phys"] = lr_phys

        train_hist.append(tr)
        val_hist.append(va)

        # === 记录当前 per-channel τ（对齐：eval/train 都可能返回 _tau；这里仍以 temp_scale 为准）===
        try:
            with torch.no_grad():
                tau_now = torch.nn.functional.softplus(temp_scale.tau_raw).detach().cpu().numpy() + 1.0
            tr["tau_vp"] = float(tau_now[0]) if tau_now.shape[0] > 0 else float("nan")
            tr["tau_vs"] = float(tau_now[1]) if tau_now.shape[0] > 1 else float("nan")
            tr["tau_rhob"] = float(tau_now[2]) if tau_now.shape[0] > 2 else float("nan")
        except Exception:
            tr["tau_vp"] = tr["tau_vs"] = tr["tau_rhob"] = float("nan")

        # === physics 当前权重 ===
        try:
            w_phys_now = float(getattr(physics_loss_fn, "physics_weight", 0.0)) if physics_loss_fn is not None else 0.0
        except Exception:
            w_phys_now = 0.0

        # === safe getter（防 KeyError + 自动兜底 mse_for_log）===
        def _g(d, k, default=0.0):
            try:
                v = d.get(k, default)
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        # =========================
        # 取指标（补齐版）
        # =========================

        # train 指标（与你 train 返回对齐）
        tr_loss = _g(tr, "loss")
        tr_mse = _g(tr, "mse", _g(tr, "mse_for_log", float("nan")))
        tr_attr = _g(tr, "attr_loss", float("nan"))
        tr_vaim = _g(tr, "vaim_loss", float("nan"))
        tr_recon = _g(tr, "recon_term", 0.0)
        tr_klw = _g(tr, "kl_w", 0.0)
        tr_phys = _g(tr, "physics_loss")
        tr_lat = _g(tr, "lat")
        tr_lat_pairs = int(tr.get("lat_pairs", 0) or 0)

        # well（如果你 train 里返回的是 well_term 更合理就优先）
        well_tr_raw = _g(tr, "well_loss", 0.0)
        well_tr_term = _g(tr, "well_term", 0.0)

        # val 指标（与你 eval 返回对齐）
        va_loss = _g(va, "loss")
        va_mse = _g(va, "mse", _g(va, "mse_for_log", float("nan")))
        va_attr = _g(va, "attr_loss", float("nan"))
        va_vaim = _g(va, "vaim_loss", float("nan"))
        va_recon = _g(va, "recon_term", 0.0)
        va_klw = _g(va, "kl_w", 0.0)
        va_phys = _g(va, "physics_loss")
        va_lat = _g(va, "lat")
        va_lat_pairs = int(va.get("lat_pairs", 0) or 0)
        va_r2 = _g(va, "r2", float("nan"))

        well_va_raw = _g(va, "well_loss", 0.0)
        well_va_term = _g(va, "well_term", 0.0)

        # cont（dense nonwell）
        cont_loss = _g(va_cont, "loss")
        cont_trace_pairs = int(va_cont.get("cont_trace_pairs", 0) or 0)
        cont_line_pairs = int(va_cont.get("cont_line_pairs", 0) or 0)
        cont_trace_ncc = _g(va_cont, "cont_trace_ncc", float("nan"))
        cont_line_ncc = _g(va_cont, "cont_line_ncc", float("nan"))
        cont_trace_l1 = _g(va_cont, "cont_trace_l1", float("nan"))
        cont_line_l1 = _g(va_cont, "cont_line_l1", float("nan"))

        # =========================
        # 打印（补齐版）
        # =========================
        print(
            f"[CHK-EPOCH] ep={ep} "
            f"w_phys={w_phys_now:.3e} "
            f"lambda_well={float(cur_lambda):.3e} lambda_vaim={float(cur_lambda_vaim):.3e} "
            f"lambda_lat={float(lambda_lat):.3e} warm_lat_ep={int(warm_lat_ep)}",
            flush=True
        )

        print(
            f"[CONT@NONWELL] loss={cont_loss:.4f} "
            f"trace_pairs={cont_trace_pairs} line_pairs={cont_line_pairs} | "
            f"ncc(trace/line)={cont_trace_ncc:.4f}/{cont_line_ncc:.4f} "
            f"l1(trace/line)={cont_trace_l1:.3f}/{cont_line_l1:.3f}",
            flush=True
        )

        print(
            f"[{ep:03d}/{args.epochs}] "
            f"lat_tr={tr_lat:.4f} pairs_tr={tr_lat_pairs} | "
            f"w_phys={w_phys_now:.3e} "
            f"obj_tr={tr_loss:.4f} mse_tr={tr_mse:.4f} "
            f"attr_tr={tr_attr:.4f} vaim_tr={tr_vaim:.4f} recon_tr={tr_recon:.4f} klw_tr={tr_klw:.4f} "
            f"phys_tr={tr_phys:.4f} well_tr={well_tr_term:.4f} | "
            f"obj_va={va_loss:.4f} mse_va={va_mse:.4f} "
            f"attr_va={va_attr:.4f} vaim_va={va_vaim:.4f} recon_va={va_recon:.4f} klw_va={va_klw:.4f} "
            f"phys_va={va_phys:.4f} well_va={well_va_term:.4f} "
            f"lat_va={va_lat:.4f} pairs_va={va_lat_pairs} r2={va_r2:.4f} | "
            f"lr_main={lr_main:.2e} lr_phys={lr_phys:.2e} "
            f"tau=[{tr.get('tau_vp', float('nan')):.3f},{tr.get('tau_vs', float('nan')):.3f},{tr.get('tau_rhob', float('nan')):.3f}]",
            flush=True
        )

        # ✅ Early stopping / best selection: val_mse + val_r2
        #   规则：mse 明显更小 -> better
        #        mse 近似持平 -> r2 更大 -> better
        # --------------------------------------------
        val_mse = float(va.get("mse", 1e9))
        val_r2 = float(va.get("r2", float("-inf")))

        improved = False
        if val_mse < best_val_mse - mse_eps:
            improved = True
        elif abs(val_mse - best_val_mse) <= mse_eps and (val_r2 > best_val_r2 + r2_eps):
            improved = True

        if improved:
            best_val_mse = val_mse
            best_val_r2 = val_r2
            best_epoch = ep
            bad_epochs = 0

            os.makedirs(args.out, exist_ok=True)

            try:
                angle_degrees_ = list(angle_degrees)
            except Exception:
                angle_degrees_ = [5, 20, 35] if len(args.angles) == 3 else [15] * len(args.angles)

            is_learnable_phys = isinstance(physics_model, torch.nn.Module)

            if isinstance(model, torch.nn.DataParallel):
                wrapper = model.module
            else:
                wrapper = model

            inv_net = getattr(wrapper, "inv_net", wrapper)
            fwd_net = getattr(wrapper, "fwd_net", None)

            inv_state = inv_net.state_dict()
            fwd_state = fwd_net.state_dict() if fwd_net is not None else None

            ckpt = {
                "model_state": model.state_dict(),
                "inv_state": inv_state,
                "fwd_state": fwd_state,
                "optimizer_main_state": optimizer_main.state_dict(),
                "optimizer_phys_state": optimizer_phys.state_dict() if optimizer_phys is not None else None,
                "scheduler_state": scheduler_main.state_dict(),

                "input_shape": (len(args.angles), getattr(args, "line_ctx", 1), args.win),
                "hidden": args.hidden,
                "layers": args.num_layers,
                "bayes": args.bayes,
                "angles_idx": list(args.angles),
                "props_idx": list(args.props),
                "win": int(args.win),

                "X_mean_supervised": ds_all.X_mean,
                "X_std_supervised": ds_all.X_std,
                "Y_mean": ds_all.Y_mean,
                "Y_std": ds_all.Y_std,

                "temp_state": temp_scale.state_dict() if temp_scale is not None else None,

                "epoch": ep,
                "best_val_mse": float(best_val_mse),
                "best_val_r2": float(best_val_r2),
                "train_history": train_hist,
                "val_history": val_hist,

                "physics": {
                    "learnable": is_learnable_phys,
                    "angle_degrees": angle_degrees_,
                    "K": getattr(args, "phys_K", 3),
                    "fmin": getattr(args, "phys_fmin", 10.0),
                    "fmax": getattr(args, "phys_fmax", 70.0),
                    "center_avg": getattr(args, "phys_center_avg", 3),
                    "eps": getattr(args, "phys_eps", 0.01),
                },
                "physics_state": physics_model.state_dict() if is_learnable_phys else None,

                "physics_config": {
                    "type": "learnable_multi_freq" if is_learnable_phys else "fixed",
                    "angles": list(args.angles),
                    "angle_degrees": angle_degrees_,
                    "K": getattr(args, "phys_K", 3),
                    "fmin": getattr(args, "phys_fmin", 10.0),
                    "fmax": getattr(args, "phys_fmax", 70.0),
                    "ricker_len": 100,
                    "dt": float(args.dt),
                    "center_avg": getattr(args, "phys_center_avg", 3),
                    "eps": getattr(args, "phys_eps", 0.01),
                    "physics_weight": args.physics_loss_weight,
                },
            }

            torch.save(ckpt, os.path.join(args.out, "best_bnn_with_physics.pt"))

            with open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "best_val_mse": float(best_val_mse),
                    "best_val_r2": float(best_val_r2),
                    "best_epoch": int(best_epoch),
                    "final_epoch": int(ep),
                    "angles_idx": args.angles,
                    "props_idx": args.props,
                    "win": args.win,
                    "model": {
                        "hidden": args.hidden,
                        "layers": args.num_layers,
                        "bayes": args.bayes
                    },
                    "physics": {
                        "weight": float(
                            getattr(physics_loss_fn, "physics_weight", 0.0)) if physics_loss_fn is not None else 0.0,
                        "config": {
                            "learnable": is_learnable_phys,
                            "angle_degrees": angle_degrees_,
                            "K": args.phys_K,
                            "fmin": args.phys_fmin,
                            "fmax": args.phys_fmax,
                            "center_avg": args.phys_center_avg,
                            "eps": args.phys_eps
                        }
                    }
                }, f, ensure_ascii=False, indent=2)

        else:
            bad_epochs += 1

        # ==========================================================
        # ✅ last.pt 每个 epoch 都保存一次
        # 注意：这里和 if improved / else 同级，仍然在 for ep 循环里面
        # ==========================================================
        os.makedirs(args.out, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": int(ep),
                "best_val_mse": float(best_val_mse),
                "best_val_r2": float(best_val_r2),
                "best_epoch": int(best_epoch),
                "train_history": train_hist,
                "val_history": val_hist,
            },
            os.path.join(args.out, "last.pt")
        )

        if bad_epochs >= patience:
            print(f"[EARLY STOP] best_mse={best_val_mse:.6f}, best_r2={best_val_r2:.4f} @ ep={best_epoch}")
            break

    # ==========================================================
    # ✅ 训练循环结束后，再统一做最终绘图和评估
    # 注意：下面这些代码必须和 for ep 对齐，不能缩进在 for ep 里面
    # ==========================================================
    print(f"训练完成。最佳验证：MSE={best_val_mse:.6f}, R2={best_val_r2:.4f} (epoch={best_epoch})")

    plot_training_curves(train_hist, val_hist, args.out, beta_kl=args.beta_kl)

    # === 训练结束后，只出一次 vres/spectra/ncc ===
    if isinstance(model, torch.nn.DataParallel):
        wrapper = model.module
    else:
        wrapper = model

    core_inv = getattr(wrapper, "inv_net", wrapper)

    if len(ds_all._wells_set) > 0:
        l_test, t_test = next(iter(ds_all._wells_set))
    else:
        L, T = ds_all.stack.shape[:2]
        l_test, t_test = L // 2, T // 2

    run_eval_and_plots(
        model, ds_all, val_loader, device, args,
        temp_scale=temp_scale,
        train_indices=train_set.indices,
        val_indices=val_set.indices,
        cont_loader=cont_loader,
    )

if __name__ == "__main__":
    import sys, torch

    sys.argv = [
        "bnn_vaim_marmousi_well.py",

        # Marmousi2 red-box synthetic data
        "--stack", "/data/cv/marmousi/marmousi2_cropped/marmousi2_crop_stack4d_35hz.mat",
        "--mod", "/data/cv/marmousi/marmousi2_cropped/marmousi2_crop_mod4d.mat",

        "--angles", "0", "1", "2",
        "--win", "29",

        # Synthetic data: 5口伪井，4训1验
        "--mode", "wellhood",
        "--wells_xlsx", "",

        # 井约束略增强，但不回到过强的 1.5 / 24
        "--well_radius", "8",
        "--well_loss_weight", "1.35",
        "--well_min_per_batch", "20",
        "--far_keep_ratio", "0.16",

        # physics: 提高物理项，有助于 RHOB 约束
        "--physics_loss_weight", "0.15",
        "--contrast_eps", "0.02",
        "--dominant_freq", "35.0",
        "--dt", "0.001",

        # 训练轮数拉长，让 RHOB 有机会继续收敛
        "--epochs", "80",
        "--batch", "128",
        "--lr", "2.5e-4",

        "--hidden", "256",

        # KL 略减弱，优先提高预测均值精度
        "--beta_kl", "5e-5",
        "--bayes", "reparam",

        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--out", "/data/cv/marmousi/bnn_ckpt_marmousi_well_residual",

        "--num_heads", "8",
        "--num_layers", "12",

        # 正则略减弱，避免 RHOB 欠拟合
        "--dropout", "0.02",
        "--weight_decay", "0.006",
        "--val_ratio", "0.12",

        "--recon", "student",
        "--student_df", "6.0",

        # 低频趋势略增强，主要帮助 RHOB 的背景趋势
        "--lf_cut", "8.0",
        "--lambda_vaim", "0.0",
        "--lf_weight", "0.08",

        # 横向连续性略增强，但不要太大
        "--lambda_lat", "0.25",
        "--warm_lat_ep", "5",
        "--lat_max_gap", "12",

        "--line_block", "16",
        "--line_quota", "0.30",
        "--line_ctx", "11",
    ]
    main()

