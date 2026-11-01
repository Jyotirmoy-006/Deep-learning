"""
src/models/vae_trainer.py
Iteration 6 Repaired VAE Training with Anti-Collapse Schedule and Health Diagnostics.
Features:
- Linear KL Warm-up (annealing) and Free-Bits schedule.
- Per-epoch recon loss and per-dimension KL tracking.
- Quantitative Health Criteria:
  1. Per-dim KL >= 0.05
  2. Decoder output variation over z at fixed y (ratio to real std)
  3. Wasserstein distance per feature
  4. Real-vs-synthetic discriminator AUC
- Two baseline augmenters: Class-conditional Gaussian and Within-class Interpolation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.models import SyntheticDataVAE, init_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Anti-Collapse Loss with Free-Bits
# =====================================================================

def vae_loss_with_free_bits(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
    """
    Computes VAE loss with per-dimension free-bits and beta annealing.
    Reconstruction loss: MSE
    Per-dim KL: -0.5 * mean(1 + logvar_d - mu_d^2 - exp(logvar_d))
    Effective KL: sum(max(free_bits, KL_d))
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    
    # Per-dimension KL divergence (batch-averaged)
    # mu: (batch, latent_dim), logvar: (batch, latent_dim)
    per_dim_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    per_dim_kl_list = [float(k.item()) for k in per_dim_kl]

    # Free bits thresholding per dimension
    free_bits_t = torch.tensor(free_bits, device=x.device)
    effective_kl = torch.sum(torch.maximum(per_dim_kl, free_bits_t))

    total_loss = recon_loss + beta * effective_kl
    raw_total_kl = torch.sum(per_dim_kl)

    return total_loss, recon_loss, raw_total_kl, per_dim_kl_list


# =====================================================================
# Repaired VAE Trainer
# =====================================================================

def train_repaired_vae(
    feats_tr: np.ndarray,
    y_tr: np.ndarray,
    latent_dim: int = 4,
    hidden_dim: int = 32,
    epochs: int = 30,
    kl_warmup_epochs: int = 15,
    free_bits: float = 0.10,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42,
    device: torch.device = DEVICE,
) -> Tuple[SyntheticDataVAE, Dict[str, Any]]:
    """
    Trains class-conditional VAE on train split with KL warmup and free bits.
    Logs per-epoch reconstruction and per-dim KL metrics.
    """
    torch.manual_seed(seed)
    vae = SyntheticDataVAE(
        input_dim=feats_tr.shape[1],
        num_classes=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    init_weights(vae, method="xavier_normal")

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(feats_tr).float(), torch.from_numpy(y_tr).long())
    ld = DataLoader(ds, batch_size=batch_size, shuffle=True)

    epoch_logs = []

    vae.train()
    for ep in range(1, epochs + 1):
        # Linear beta warmup from 0.0 to 1.0
        beta = min(1.0, ep / max(1, kl_warmup_epochs))

        ep_recon = 0.0
        ep_kl = 0.0
        ep_per_dim = np.zeros(latent_dim)
        n_batches = 0

        for xb, yb in ld:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            recon_x, mu, logvar = vae(xb, yb)
            loss, recon, kl, dim_kls = vae_loss_with_free_bits(
                recon_x, xb, mu, logvar, beta=beta, free_bits=free_bits
            )
            loss.backward()
            optimizer.step()

            ep_recon += float(recon.item())
            ep_kl += float(kl.item())
            ep_per_dim += np.array(dim_kls)
            n_batches += 1

        ep_recon /= max(1, n_batches)
        ep_kl /= max(1, n_batches)
        ep_per_dim /= max(1, n_batches)

        epoch_logs.append({
            "epoch": ep,
            "beta": round(beta, 4),
            "recon_loss": round(ep_recon, 6),
            "total_kl": round(ep_kl, 6),
            "per_dim_kl": [round(float(k), 6) for k in ep_per_dim],
        })

    return vae, {"epoch_logs": epoch_logs, "final_beta": beta}


# =====================================================================
# VAE Health Diagnostics
# =====================================================================

def evaluate_vae_health(
    vae: SyntheticDataVAE,
    feats_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    num_samples: int = 500,
    seed: int = 42,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """
    Evaluates quantitative VAE health criteria:
    1. Per-dimension KL divergence
    2. Decoder output variation over z at fixed class y relative to real within-class std
    3. Per-class per-feature Wasserstein distance and std ratio
    4. Held-out real vs synthetic discriminator AUC
    """
    vae.eval()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # 1. Per-dimension KL on training data
    with torch.no_grad():
        x_t = torch.from_numpy(feats_tr).float().to(device)
        y_t = torch.from_numpy(y_tr).long().to(device)
        mu, logvar = vae.encode(x_t, y_t)
        per_dim_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0).cpu().numpy()

    per_dim_kl_table = [
        {
            "latent_dim": d,
            "kl_divergence": round(float(per_dim_kl[d]), 6),
            "threshold": 0.05,
            "posterior_collapsed": bool(per_dim_kl[d] < 0.05),
        }
        for d in range(vae.latent_dim)
    ]
    collapsed_count = sum(1 for r in per_dim_kl_table if r["posterior_collapsed"])

    # 2. Decoder output variation over z at fixed y
    with torch.no_grad():
        # Fixed y=0
        z_sample = torch.randn(num_samples, vae.latent_dim, device=device)
        y0_t = torch.zeros(num_samples, dtype=torch.long, device=device)
        syn_y0 = vae.decode(z_sample, y0_t).cpu().numpy()

        # Fixed y=1
        y1_t = torch.ones(num_samples, dtype=torch.long, device=device)
        syn_y1 = vae.decode(z_sample, y1_t).cpu().numpy()

    real_y0 = feats_tr[y_tr == 0]
    real_y1 = feats_tr[y_tr == 1]

    decoder_variation_table = []
    wasserstein_table = []

    for j, name in enumerate(feature_names):
        std_dec_y0 = float(np.std(syn_y0[:, j]))
        std_dec_y1 = float(np.std(syn_y1[:, j]))
        std_real_y0 = float(np.std(real_y0[:, j]))
        std_real_y1 = float(np.std(real_y1[:, j]))

        ratio_y0 = std_dec_y0 / max(1e-6, std_real_y0)
        ratio_y1 = std_dec_y1 / max(1e-6, std_real_y1)

        w_y0 = float(scipy.stats.wasserstein_distance(real_y0[:, j], syn_y0[:, j]))
        w_y1 = float(scipy.stats.wasserstein_distance(real_y1[:, j], syn_y1[:, j]))

        decoder_variation_table.append({
            "feature": name,
            "std_syn_y0": round(std_dec_y0, 4),
            "std_real_y0": round(std_real_y0, 4),
            "ratio_y0": round(ratio_y0, 4),
            "std_syn_y1": round(std_dec_y1, 4),
            "std_real_y1": round(std_real_y1, 4),
            "ratio_y1": round(ratio_y1, 4),
            "health_ratio_pass": bool(ratio_y0 >= 0.15 and ratio_y1 >= 0.15),
        })

        wasserstein_table.append({
            "feature": name,
            "wasserstein_y0": round(w_y0, 4),
            "wasserstein_y1": round(w_y1, 4),
            "threshold": 0.20,
            "compliant": bool(w_y0 <= 0.20 and w_y1 <= 0.20),
        })

    # 4. Held-out real vs synthetic discriminator AUC
    half_real = len(feats_tr) // 2
    tr_disc_real = feats_tr[:half_real]
    te_disc_real = feats_tr[half_real:]
    n_test_real = len(te_disc_real)

    with torch.no_grad():
        z_disc_tr = torch.randn(half_real, vae.latent_dim, device=device)
        y_disc_tr_lbl = torch.randint(0, 2, (half_real,), dtype=torch.long, device=device)
        syn_disc_tr = vae.decode(z_disc_tr, y_disc_tr_lbl).cpu().numpy()

        z_disc_te = torch.randn(n_test_real, vae.latent_dim, device=device)
        y_disc_te_lbl = torch.randint(0, 2, (n_test_real,), dtype=torch.long, device=device)
        syn_disc_te = vae.decode(z_disc_te, y_disc_te_lbl).cpu().numpy()

    X_disc_tr = np.vstack([tr_disc_real, syn_disc_tr])
    y_disc_tr = np.array([0] * half_real + [1] * half_real)

    X_disc_te = np.vstack([te_disc_real, syn_disc_te])
    y_disc_te = np.array([0] * n_test_real + [1] * n_test_real)

    disc = LogisticRegression(max_iter=200).fit(X_disc_tr, y_disc_tr)
    disc_auc = float(roc_auc_score(y_disc_te, disc.predict_proba(X_disc_te)[:, 1]))

    return {
        "per_latent_dim_kl": per_dim_kl_table,
        "collapsed_dims_count": collapsed_count,
        "posterior_collapse_detected": bool(collapsed_count == vae.latent_dim),
        "decoder_output_variation": decoder_variation_table,
        "wasserstein_distance_table": wasserstein_table,
        "discriminator_auc": round(disc_auc, 4),
    }


# =====================================================================
# Baseline Augmenters (Part 4c)
# =====================================================================

class GaussianFeatureAugmenter:
    """Class-conditional Gaussian fit on real train features."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.params: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianFeatureAugmenter":
        for c in np.unique(y):
            Xc = X[y == c]
            mu = np.mean(Xc, axis=0)
            cov = np.cov(Xc, rowvar=False) + 1e-4 * np.eye(X.shape[1])
            self.params[int(c)] = (mu, cov)
        return self

    def sample(self, num_samples: int, target_class: int) -> np.ndarray:
        mu, cov = self.params[target_class]
        samples = self.rng.multivariate_normal(mu, cov, size=num_samples)
        return np.clip(samples, 0.0, 1.0).astype(np.float32)


class InterpolationFeatureAugmenter:
    """Within-class interpolation (Mixup within class)."""

    def __init__(self, alpha: float = 2.0, seed: int = 42):
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        self.class_data: Dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InterpolationFeatureAugmenter":
        for c in np.unique(y):
            self.class_data[int(c)] = X[y == c]
        return self

    def sample(self, num_samples: int, target_class: int) -> np.ndarray:
        Xc = self.class_data[target_class]
        n_c = len(Xc)
        idx1 = self.rng.choice(n_c, size=num_samples, replace=True)
        idx2 = self.rng.choice(n_c, size=num_samples, replace=True)
        lambdas = self.rng.beta(self.alpha, self.alpha, size=(num_samples, 1)).astype(np.float32)
        samples = lambdas * Xc[idx1] + (1.0 - lambdas) * Xc[idx2]
        return np.clip(samples, 0.0, 1.0).astype(np.float32)

