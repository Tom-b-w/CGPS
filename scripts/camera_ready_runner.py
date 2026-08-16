"""
Prompt refinement: Lambda-weighted top-k with high-confidence centroids.

Variants
--------
adEM          adEM baseline (no refinement)
refine_noisy  Lambda @ dota.mu[c] scoring, top-10%, std fallback   (previous result)
refine_hc     Lambda @ hc_mu[c] scoring when hc_cnt[c]>=min_hc,
              else falls back to dota.mu.  hc_mu = mean of high-conf
              samples (max(prob)>conf_thresh) only.
              DOTA Gaussian still uses ALL samples (no sparse-class issue).
refine_unified  Unified candidate pool: individual std templates + CuPL prompts
              compete in the same scored pool.  Lambda picks top top_frac from
              ALL candidates; the OOD-aware templates in imagenet_r/s naturally
              score higher than generic natural-image descriptions.
              Fallback when best score<0: single best-scoring std template
              (not mean of all T, which dilutes the OOD signal).

Pipeline (refine_* variants)
------------------------------
1. Phase 1 (first refine_frac of stream):
   - adEM supervision (lower-entropy of std vs cupl wins)
   - DOTA accumulates:  mu[C,D], Lambda[D,D]
   - HC accumulator:    hc_sum[C,D], hc_cnt[C]  (consensus: std AND cupl agree)
2. One-shot refinement at step = int(N_total * refine_frac):
   - Build W_c = normalize(Lambda @ mu_c)
     where mu_c = hc_mu[c] if hc_cnt[c]>=min_hc else dota.mu[c]
   - refine_hc:      score only cupl_embs[c];  selected + std_mean → weight
   - refine_unified: score cat([std_embs[c], cupl_embs[c]]);  top-k → weight
   - Replace cupl EM branch with refined weights
3. Phase 2 (remaining stream): adEM with refined cupl branch; DOTA continues.

Usage
-----
python scripts/evaluate.py --data-root /path/to/data --datasets dtd
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_DOTA_DIR = str(_ROOT / "third_party" / "dota")
_CUPL_DIR = str(_ROOT / "assets" / "prompts")
sys.path.insert(0, _DOTA_DIR)

import clip
from dota_gda_em_aug import DOTA
from utils import build_test_data_loader, get_clip_logits_aug, get_config_file


# ── DOTABayes: text-prior shrinkage on prediction-mu, NOT on Sigma-centering mu
# Why: applying shrinkage to self.mu would bias the Sigma update (Welford uses
# `x - self.mu` as centered features). The bias is rank-1 in the direction
# (mu_em - mu_text) and inflates Sigma_c → Lambda becomes less discriminative
# in exactly the text-visual mismatch direction. Confirmed by EuroSAT sweep
# where every kappa setting (1.0–10.0) hurt accuracy ~1%, even when final
# kappa ≈ 0 (the early-phase Sigma damage persists).
#
# v2 design — two separately-updated means per class:
#   mu_em    = standard DOTA EM mean         (used for Sigma & internal state)
#   mu_pred  = shrinkage(mu_em, mu_text)     (used only by predict())
# Original DOTA fit() is inherited unchanged. Only predict() is overridden:
#   alpha_c  = kappa_c / (kappa_c + n_eff_c)
#   mu_pred  = (1 - alpha_c)·mu_em + alpha_c·mu_text_prior
#   kappa_c  = kappa_max · exp(-n_eff_c / n_decay),   n_eff_c = c - c_init
# Effect:
#   - sparse / OOD classes (small n_eff) → alpha large → mu_pred near mu_text
#   - mature classes (large n_eff)        → alpha → 0  → mu_pred = mu_em
#   - Sigma estimation is never disturbed (always centered on mu_em)
class DOTABayes(DOTA):
    def __init__(self, cfg, input_shape: int, num_classes: int,
                 clip_weights: torch.Tensor, mu_text_prior: torch.Tensor,
                 kappa_max: float = 10.0, n_decay: float = 200.0,
                 streaming_update_Sigma: bool = True):
        super().__init__(cfg, input_shape, num_classes, clip_weights,
                         streaming_update_Sigma=streaming_update_Sigma)
        # Frozen text prior in [C, D] float32 (matches self.mu layout / dtype)
        self.register_buffer("mu_text_prior",
                             mu_text_prior.to(self.device).float())
        self.kappa_max = float(kappa_max)
        self.n_decay   = float(n_decay)
        # DOTA initializes self.c = ones(C), so c_init = 1 per class
        self.register_buffer("c_init", torch.ones_like(self.c))

    # fit() inherited unchanged — self.mu evolves as in standard DOTA

    # Shrinkage gate: external code sets this True once Phase 1 ends.
    # Until then, predict() falls back to standard DOTA (self.mu, no shrinkage).
    # Rationale: during Phase 1, n_eff is small → alpha large → mu_pred ≈ mu_text.
    # But x^T Λ μ_text suffers a modality mismatch (visual Λ sharpens visual
    # directions, μ_text lives partly off that subspace), so early-phase
    # bayes predictions are systematically worse than standard DOTA.
    # On ImageNet-A 1000 classes / 7500 samples, per-class n_eff stays small
    # for the WHOLE run, so alpha remains large in Phase 2 too — that is
    # exactly where shrinkage should still help.
    use_shrinkage: bool = False

    def _mu_pred(self) -> torch.Tensor:
        """Shrinkage between EM mean (self.mu) and frozen text prior. [C, D]"""
        n_eff = (self.c - self.c_init).clamp(min=0.0)              # [C]
        kappa = self.kappa_max * torch.exp(-n_eff / self.n_decay)  # [C]
        alpha = kappa / (kappa + n_eff).clamp(min=1e-8)            # [C]
        a = alpha.unsqueeze(1)                                      # [C, 1]
        return (1.0 - a) * self.mu + a * self.mu_text_prior

    def predict(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.use_shrinkage:
            return super().predict(X)
        X = X.to(self.device)
        with torch.no_grad():
            Lambda = self.Lambda
            M = self._mu_pred().transpose(1, 0).half()      # [D, C]
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)
            return torch.matmul(X, W) - c


# ── DOTAShare: per-class Lambda via text-similarity-weighted Sigma sharing ─
# Direction B: text inter-class geometry K_t[c, c'] = cos(mu_t_c, mu_t_c')
# guides how each class borrows covariance structure from other classes.
#
# Empirically validated (visualize_dirB_assumptions.py):
#   - H4: corr(K_t, Frobenius_cos(Sigma_v_c, Sigma_v_c')) = 0.54 on EuroSAT
#     (similar text → similar visual covariance shape)
#
# update() override:
#   W       = softmax_row(K_t / tau)              [C, C], sharing weights
#   Sigma_shared_c = sum_c' W[c, c'] · Sigma_c'   [C, D, D]
#   alpha_c = clamp(1 - n_eff_c / n_target, 0, 1) per-class shrinkage weight
#   Sigma_eff_c    = (1-alpha_c) Sigma_c + alpha_c Sigma_shared_c
#   Lambda_c       = inv((1-eps) Sigma_eff_c + eps I)
# predict() uses per-class Lambda_c (not shared Lambda).
#
# fit() inherited: same M-step as DOTA (mu_c, Sigma_c).
class DOTAShare(DOTA):
    def __init__(self, cfg, input_shape: int, num_classes: int,
                 clip_weights: torch.Tensor, K_t: torch.Tensor,
                 tau: float = 0.05, n_target: float = 200.0,
                 streaming_update_Sigma: bool = True):
        super().__init__(cfg, input_shape, num_classes, clip_weights,
                         streaming_update_Sigma=streaming_update_Sigma)
        # K_t: [C, C] frozen text-similarity matrix
        self.register_buffer("K_t", K_t.to(self.device).float())
        self.tau = float(tau)
        self.n_target = float(n_target)
        self.register_buffer("c_init", torch.ones_like(self.c))
        # Per-class Lambda, populated by update().
        # Stored on the module but not registered as buffer (avoid state_dict bloat).
        self.Lambda_per_class: torch.Tensor | None = None

    @torch.no_grad()
    def update(self) -> None:  # type: ignore[override]
        # 1. Sharing weights (rows sum to 1)
        W_share = F.softmax(self.K_t / self.tau, dim=1)           # [C, C]

        # 2. Per-class text-shared Sigma
        Sigma_shared = torch.einsum("ck,kij->cij", W_share, self.Sigma)  # [C, D, D]

        # 3. Adaptive mixing: sparse classes (small n_eff) lean on shared
        n_eff = (self.c - self.c_init).clamp(min=0.0)
        alpha = (1.0 - n_eff / self.n_target).clamp(min=0.0, max=1.0)    # [C]
        a = alpha.view(-1, 1, 1)                                          # [C, 1, 1]

        # 4. Effective per-class Sigma
        Sigma_eff = (1.0 - a) * self.Sigma + a * Sigma_shared

        # 5. Per-class Lambda (batched inverse)
        eye = torch.eye(self.input_shape, device=self.device).unsqueeze(0)  # [1, D, D]
        reg = (1.0 - self.epsilon) * Sigma_eff + self.epsilon * eye
        self.Lambda_per_class = torch.inverse(reg).half()                   # [C, D, D]

        # 6. Maintain shared Lambda too (used as fallback before any update or
        # for any external code that accesses self.Lambda).
        self.overall_Sigma = Sigma_eff.mean(dim=0)
        self.Lambda = torch.inverse(
            (1.0 - self.epsilon) * self.overall_Sigma
            + self.epsilon * torch.eye(self.input_shape, device=self.device)
        ).half()

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Use full Mahalanobis form  -0.5 (x - mu_c)^T Lambda_c (x - mu_c)
        # NOT  x^T Lambda_c mu_c - 0.5 mu_c^T Lambda_c mu_c (DOTA's LDA form).
        # The LDA form is only valid when Lambda is shared across classes — it
        # drops the -0.5 x^T Lambda x term as a class-constant. With per-class
        # Lambda_c, the scale of x^T Lambda_c mu_c varies with ||Lambda_c||,
        # making the formula systematically biased toward classes with the
        # sharpest Lambda. Mahalanobis distance restores cross-class comparability.
        if self.Lambda_per_class is None:
            return super().predict(X)
        X = X.to(self.device)
        mu = self.mu.half()                                                  # [C, D]
        Lambda_pc = self.Lambda_per_class                                    # [C, D, D] half
        # diff[b, c, d] = X[b, d] - mu[c, d]
        diff = X.unsqueeze(1) - mu.unsqueeze(0)                              # [B, C, D]
        # M_diff[b, c, :] = Lambda_c @ diff[b, c, :]
        M_diff = torch.einsum("cde, bce -> bcd", Lambda_pc, diff)            # [B, C, D]
        quad   = (diff * M_diff).sum(dim=-1)                                  # [B, C]
        logits = -0.5 * quad                                                  # [B, C]
        # Center across classes so the per-class offset from x^T Lambda_c x
        # (which varies when Lambda_c != Lambda_c') does not bias predictions.
        # Equivalent to DOTA's LDA form when all Lambda_c are equal.
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits

DATA_ROOT = os.environ.get("CGPS_DATA_ROOT", "")
CONFIG_DIR = os.path.join(_DOTA_DIR, "configs", "vit")

CUPL_FILES = {
    "caltech101":     "cupl_prompts_caltech101.json",
    "dtd":            "cupl_prompts_dtd.json",
    "eurosat":        "cupl_prompts_eurosat.json",
    "fgvc":           "cupl_prompts_fgvc.json",
    "food101":        "cupl_prompts_food101.json",
    "oxford_flowers": "cupl_prompts_oxford_flowers.json",
    "oxford_pets":    "cupl_prompts_oxford_pets.json",
    "stanford_cars":  "cupl_prompts_stanford_cars.json",
    "sun397":         "cupl_prompts_sun397.json",
    "ucf101":         "cupl_prompts_ucf101.json",
    "imagenet":       "CuPL_prompts_imagenet.json",
    "imagenet_a":     "CuPL_prompts_imagenet.json",
    "imagenet_r":     "CuPL_prompts_imagenet.json",
    "imagenet_s":     "CuPL_prompts_imagenet.json",
    "imagenet_v":     "CuPL_prompts_imagenet.json",
}

# build_test_data_loader / get_config_file expect single-letter codes for ImageNet variants
DOTA_INTERNAL_NAME = {
    "imagenet":   "I",
    "imagenet_a": "A",
    "imagenet_r": "R",
    "imagenet_s": "S",
    "imagenet_v": "V",
}

DEFAULT_DATASETS = [
    "fgvc", "caltech101", "stanford_cars", "dtd", "eurosat",
    "oxford_flowers", "food101", "oxford_pets", "sun397", "ucf101",
]


# ── fast multi-weight logit helper ───────────────────────────────────────────

@torch.no_grad()
def encode_images(images, clip_model) -> torch.Tensor:
    """Encode augmented views once → [N_views, D] L2-normalized, same dtype as CLIP model."""
    model_device = next(clip_model.parameters()).device
    if isinstance(images, list):
        imgs = torch.cat(images, dim=0).to(model_device)
    else:
        imgs = images.to(model_device)
    feats = clip_model.encode_image(imgs)
    return F.normalize(feats, dim=-1)  # keep original dtype (fp16 for ViT-B/16)


@torch.no_grad()
def logits_from_features(
    image_features_all: torch.Tensor, clip_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (selected_feats, mean_logits [1,C], prob_map [K,C]) from pre-encoded features.

    Performs the same entropy-based top-10% view selection as get_clip_logits_aug
    but without re-encoding the image.  Features kept in original dtype; logits in fp32.
    """
    clip_logits = (100.0 * image_features_all.float() @ clip_weights.float())  # [N, C] fp32

    if image_features_all.size(0) > 1:
        ent = -(clip_logits.softmax(-1) * clip_logits.log_softmax(-1)).sum(-1)
        k = max(1, int(ent.size(0) * 0.1))
        sel = ent.argsort()[:k]
        sel_logits  = clip_logits[sel]
        sel_feats   = image_features_all[sel]           # keep original dtype for DOTA
        mean_logits = sel_logits.mean(0, keepdim=True)
        prob_map    = sel_logits.softmax(-1)
    else:
        sel_feats   = image_features_all
        mean_logits = clip_logits
        prob_map    = clip_logits.softmax(-1)

    return sel_feats, mean_logits, prob_map


# ── text encoding helpers ─────────────────────────────────────────────────────

def encode_texts(
    clip_model, texts: list[str], device: torch.device, batch: int = 512
) -> torch.Tensor:
    """[N, D] L2-normalized."""
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            toks = clip.tokenize(texts[i : i + batch]).to(device)
            v = clip_model.encode_text(toks).float()
            vecs.append(F.normalize(v, dim=-1))
    return torch.cat(vecs, dim=0)


def build_std_weights(
    classnames: list[str], template: list[str], clip_model, device: torch.device
) -> torch.Tensor:
    """[D, C] L2-normalized."""
    vecs = []
    for cname in classnames:
        prompts = [t.format(cname.replace("_", " ")) for t in template]
        embs = encode_texts(clip_model, prompts, device)
        vecs.append(F.normalize(embs.mean(0), dim=-1))
    return torch.stack(vecs, dim=1)


def build_cupl_weights(
    classnames: list[str], template: list[str],
    cupl_data: dict, clip_model, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Returns (clip_weights_cupl [D,C], std_embs [C×[3,D]], cupl_embs [C×[M,D]])."""
    std_embs: list[torch.Tensor] = []
    cupl_embs: list[torch.Tensor] = []
    vecs: list[torch.Tensor] = []

    for cname in classnames:
        std_p = [t.format(cname.replace("_", " ")) for t in template]
        se = encode_texts(clip_model, std_p, device)
        std_embs.append(se)

        prompts = cupl_data.get(cname) or cupl_data.get(cname.replace("_", " ")) or []
        ce = encode_texts(clip_model, prompts, device) if prompts else se
        cupl_embs.append(ce)

        combined = torch.cat([se, ce], dim=0)
        vecs.append(F.normalize(combined.mean(0), dim=-1))

    return torch.stack(vecs, dim=1), std_embs, cupl_embs


# ── high-confidence accumulator ───────────────────────────────────────────────

@dataclass
class HCAccumulator:
    """Consensus HC: only accumulates when std and cupl agree on the predicted class.

    For EuroSAT, CuPL has lower entropy but worse accuracy, so high max(prob_cu)
    samples are confidently wrong.  Requiring agreement between std and cupl
    filters out most systematic errors: when both models predict the same class
    the sample is much more likely to be correct.

    Optionally also requires max(prob) > conf_thresh for the agreed class.

    ── HC-Λ (Bayes-optimal LDA via clean within-class scatter) ────────────────
    Beyond per-class sums, we also accumulate a SHARED within-class scatter via
    online Welford updates.  Because HC labels are ~0% noise, Σ_HC is an
    unbiased estimator of the true within-class scatter — precisely what LDA's
    Bayes-optimality requires.  DOTA's M-step Σ is biased by boundary samples
    misassigned with non-trivial probability mass; HC-Σ corrects this flaw.

    Λ_HC = ((1-ε) Σ_HC + ε I)^{-1}  is then a clean Mahalanobis matrix that can
    drive BOTH visual prediction (DOTA-style LDA) and text template selection
    (refine_hc-style Mahalanobis scoring), unifying the two mechanisms around
    a single, theoretically grounded central object.
    """
    hc_sum:    torch.Tensor   # [C, D]   per-class running sum (for class means)
    hc_cnt:    torch.Tensor   # [C]      per-class HC sample counts
    m2_shared: torch.Tensor   # [D, D]   shared within-class second moment (Welford)
    n_total:   torch.Tensor   # scalar   total HC samples accumulated
    conf_thresh: float

    @classmethod
    def create(cls, C: int, D: int, device: torch.device, conf_thresh: float = 0.3):
        return cls(
            hc_sum=torch.zeros(C, D, device=device),
            hc_cnt=torch.zeros(C, device=device),
            m2_shared=torch.zeros(D, D, device=device),
            n_total=torch.zeros((), device=device),
            conf_thresh=conf_thresh,
        )

    @torch.no_grad()
    def update(
        self,
        img_feats: torch.Tensor,
        prob_std: torch.Tensor,
        prob_cu: torch.Tensor,
    ) -> None:
        """img_feats [N,D], prob_std [N,C], prob_cu [N,C].

        For each new HC sample x with class c, applies Welford:
            old_mean = hc_sum[c] / hc_cnt[c]   (before update; 0 if first sample)
            hc_sum[c] += x; hc_cnt[c] += 1
            new_mean = hc_sum[c] / hc_cnt[c]
            m2_shared += outer(x - old_mean, x - new_mean)
        This gives an unbiased shared within-class second moment.
        """
        max_std, pred_std = prob_std.max(dim=1)  # [N]
        max_cu,  pred_cu  = prob_cu.max(dim=1)
        agree = pred_std == pred_cu               # both models same class
        conf  = (max_std > self.conf_thresh) & (max_cu > self.conf_thresh)
        mask  = agree & conf
        for i in range(img_feats.size(0)):
            if mask[i]:
                c = int(pred_std[i].item())
                x = img_feats[i].float()                       # [D]
                # Welford for shared within-class scatter (before sum/cnt update)
                cnt_old = self.hc_cnt[c].item()
                if cnt_old > 0:
                    old_mean = self.hc_sum[c] / cnt_old
                else:
                    old_mean = torch.zeros_like(x)
                self.hc_sum[c] += x
                self.hc_cnt[c] += 1.0
                new_mean = self.hc_sum[c] / self.hc_cnt[c]
                delta  = x - old_mean
                delta2 = x - new_mean
                self.m2_shared += torch.outer(delta, delta2)
                self.n_total   += 1.0

    def mu(self) -> torch.Tensor:
        """[C, D] L2-normalized class centroids (zero for unseen classes)."""
        cnt = self.hc_cnt.clamp(min=1).unsqueeze(1)
        return F.normalize(self.hc_sum / cnt, dim=-1)

    def raw_mu(self) -> torch.Tensor:
        """[C, D] raw class means (un-normalized), for LDA prediction.

        Mirrors DOTA's running mean storage: NOT re-normalized so that the
        magnitude (which reflects class-internal coherence) is preserved.
        """
        cnt = self.hc_cnt.clamp(min=1).unsqueeze(1)
        return self.hc_sum / cnt

    def lambda_hc(
        self,
        epsilon: float,
        min_total: int = 10,
        use_shrinkage: bool = True,
    ) -> torch.Tensor:
        """Clean Mahalanobis Λ_HC = inv((1-ε) Σ_HC + ε I), optionally with
        Ledoit-Wolf shrinkage to handle the n_HC < D regime.

        Σ_HC = m2_shared / n_total is the unbiased shared within-class scatter
        of HC samples.  Because HC labels have ~0% noise (co-training theorem),
        this Σ obeys LDA's homoscedastic-Gaussian assumption — unlike DOTA's
        soft-label Σ which is biased by boundary-sample misassignment.

        When n_HC ≪ D (e.g., 1000 samples for D=512), the empirical Σ_HC is
        rank-deficient and ε^{-1} amplifies noise eigendirections explosively.
        Ledoit-Wolf shrinkage Σ_eff = (1-λ) Σ_HC + λ σ̄² I with
        λ = D / (D + n_HC) automatically blends toward isotropic for small n,
        keeping Λ well-conditioned at any sample count.

        Returns scaled identity (1/ε) I when n_total < min_total (cold start).
        """
        D = self.m2_shared.shape[0]
        n   = float(self.n_total.item())
        dev = self.m2_shared.device
        eye = torch.eye(D, device=dev)
        if n < min_total:
            return eye / epsilon
        Sigma_HC = self.m2_shared / n
        if use_shrinkage:
            sigma_bar2 = Sigma_HC.diag().mean()
            lam        = D / (D + n)                    # ∈ (0, 1]; → 0 as n → ∞
            Sigma_HC   = (1.0 - lam) * Sigma_HC + lam * sigma_bar2 * eye
        reg = (1.0 - epsilon) * Sigma_HC + epsilon * eye
        return torch.inverse(reg)


# ── visual cache (TDA-style positive cache with two filter strategies) ──────
class _PositiveCache:
    """TDA-style per-class top-K positive cache.

    Two filter variants share the same retrieval logic but use different
    sample-priority signals:

    * **EntropyCache**: priority = -entropy (i.e., low-entropy preferred).
      Mirrors TDA / ReTA's positive cache exactly.  Vulnerable to CLIP's
      overconfidence on incorrect predictions, leading to noisy prototypes.

    * **HCCache**: priority = min(p_std_max, p_cupl_max) for samples that
      additionally satisfy Hard Consensus (W_std and W_cupl agree).  Because
      HC samples are drawn from a near-zero-error pool (co-training theorem),
      the resulting cache prototypes are dramatically cleaner.

    Retrieval logits follow the TDA/ReTA formula:
        ℓ_c(x) = α · exp(-β + β · cos(x, prototype_c)),
    where prototype_c is the mean of cached features for class c.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        shot_capacity: int = 3,
        alpha: float = 8.6,
        beta: float = 3.0,
        device: torch.device | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.shot_capacity = shot_capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        # priority is "larger is better"; storage initialized lazily
        self._features:  torch.Tensor | None = None  # [C, K, D]
        self._priority:  torch.Tensor | None = None  # [C, K]  larger = better
        self._counts:    torch.Tensor | None = None  # [C]

    def _ensure_storage(self, feature: torch.Tensor) -> None:
        if self._features is not None:
            return
        device = self.device if self.device is not None else feature.device
        D = int(feature.numel())
        self._features = torch.zeros(
            self.num_classes, self.shot_capacity, D,
            dtype=torch.float32, device=device,
        )
        self._priority = torch.full(
            (self.num_classes, self.shot_capacity), -float("inf"),
            dtype=torch.float32, device=device,
        )
        self._counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, feature: torch.Tensor, *, pred: int, priority: float) -> None:
        """Push (feature, pred) into cache; evict lowest-priority if full."""
        feature = F.normalize(feature.float().view(-1), dim=-1)
        if self.device is not None:
            feature = feature.to(self.device)
        self._ensure_storage(feature)
        assert self._features is not None
        assert self._priority is not None
        assert self._counts  is not None
        pred = int(pred)
        prio = float(priority)
        count = int(self._counts[pred].item())
        if count < self.shot_capacity:
            slot = count
            self._counts[pred] += 1
        else:
            slot = int(torch.argmin(self._priority[pred]).item())
            if prio <= float(self._priority[pred, slot].item()):
                return  # incoming sample worse than current worst → drop
        self._features[pred, slot] = feature.detach()
        self._priority[pred, slot] = prio
        order = torch.argsort(self._priority[pred], descending=True)
        self._features[pred] = self._features[pred, order]
        self._priority[pred] = self._priority[pred, order]

    @torch.no_grad()
    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        """Cache-based per-class logits [C], computed as
        ℓ_c = α · exp(-β + β · cos(x, μ_c))  for classes with non-empty cache.
        """
        feature = F.normalize(feature.float().view(-1), dim=-1)
        if self.device is not None:
            feature = feature.to(self.device)
        logits = torch.zeros(self.num_classes, dtype=feature.dtype, device=feature.device)
        if self._features is None or self._counts is None:
            return logits
        counts = self._counts.to(feature.device)
        active = counts > 0
        if not active.any():
            return logits
        features = self._features.to(feature.device, dtype=feature.dtype)
        slots = torch.arange(self.shot_capacity, device=feature.device)
        valid = slots.unsqueeze(0) < counts.unsqueeze(1)
        summed = (features * valid.unsqueeze(-1)).sum(dim=1)
        prototypes = summed[active] / counts[active].to(feature.dtype).unsqueeze(1)
        prototypes = F.normalize(prototypes, dim=-1)
        affinities = prototypes @ feature
        logits[active] = self.alpha * torch.exp(-self.beta + self.beta * affinities)
        return logits

    def size(self) -> int:
        if self._counts is None:
            return 0
        return int(self._counts.sum().item())


# ── TDA-faithful cache (positive + negative, per-class dict-of-lists) ────────
class _TDADictCache:
    """Faithful re-implementation of TDA's dict-of-lists cache.

    Mirrors tda_runner.py:update_cache + compute_cache_logits exactly.
    Two storage modes:
      * positive: store (feature, loss).  retrieval = one-hot class indicator.
      * negative: store (feature, loss, prob_map). retrieval = soft prob_map
        masked to {0,1} via (lower < p < upper) gating.

    Update rule:
      - cache[pred] keeps up to shot_capacity entries
      - sorted ascending by loss (entropy); newer with lower loss replaces last
    Retrieval:
      - affinity = z @ keys.T
      - kernel(affinity) = exp(-beta + beta*affinity)
      - logits = alpha * kernel(affinity) @ values
    """

    def __init__(
        self,
        num_classes: int,
        *,
        is_negative: bool = False,
        shot_capacity: int = 3,
        alpha: float = 5.0,
        beta: float = 5.0,
        device: torch.device | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.is_negative = is_negative
        self.shot_capacity = shot_capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.cache: dict[int, list[list]] = {}

    @torch.no_grad()
    def update(
        self,
        feature: torch.Tensor,
        loss: float,
        pred: int,
        prob_map: torch.Tensor | None = None,
    ) -> None:
        feature = F.normalize(feature.float().view(1, -1), dim=-1)
        if self.device is not None:
            feature = feature.to(self.device)
        pred = int(pred)
        item: list = [feature.detach(), float(loss)]
        if self.is_negative:
            assert prob_map is not None
            pm = prob_map.float().view(-1)
            if self.device is not None:
                pm = pm.to(self.device)
            item.append(pm.detach())
        if pred in self.cache:
            if len(self.cache[pred]) < self.shot_capacity:
                self.cache[pred].append(item)
            elif item[1] < self.cache[pred][-1][1]:
                self.cache[pred][-1] = item
            self.cache[pred] = sorted(self.cache[pred], key=lambda x: x[1])
        else:
            self.cache[pred] = [item]

    @torch.no_grad()
    def logits(
        self,
        feature: torch.Tensor,
        neg_mask_lower: float | None = None,
        neg_mask_upper: float | None = None,
    ) -> torch.Tensor:
        """Returns [C] cache logits.  Empty cache returns zeros.

        For negative cache, neg_mask_(lower,upper) gate the prob_map values
        (TDA's mask_threshold). The masked prob_map serves as cache_values.
        """
        if not self.cache:
            return torch.zeros(self.num_classes, dtype=feature.dtype, device=feature.device)
        feature = F.normalize(feature.float().view(1, -1), dim=-1)
        if self.device is not None:
            feature = feature.to(self.device)
        keys_list: list[torch.Tensor] = []
        vals_list: list[torch.Tensor] = []
        for c_idx in sorted(self.cache.keys()):
            for item in self.cache[c_idx]:
                keys_list.append(item[0])
                if self.is_negative:
                    vals_list.append(item[2])  # prob_map [C]
                else:
                    onehot = torch.zeros(self.num_classes, device=feature.device)
                    onehot[c_idx] = 1.0
                    vals_list.append(onehot)
        keys = torch.cat(keys_list, dim=0)              # [N, D]
        values = torch.stack(vals_list, dim=0)          # [N, C]
        if self.is_negative and neg_mask_lower is not None and neg_mask_upper is not None:
            values = ((values > neg_mask_lower) & (values < neg_mask_upper)).float()
        affinity = (feature @ keys.T).view(-1)          # [N]
        kernel = (-self.beta + self.beta * affinity).exp()   # [N]
        logits = self.alpha * (kernel.unsqueeze(0) @ values).view(-1)  # [C]
        return logits.to(feature.dtype)

    def size(self) -> int:
        return sum(len(v) for v in self.cache.values())


# ── TDA per-dataset hyperparameters (from official TDA configs) ──────────────
TDA_HYPERPARAMS: dict[str, dict] = {
    "caltech101":     {"pos_alpha": 5.0,   "pos_beta": 5.0},
    "dtd":            {"pos_alpha": 2.0,   "pos_beta": 3.0},
    "eurosat":        {"pos_alpha": 4.0,   "pos_beta": 8.0},
    "fgvc":           {"pos_alpha": 2.0,   "pos_beta": 2.0},
    "food101":        {"pos_alpha": 1.0,   "pos_beta": 1.0},
    "oxford_flowers": {"pos_alpha": 1.0,   "pos_beta": 5.0},
    "oxford_pets":    {"pos_alpha": 2.0,   "pos_beta": 7.0},
    "stanford_cars":  {"pos_alpha": 1.0,   "pos_beta": 7.0},
    "sun397":         {"pos_alpha": 2.0,   "pos_beta": 3.0},
    "ucf101":         {"pos_alpha": 3.0,   "pos_beta": 8.0},
    "imagenet_a":     {"pos_alpha": 2.0,   "pos_beta": 5.0},
    "imagenet_r":     {"pos_alpha": 1.0,   "pos_beta": 8.0},
    "imagenet_s":     {"pos_alpha": 2.363, "pos_beta": 7.45},
    "imagenet_v":     {"pos_alpha": 1.0,   "pos_beta": 8.0},
    "imagenet":       {"pos_alpha": 2.0,   "pos_beta": 5.0},
}
# negative cache hyperparameters are identical across datasets in TDA configs
TDA_NEG_PARAMS = {
    "alpha": 0.117, "beta": 1.0,
    "entropy_lower": 0.2, "entropy_upper": 0.5,
    "mask_lower": 0.03,  "mask_upper": 1.0,
    "shot_capacity": 2,
}
TDA_POS_SHOT_CAPACITY = 3


# ── prompt refinement ─────────────────────────────────────────────────────────

def _build_W_all(
    C: int, D: int,
    Lambda: torch.Tensor,
    hc_mu: torch.Tensor,
    hc_cnt: torch.Tensor,
    dota_mu: torch.Tensor,
    min_hc: int,
    device: torch.device,
    std_embs: list[torch.Tensor] | None = None,
    use_text_fallback: bool = False,
) -> torch.Tensor:
    """Per-class scoring directions W_all[C, D] = norm(Lambda @ mu_c)."""
    W_all = torch.empty(C, D, device=device)
    for c in range(C):
        if hc_cnt[c].item() >= min_hc:
            mu_c = hc_mu[c].float()
        elif use_text_fallback and std_embs is not None:
            mu_c = F.normalize(std_embs[c].float().mean(0), dim=-1)
        else:
            mu_c = F.normalize(dota_mu[c].float(), dim=-1)
        W_all[c] = F.normalize(Lambda @ mu_c, dim=-1)
    return W_all


def _build_scoring_direction(
    dota: DOTA,
    hc_acc: HCAccumulator | None,
    std_embs: list[torch.Tensor],
    min_hc: int,
    use_text_fallback: bool = False,
    lambda_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pre-compute Lambda-sharpened scoring directions W_all [C, D].

    use_text_fallback: when True, classes without enough HC samples fall back to
    the clean text prototype (mean of std templates) instead of Lambda@dota.mu.
    Helps on adversarial domains (imagenet_a) where dota.mu is biased by wrong
    Phase-1 predictions and Lambda sharpening amplifies the noise.

    lambda_override: when provided, used in place of dota.Lambda.  Enables the
    HC-Λ pathway where a clean within-class scatter (from HCAccumulator) drives
    template scoring instead of the noisier dota M-step Σ.
    """
    C = len(std_embs)
    Lambda = (lambda_override if lambda_override is not None else dota.Lambda).float()
    dota_mu = dota.mu.float()
    hc_mu  = hc_acc.mu()     if hc_acc is not None else None
    hc_cnt = hc_acc.hc_cnt   if hc_acc is not None else None

    W_all = torch.zeros(C, std_embs[0].shape[-1], device=dota_mu.device)
    for c in range(C):
        if hc_mu is not None and hc_cnt is not None and hc_cnt[c].item() >= min_hc:
            mu_c = hc_mu[c]
        elif use_text_fallback:
            # clean text prototype — unaffected by adversarial visual distribution
            mu_c = F.normalize(std_embs[c].float().mean(0), dim=-1)
        else:
            mu_c = F.normalize(dota_mu[c], dim=-1)
        W_all[c] = F.normalize(Lambda @ mu_c, dim=-1)
    return W_all


def refine_weights(
    dota: DOTA,
    hc_acc: HCAccumulator | None,
    std_embs: list[torch.Tensor],
    cupl_embs: list[torch.Tensor],
    top_frac: float,
    min_hc: int,
    use_text_fallback: bool = False,
    include_std_anchor: bool = True,
    lambda_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Refined clip_weights [D, C]: score only CuPL pool; std always included as anchor.

    include_std_anchor=False: final weight = mean(selected_only), no std concatenated.
    Useful when cupl_embs IS std_embs (template-only scoring) to avoid double-counting.

    lambda_override: forwards a custom Λ to template scoring (e.g. Λ_HC).
    """
    C = len(std_embs)
    W_all = _build_scoring_direction(
        dota, hc_acc, std_embs, min_hc, use_text_fallback,
        lambda_override=lambda_override,
    )

    vecs: list[torch.Tensor] = []
    for c in range(C):
        W_c  = W_all[c]
        embs = cupl_embs[c]          # [M, D] — only CuPL scored
        M_c  = embs.size(0)

        pos   = embs @ W_c
        neg   = (embs @ W_all.T).sum(1).sub(pos).div(max(C - 1, 1))
        score = pos - neg

        k       = max(1, int(M_c * top_frac))
        top_idx = score.topk(k).indices

        if score[top_idx[0]].item() < 0:
            best_std = (std_embs[c] @ W_c).argmax()
            vecs.append(std_embs[c][best_std])
        else:
            selected = embs[top_idx]
            if include_std_anchor:
                combined = torch.cat([std_embs[c], selected], dim=0)
                vecs.append(F.normalize(combined.mean(0), dim=-1))
            else:
                vecs.append(F.normalize(selected.mean(0), dim=-1))

    return torch.stack(vecs, dim=1).half()


def refine_weights_unified(
    dota: DOTA,
    hc_acc: HCAccumulator | None,
    std_embs: list[torch.Tensor],
    cupl_embs: list[torch.Tensor],
    top_frac: float,
    min_hc: int,
    use_text_fallback: bool = False,
) -> torch.Tensor:
    """Refined clip_weights [D, C]: unified pool of all std templates + CuPL prompts.

    Every individual std template competes alongside CuPL prompts in the same
    scored pool.  For datasets with OOD-aware templates (imagenet_r/s), Lambda
    naturally promotes the domain-aligned templates and suppresses mismatched ones.
    Fallback when best score<0: single best-scoring std template (not mean of all T).
    """
    C = len(std_embs)
    W_all = _build_scoring_direction(dota, hc_acc, std_embs, min_hc, use_text_fallback)

    vecs: list[torch.Tensor] = []
    for c in range(C):
        W_c      = W_all[c]
        std_c    = std_embs[c]                              # [T, D]
        cupl_c   = cupl_embs[c]                            # [M, D]
        all_cands = torch.cat([std_c, cupl_c], dim=0)      # [T+M, D]
        N_cands  = all_cands.size(0)

        pos   = all_cands @ W_c                            # [T+M]
        neg   = (all_cands @ W_all.T).sum(1).sub(pos).div(max(C - 1, 1))
        score = pos - neg

        k       = max(1, int(N_cands * top_frac))
        top_idx = score.topk(k).indices

        if score[top_idx[0]].item() < 0:
            # no candidate discriminative → single best std template
            best = (std_c @ W_c).argmax()
            vecs.append(std_c[best])
        else:
            selected = all_cands[top_idx]
            vecs.append(F.normalize(selected.mean(0), dim=-1))

    return torch.stack(vecs, dim=1).half()


def update_otw_scores(
    otw_scores: list[torch.Tensor],
    hc_X: torch.Tensor,
    hc_y: torch.Tensor,
    cupl_embs: list[torch.Tensor],
    beta: float = 0.5,
) -> None:
    """In-place update of online template-weight scores from a batch of HC samples.

    For each class c and template k:
        Δs_{c,k} = Σ_{i: y_i = c}  cos(x_i, t_{c,k})
                 - β · Σ_{i: y_i != c} cos(x_i, t_{c,k})

    The score s_{c,k} accumulates the discriminative quality of template k
    for class c.  POSITIVE evidence from HC samples of class c (sample close to
    template ⇒ template captures the class).  NEGATIVE evidence from HC
    samples of OTHER classes (sample close to template ⇒ false positive, bad).

    Args:
        otw_scores: list of [M_c] tensors, per-class per-template running scores
        hc_X: [n_hc, D] L2-normalized HC features (batch)
        hc_y: [n_hc] HC predicted class indices (~0% error by HC theorem)
        cupl_embs: list of [M_c, D] per-class CuPL template embeddings (L2-normalized)
        beta: weight of negative validation (typically 0.3-1.0)
    """
    if hc_X.numel() == 0:
        return
    C = len(otw_scores)
    hc_X_f = hc_X.float()                       # cast once for matmul dtype-match
    for c in range(C):
        pos_mask = hc_y == c
        sims_c = hc_X_f @ cupl_embs[c].float().T   # [n_hc, M_c]
        if pos_mask.any():
            otw_scores[c] += sims_c[pos_mask].sum(0)
        if (~pos_mask).any():
            otw_scores[c] -= beta * sims_c[~pos_mask].sum(0)


def otw_weights(
    otw_scores: list[torch.Tensor],
    cupl_embs: list[torch.Tensor],
    std_embs: list[torch.Tensor],
    tau: float,
    min_n_hc: int = 10,
) -> torch.Tensor:
    """Build clip_weights [D, C] via softmax-weighted CuPL template ensemble.

    For each class c:
        w_{c,k} = softmax(s_{c,k} / τ) over k=1..M_c
        μ_c     = normalize(Σ_k w_{c,k} · t_{c,k})
    The std template is included as a stable anchor in the mean (similar to
    refine_weights' include_std_anchor).  When scores are nearly zero (few HC
    samples), softmax becomes near-uniform → equivalent to vanilla CuPL mean.

    τ controls the softness:
      τ → 0:   peaked on best template (≈ argmax selection)
      τ → ∞:   uniform (≈ CuPL average baseline)
      τ ≈ 1:   smooth weighting respecting score magnitudes (recommended)
    """
    C = len(cupl_embs)
    device = cupl_embs[0].device
    vecs: list[torch.Tensor] = []
    for c in range(C):
        scores = otw_scores[c]
        # When score magnitude is tiny (cold start), fall back to uniform mean
        if scores.abs().sum().item() < 1e-3:
            cupl_mean = cupl_embs[c].float().mean(0)
            combined = torch.cat(
                [std_embs[c].float(), cupl_mean.unsqueeze(0)], dim=0
            ).mean(0)
            vecs.append(F.normalize(combined, dim=-1))
            continue
        weights = (scores / tau).softmax(dim=0)                   # [M_c]
        cupl_mu = (weights.unsqueeze(-1) * cupl_embs[c].float()).sum(0)
        # Anchor with std template ensemble (same role as refine_weights)
        combined = torch.cat(
            [std_embs[c].float(), cupl_mu.unsqueeze(0)], dim=0
        ).mean(0)
        vecs.append(F.normalize(combined, dim=-1))
    return torch.stack(vecs, dim=1).half()


def hc_lda_predict(
    X: torch.Tensor,
    Lambda_HC: torch.Tensor,
    dota: DOTA,
) -> torch.Tensor:
    """HC-supervised LDA prediction (Λ-only refinement).

    score(c | x) = x @ Λ_HC @ μ_c - 0.5 · μ_c^⊤ Λ_HC μ_c

    The class mean μ_c is taken DIRECTLY from DOTA's running Welford mean,
    which has been shown to be stable.  Only Λ is replaced by the HC-supervised
    clean Mahalanobis matrix.  This isolates the contribution of clean Σ vs
    DOTA's noisy soft-label Σ — the most theoretically motivated swap.

    The "two mechanisms unified by Λ" story still holds:  Λ_HC drives both
    template scoring (refine_weights via lambda_override) AND visual prediction
    (this function), forming the central object that ties text and visual.
    """
    Lambda = Lambda_HC.half()
    M = dota.mu.half().T                             # [D, C]
    W = Lambda @ M                                   # [D, C]
    bias = 0.5 * (M * W).sum(dim=0)                  # [C]
    return X.half() @ W - bias                       # [B, C]


def estimate_modality_gap(
    hc_acc: "HCAccumulator",
    mu_t: torch.Tensor,          # [C, D] L2-normalized text prototypes
    min_hc: int = 5,
    n_mix: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate global modality gap δ̄ from reliable HC anchors.

    Returns:
      delta_bar [D]  — weighted mean gap vector (visual - text), NOT normalized
      mu_t_corrected [C, D] — L2-normalized corrected text prototypes
      alpha [C]       — per-class blend weight (0=text only, 1=HC centroid fully used)

    For classes with enough HC samples:
      mu_corrected[c] = Norm(beta[c] * mu_hc[c] + (1-beta[c]) * Norm(mu_t[c] + delta_bar))
    For classes without HC samples:
      mu_corrected[c] = Norm(mu_t[c] + delta_bar)

    The global delta_bar benefits ALL classes, including those without any HC samples.
    """
    C, D = mu_t.shape
    device = mu_t.device
    mu_hc  = hc_acc.mu().to(device)    # [C, D] L2-normalized
    hc_cnt = hc_acc.hc_cnt.to(device)  # [C]

    reliable = hc_cnt >= min_hc         # bool mask
    n_reliable = reliable.sum().item()

    if n_reliable == 0:
        # no reliable classes → no correction possible
        return torch.zeros(D, device=device), mu_t.clone(), torch.zeros(C, device=device)

    # Gap vectors for reliable classes: visual centroid - text prototype
    mu_hc_rel = mu_hc[reliable].float()    # [K, D]
    mu_t_rel  = mu_t[reliable].float()     # [K, D]
    gap_vecs  = mu_hc_rel - mu_t_rel       # [K, D] — raw difference

    # Weighted mean: more HC samples → more reliable gap estimate
    weights = hc_cnt[reliable].float()     # [K]
    weights = weights / weights.sum()
    delta_bar = (gap_vecs * weights.unsqueeze(1)).sum(0)  # [D]

    # Apply global correction to ALL classes
    mu_t_shifted = mu_t.float() + delta_bar.unsqueeze(0)          # [C, D]
    mu_t_global  = F.normalize(mu_t_shifted, dim=-1)               # [C, D]

    # For reliable classes: blend HC centroid with globally-corrected prototype
    # beta[c] increases with HC count; saturates at n_mix samples
    beta = (hc_cnt.float() / n_mix).clamp(max=1.0)                # [C]
    mu_t_corrected = mu_t_global.clone()
    for c in range(C):
        if hc_cnt[c].item() >= min_hc:
            b = beta[c].item()
            blended = b * mu_hc[c].float() + (1.0 - b) * mu_t_global[c]
            mu_t_corrected[c] = F.normalize(blended, dim=-1)

    return delta_bar, mu_t_corrected, beta


def estimate_gap_meanshift(
    mean_visual: torch.Tensor,   # [D] — raw running mean of ALL image features (NOT L2-normed)
    mu_t: torch.Tensor,          # [C, D] — L2-normalized text prototypes
    hc_acc: "HCAccumulator | None" = None,
    min_hc: int = 5,
    n_mix: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributional Mean-Shift gap estimation.

    delta = mean(visual features) - mean(text prototypes)
    Both means are in the original embedding space (not re-normalized), so delta is
    a small vector (~0.05-0.15 norm) representing the global modality shift.

    Unlike HC-based estimation, this uses ALL test images → stable even for sparse datasets.
    Optionally blends per-class HC centroid for reliable classes (avoids over-smoothing).

    Returns:
      delta_ms [D]           — mean-shift gap vector (small norm, in embedding space)
      mu_t_corrected [C, D]  — L2-normalized corrected text prototypes
    """
    C, D = mu_t.shape
    device = mu_t.device

    # Global gap = barycenter of visual cloud - barycenter of text cloud (both in R^D)
    mean_text = mu_t.float().mean(0)          # [D] — raw mean text prototype (norm < 1)
    delta_ms  = mean_visual.float() - mean_text   # [D] small offset vector

    # Apply global shift to ALL classes (no need for HC samples)
    mu_t_shifted  = mu_t.float() + delta_ms.unsqueeze(0)  # [C, D]
    mu_t_corrected = F.normalize(mu_t_shifted, dim=-1)     # [C, D]

    # Optionally blend HC centroid for reliable classes (same as HC-based method)
    if hc_acc is not None:
        mu_hc  = hc_acc.mu().to(device)        # [C, D]
        hc_cnt = hc_acc.hc_cnt.to(device)      # [C]
        for c in range(C):
            if hc_cnt[c].item() >= min_hc:
                b = min(hc_cnt[c].item() / n_mix, 1.0)
                blended = b * mu_hc[c].float() + (1.0 - b) * mu_t_corrected[c]
                mu_t_corrected[c] = F.normalize(blended, dim=-1)

    return delta_ms, mu_t_corrected


# ── single-dataset runner ─────────────────────────────────────────────────────

def _batch_entropy(p: torch.Tensor) -> float:
    return -(p * p.log().clamp(min=-100)).sum(-1).mean().item()


def run_dataset(
    dataset_name: str,
    clip_model,
    preprocess,
    *,
    seed: int,
    device: torch.device,
    refine_frac: float = 0.25,
    top_frac: float = 0.10,
    conf_thresh: float = 0.50,
    min_hc: int = 5,
    bayes_kappa_max: float = 10.0,
    bayes_n_decay: float = 200.0,
    share_tau: float = 0.05,
    share_n_target: float = 200.0,
) -> dict[str, float]:
    t_start = time.time()
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True

    internal = DOTA_INTERNAL_NAME.get(dataset_name, dataset_name)
    cfg = get_config_file(CONFIG_DIR, internal)
    loader, classnames, template = build_test_data_loader(internal, DATA_ROOT, preprocess)
    C = len(classnames)
    N_total = len(loader.dataset)
    refine_step = int(N_total * refine_frac)

    # imagenet variants: Phase-1 accuracy is lower (adversarial/OOD) so
    # dota.mu can be biased. Use clean text prototype as fallback instead.
    use_text_fallback = internal in ("I", "A", "R", "S")

    # Save the original dataset template before any substitution.
    # Official DOTA uses the original template (7 for imagenet_a) for its
    # clip_classifier; we reproduce this faithfully for the dota_official variant.
    template_orig = list(template)

    # imagenet_a has only 7 generic templates.  Substitute imagenet_r's 80-template
    # set so the unified scoring pool is richer: Lambda can then select the subset
    # that best matches the adversarial visual distribution (e.g. "corrupted photo",
    # "blurry photo", "hard to see") and discard OOD templates (sculpture, tattoo).
    if dataset_name == "imagenet_a":
        from datasets.imagenet_r import template as _imr_template
        template = _imr_template

    cupl_path = os.path.join(_CUPL_DIR, CUPL_FILES[dataset_name])
    with open(cupl_path) as f:
        cupl_data = json.load(f)

    clip_weights_std = build_std_weights(classnames, template, clip_model, device).half()
    clip_weights_cupl, std_embs, cupl_embs = build_cupl_weights(
        classnames, template, cupl_data, clip_model, device
    )
    clip_weights_cupl = clip_weights_cupl.half()

    # Official DOTA text weights: original template (no imagenet_r substitution).
    # For all datasets except imagenet_a this is identical to clip_weights_std.
    if dataset_name == "imagenet_a":
        clip_weights_official = build_std_weights(classnames, template_orig, clip_model, device).half()
    else:
        clip_weights_official = clip_weights_std

    D = clip_weights_std.shape[0]

    def _make_dota() -> DOTA:
        init_w = torch.full((D, C), 0.001, device=device)
        m = DOTA(cfg, input_shape=D, num_classes=C, clip_weights=init_w)
        m.eval()
        return m

    # ── dota_official: faithful DOTA reproduction (official prompt, 0.001 init) ──
    # Matches official DOTA: same 0.001 init, standard templates (7 for imagenet_a),
    # M-step uses prob from standard-template logits. Only missing: TTA augmentation.
    dota_official = _make_dota()

    # four variants share identical DOTA structure but diverge at refine step
    dota_adem    = _make_dota()
    dota_noisy   = _make_dota()
    dota_hc      = _make_dota()
    dota_unified = _make_dota()
    dota_adapt   = _make_dota()   # acc-adaptive top_frac, CuPL candidates
    dota_adapt2  = _make_dota()   # acc-adaptive: CuPL when cupl_gain>0, tmpl scoring when <=0
    dota_hcproto = _make_dota()   # ablation M1 only: HC centroid as text weights, no CuPL selection
    dota_cg      = _make_dota()   # ablation M2 only: DOTA-mu centroid-guided CuPL selection, no HC filter

    # ── refine_hc_bayes: refine_hc + Bayesian text-prior shrinkage in M-step ──
    # mu_text_prior is the *unchanged* CuPL combined weight (same prototype as logits_cu)
    mu_text_prior   = clip_weights_cupl.float().T.contiguous()   # [C, D]
    init_w          = torch.full((D, C), 0.001, device=device)
    dota_hc_bayes   = DOTABayes(
        cfg, input_shape=D, num_classes=C, clip_weights=init_w,
        mu_text_prior=mu_text_prior,
        kappa_max=bayes_kappa_max, n_decay=bayes_n_decay,
    ).eval()

    # ── refine_hc_share: refine_hc + text-similarity-weighted per-class Lambda ─
    # K_t[c, c'] = cos(mu_t_c, mu_t_c') — derived from clip_weights_cupl columns
    # (each column already L2-normalized by build_cupl_weights).
    K_t = clip_weights_cupl.float().T @ clip_weights_cupl.float()   # [C, C]
    init_w_share  = torch.full((D, C), 0.001, device=device)
    dota_hc_share = DOTAShare(
        cfg, input_shape=D, num_classes=C, clip_weights=init_w_share,
        K_t=K_t, tau=share_tau, n_target=share_n_target,
    ).eval()

    hc_acc = HCAccumulator.create(C, D, device, conf_thresh=conf_thresh)

    # ── refine_hc_v2: M1+M2+M3（HC过滤更新，EuroSAT好但DTD差）────────────
    # 修改1：25% 触发时用 μ_hc_c 重置 DOTA 视觉均值
    # 修改2：只有 HC 双一致样本才进入 DOTA 更新（少样本类别过于苛刻）
    # 修改3：最终预测文本基础改用精选后的 cw_hc_v2
    dota_v2 = _make_dota()

    # ── refine_hc_v3: M1+M3（去掉 M2，保持原始 DOTA 更新策略）─────────────
    # 只重置起点（M1）+ 用精选文本（M3），让 DOTA 照常更新所有样本
    dota_v3 = _make_dota()

    # ── refine_hc_v4: 软混合重置（M1_soft+M3）──────────────────────────────
    # 问题：硬重置（v3）在细粒度少样本数据集上有害（HC质心偏向易分样本）
    # 方案：mu_new = (1-α) * dota.mu[c] + α * hc_mu[c]
    #       α = clamp(hc_cnt[c] / hc_mix_n, 0, 1)
    # 效果：HC样本越多，重置幅度越大；少样本时仅做轻微修正而非完全覆盖
    hc_mix_n = 50   # α=1 所需的 HC 样本数（可调）
    dota_v4 = _make_dota()

    # ── refine_hc_gap: 全局模态间隙估计与修正（基于 HC 锚点）────────────────
    dota_gap = _make_dota()

    # ── refine_gap_ms: 均值偏移 gap 估计（用全部图像，无需 HC）──────────────
    # δ̂ = mean(img_feats) - mean(μ_t)  — 完全标签无关，对稀疏数据集更稳定
    dota_gap_ms = _make_dota()

    # ── refine_hc_full: 精选文本（同 refine_hc）+ HC 硬标签监督视觉高斯 ────────
    # 核心思想：adEM 用软伪标签（30~70% 噪声）更新视觉高斯，
    # HC 双一致过滤后错误率≈0%，直接用 one-hot 标签替换 HC 样本的软标签，
    # 非 HC 样本仍用熵门控软标签，形成"分层"伪标签策略。
    dota_hcfull = _make_dota()

    # ── refine_hc_phase2: 精选文本 + 两阶段视觉高斯更新 ─────────────────────────
    # Phase 1 (<25%): adEM 风格，全样本软标签更新（warm-up，保留覆盖率）
    # Phase 2 (≥25%): HC 过滤软标签更新（只用 HC 样本，保留软结构，去除噪声）
    # 动机：one-hot 破坏 DOTA 内类方差估计；软标签 + HC 过滤才是正确组合
    dota_phase2 = _make_dota()

    # ── refine_hc_lambda: HC-Λ Bayes-Optimal LDA ──────────────────────────────
    # **Core innovation**:  Λ becomes the single central object that unifies the
    # two mechanisms (text refinement + visual prediction) into a dynamic whole.
    #
    # DOTA's Σ is estimated by entropy-gated soft-label M-step — boundary
    # samples are misassigned with non-trivial probability mass, biasing the
    # within-class scatter and violating LDA's Bayes-optimality precondition.
    #
    # HC samples have ~0% label noise (co-training theorem), so the shared
    # within-class scatter Σ_HC accumulated from HC samples is an unbiased
    # estimator.  Λ_HC = ((1-ε) Σ_HC + ε I)^{-1} simultaneously drives:
    #   (a) template selection via Mahalanobis discriminative scoring
    #   (b) visual prediction via LDA score(c|x) = x'Λ_HC μ_c - 0.5 μ_c'Λ_HC μ_c
    #   (c) μ_v_HC (clean visual centroids) feeds back into both (a) and (b)
    # The mutual conditioning: better Λ_HC → better text → better HC → better
    # Λ_HC, forming a self-improving cycle around a single quantity.
    #
    # M-step shares dota_hc to keep memory cost low and make this a *clean*
    # A/B test against refine_hc: the ONLY difference is which Λ drives
    # scoring and prediction (dota_hc.Lambda vs Λ_HC).

    # ── refine_dbr: Dynamic Bidirectional Refinement ─────────────────────────
    # 核心机制：文本精化与视觉高斯互相促进，形成在线闭环（不依赖固定触发点）
    #
    # 每步执行三步：
    #   E-step : 用动态文本权重 cw_dbr + DOTA 视觉修正预测
    #   M-step : HC 过滤软标签更新视觉高斯（冷启动阶段用全量 warm-up）
    #   T-step : 当 HC 计数累积到阈值时，用当前 DOTA Lambda 重新评分 CuPL 模板
    #            更新 cw_dbr（无固定 25% 触发点，自适应于数据分布）
    #
    # 闭环：更好的视觉高斯 → 更准的 Lambda → 更好的模板评分 → 更好的文本权重
    #       → 更多 HC 样本 → 更好的视觉高斯更新 → …
    dota_dbr = _make_dota()
    # 第一次 T-step 与 refine_hc 的 25% 触发点对齐：
    #   dbr_min_hc_total ≈ N_total × refine_frac × HC_rate(≈0.5)
    # 后续 T-step 每 ~5% 数据触发一次：
    #   dbr_update_every ≈ N_total × 0.05 × 0.5
    # 这样 Lambda 在第一次 T-step 时已经累积了足够的数据，评分才有区分力
    dbr_min_hc_total  = max(int(N_total * refine_frac * 0.5), min_hc * C)
    dbr_update_every  = max(int(N_total * 0.05 * 0.5), C)
    dbr_last_hc_count = 0.0   # 上次 T-step 时的 total HC count
    dbr_t_step_count  = 0     # T-step 触发次数（用于诊断输出）

    # per-variant state
    correct   = {"dota_official": 0, "adEM": 0, "refine_noisy": 0, "refine_hc": 0, "refine_unified": 0, "refine_adaptive": 0, "refine_adapt2": 0, "refine_pathA": 0, "refine_hc_bayes": 0, "refine_hc_share": 0, "refine_hc_v2": 0, "refine_hc_v3": 0, "refine_hc_v4": 0, "refine_hc_gap": 0, "refine_gap_ms": 0, "refine_hc_full": 0, "refine_hc_phase2": 0, "refine_dbr": 0, "refine_hc_lambda": 0, "refine_hc_otw": 0, "tda_entropy_cache": 0, "tda_hc_cache": 0, "tda_baseline": 0, "tda_refine_hc": 0, "tda_hc_proto": 0, "tda_centroid_guided": 0, "dota_hc_proto": 0, "dota_centroid_guided": 0}
    total     = {"dota_official": 0, "adEM": 0, "refine_noisy": 0, "refine_hc": 0, "refine_unified": 0, "refine_adaptive": 0, "refine_adapt2": 0, "refine_pathA": 0, "refine_hc_bayes": 0, "refine_hc_share": 0, "refine_hc_v2": 0, "refine_hc_v3": 0, "refine_hc_v4": 0, "refine_hc_gap": 0, "refine_gap_ms": 0, "refine_hc_full": 0, "refine_hc_phase2": 0, "refine_dbr": 0, "refine_hc_lambda": 0, "refine_hc_otw": 0, "tda_entropy_cache": 0, "tda_hc_cache": 0, "tda_baseline": 0, "tda_refine_hc": 0, "tda_hc_proto": 0, "tda_centroid_guided": 0, "dota_hc_proto": 0, "dota_centroid_guided": 0}
    # refined weights: start as cupl, replaced at refine_step
    cw_noisy    = clip_weights_cupl
    cw_hc       = clip_weights_cupl
    cw_unified  = clip_weights_cupl
    cw_adapt    = clip_weights_cupl
    cw_adapt2   = clip_weights_cupl
    cw_hc_bayes = clip_weights_cupl     # mirrors cw_hc; refined at 25%
    cw_hc_share = clip_weights_cupl     # mirrors cw_hc; refined at 25%
    cw_hc_v2    = clip_weights_cupl     # refined at 25%
    cw_hc_v3    = clip_weights_cupl     # refined at 25%
    cw_hc_v4    = clip_weights_cupl     # refined at 25%
    cw_hc_gap   = clip_weights_cupl     # corrected by HC gap; refined at 25%
    cw_gap_ms   = clip_weights_cupl     # corrected by mean-shift gap; refined at 25%
    cw_hc_full  = clip_weights_cupl     # text side: same selection as refine_hc; refined at 25%
    cw_hc_phase2 = clip_weights_cupl    # text side: same as refine_hc; Phase2 uses HC-filtered update
    cw_dbr      = clip_weights_cupl     # DBR: starts as CuPL, updated by T-step dynamically
    cw_hc_lambda = clip_weights_cupl    # HC-Λ: refined at 25% using Λ_HC for scoring
    cw_tda_refined = clip_weights_std   # TDA-refined: starts as std, refined at 25% to include top-K CuPL
    cw_tda_hc_proto        = clip_weights_std   # ablation M1 only: HC centroids as text weights
    cw_tda_centroid_guided = clip_weights_std   # ablation M2 only: DOTA-mu centroid-guided CuPL
    cw_dota_hc_proto       = clip_weights_std   # ablation M1 only: HC centroids for DOTA
    cw_dota_cg             = clip_weights_cupl  # ablation M2 only: DOTA-mu centroid-guided CuPL
    # Λ_HC snapshot used for prediction throughout Phase 2.  Re-computed at each
    # T-step (here: just at refine_step) from hc_acc's accumulated second moment.
    Lambda_HC: torch.Tensor | None = None

    # ── refine_hc_otw: HC-Validated Online Template Weighting ───────────────
    # Continuous softmax weighting of CuPL templates by HC-validation score.
    # Updates EVERY step (truly online) — no fixed refine_step needed.
    # Combines:
    #   (a) POSITIVE validation: HC samples of class c pull weight toward
    #       templates that match those samples
    #   (b) NEGATIVE validation: HC samples of OTHER classes push weight AWAY
    #       from templates that mismatch those samples (false positive penalty)
    # Avoids Σ_HC bias entirely (no covariance estimation needed).
    # Differs from refine_hc: continuous weights vs top-K selection;
    # online vs one-shot; uses negative HC for cross-class discrimination.
    otw_tau  = 1.0     # softmax temperature; τ→0 ≈ argmax, τ→∞ ≈ uniform
    otw_beta = 0.5     # negative validation weight
    otw_scores = [torch.zeros(cupl_embs[c].size(0), device=device, dtype=torch.float32)
                  for c in range(C)]
    cw_otw = clip_weights_cupl          # initial: CuPL baseline (scores are zero)
    otw_update_every_n_hc = max(C // 2, 5)
    otw_last_hc_count = 0.0

    # ── TDA-style positive caches: ENTROPY vs HC filter ─────────────────────
    # Both caches share the TDA retrieval formula  ℓ_c = α·exp(-β + β·cos(x,μ_c))
    # and the same shot_capacity / α / β; they differ only in WHICH samples are
    # admitted and how priority is assigned:
    #   - entropy_cache: priority = -entropy(prob_cu)        (TDA / ReTA baseline)
    #   - hc_cache:      priority = min(p_std_max, p_cupl_max) for HC samples
    # If HC mechanism is a fundamentally stronger reliability signal than
    # entropy, hc_cache should consistently outperform entropy_cache.
    cache_shot_capacity = 3
    cache_alpha         = 8.6
    cache_beta          = 3.0
    entropy_cache = _PositiveCache(
        C, shot_capacity=cache_shot_capacity,
        alpha=cache_alpha, beta=cache_beta, device=device,
    )
    hc_cache = _PositiveCache(
        C, shot_capacity=cache_shot_capacity,
        alpha=cache_alpha, beta=cache_beta, device=device,
    )

    # ── Faithful TDA implementation: pos + neg cache, std prompts, per-dataset α/β ──
    # tda_baseline:      faithful TDA = zero-shot logits(std) + pos cache + neg cache
    # tda_refine_hc:     same TDA architecture but text uses HC-refined (std+top-K CuPL)
    #                    instead of std-only — the key experiment requested by user
    tda_params = TDA_HYPERPARAMS.get(dataset_name, {"pos_alpha": 2.0, "pos_beta": 5.0})
    # pos_cache shared across tda_baseline and tda_refine_hc (TDA logs identical)
    tda_pos_baseline = _TDADictCache(
        C, is_negative=False, shot_capacity=TDA_POS_SHOT_CAPACITY,
        alpha=tda_params["pos_alpha"], beta=tda_params["pos_beta"], device=device,
    )
    tda_neg_baseline = _TDADictCache(
        C, is_negative=True, shot_capacity=TDA_NEG_PARAMS["shot_capacity"],
        alpha=TDA_NEG_PARAMS["alpha"], beta=TDA_NEG_PARAMS["beta"], device=device,
    )
    tda_pos_refined = _TDADictCache(
        C, is_negative=False, shot_capacity=TDA_POS_SHOT_CAPACITY,
        alpha=tda_params["pos_alpha"], beta=tda_params["pos_beta"], device=device,
    )
    tda_neg_refined = _TDADictCache(
        C, is_negative=True, shot_capacity=TDA_NEG_PARAMS["shot_capacity"],
        alpha=TDA_NEG_PARAMS["alpha"], beta=TDA_NEG_PARAMS["beta"], device=device,
    )
    # ablation: M1 only (HC centroid as text weights, TDA cache)
    tda_pos_hc_proto = _TDADictCache(
        C, is_negative=False, shot_capacity=TDA_POS_SHOT_CAPACITY,
        alpha=tda_params["pos_alpha"], beta=tda_params["pos_beta"], device=device,
    )
    tda_neg_hc_proto = _TDADictCache(
        C, is_negative=True, shot_capacity=TDA_NEG_PARAMS["shot_capacity"],
        alpha=TDA_NEG_PARAMS["alpha"], beta=TDA_NEG_PARAMS["beta"], device=device,
    )
    # ablation: M2 only (DOTA-mu centroid-guided CuPL selection, TDA cache)
    tda_pos_centroid_guided = _TDADictCache(
        C, is_negative=False, shot_capacity=TDA_POS_SHOT_CAPACITY,
        alpha=tda_params["pos_alpha"], beta=tda_params["pos_beta"], device=device,
    )
    tda_neg_centroid_guided = _TDADictCache(
        C, is_negative=True, shot_capacity=TDA_NEG_PARAMS["shot_capacity"],
        alpha=TDA_NEG_PARAMS["alpha"], beta=TDA_NEG_PARAMS["beta"], device=device,
    )

    refined     = {"refine_noisy": False, "refine_hc": False, "refine_unified": False, "refine_adaptive": False, "refine_adapt2": False, "refine_pathA": False, "refine_hc_bayes": False, "refine_hc_share": False, "refine_hc_v2": False, "refine_hc_v3": False, "refine_hc_v4": False, "refine_hc_gap": False, "refine_gap_ms": False, "refine_hc_full": False, "refine_hc_phase2": False, "refine_hc_lambda": False, "refine_hc_otw": False, "tda_refine_hc": False, "tda_hc_proto": False, "tda_centroid_guided": False, "dota_hc_proto": False, "dota_centroid_guided": False}

    # ── 在线视觉均值累积（无需标签，用于均值偏移 gap 估计）──────────────────
    running_visual_sum = torch.zeros(D, device=device)
    running_visual_n   = 0

    # ── pathA state ──────────────────────────────────────────────────────────
    # clip_weights_pa: dynamic text weights for prediction base (logits_pa)
    # EM supervision reuses dota_hc (Lambda identical to refine_hc throughout)
    clip_weights_pa  = clip_weights_cupl.clone()
    pa_ema_score     = [torch.zeros(cupl_embs[c].size(0), device=device) for c in range(C)]
    pa_ema_init      = [False] * C
    pa_selection     = [None]  * C
    pa_update_count  = [0]     * C
    pa_ema_alpha     = 0.10
    pa_min_score_gain = 0.02

    # Phase-1 tracking for adaptive top_frac:
    # We use CuPL accuracy gain over std to decide top_frac:
    #   cupl_gain >= hi_gain% → CuPL genuinely better → aggressive selection (tf=top_frac)
    #   cupl_gain <= lo_gain% → CuPL not better      → no selection       (tf=1.0)
    # Entropy is also tracked for diagnostic output only.
    p1_H_std:       float = 0.0
    p1_H_cu:        float = 0.0
    p1_n:           int   = 0
    p1_correct_std: int   = 0
    p1_correct_cu:  int   = 0
    p1_total:       int   = 0
    auto_top_frac:  float = top_frac  # computed at refine_step

    step = 0

    with torch.no_grad():
        for images, target in tqdm(loader, desc=f"  {dataset_name}", leave=False):
            target = target.to(device)

            # ── one-shot refinement trigger ───────────────────────────────
            if step >= refine_step:
                if not refined["refine_noisy"]:
                    cw_noisy = refine_weights(
                        dota_noisy, None, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_noisy"] = True
                if not refined["refine_hc"]:
                    cw_hc = refine_weights(
                        dota_hc, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_hc"] = True
                if not refined["refine_hc_bayes"]:
                    # Same E-step refinement as refine_hc, scored against dota_hc_bayes
                    cw_hc_bayes = refine_weights(
                        dota_hc_bayes, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    # Enable text-prior shrinkage in predict() starting now
                    dota_hc_bayes.use_shrinkage = True
                    refined["refine_hc_bayes"] = True
                if not refined["refine_hc_share"]:
                    # E-step refinement identical to refine_hc (inherits its cw)
                    cw_hc_share = refine_weights(
                        dota_hc_share, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_hc_share"] = True
                if not refined["refine_hc_v2"]:
                    # E步（修改3前提）：同 refine_hc 选 prompt
                    cw_hc_v2 = refine_weights(
                        dota_v2, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    # 修改1：用 HC 视觉质心重置 DOTA 视觉均值
                    # 理由：DOTA mu 从 0 出发靠污染伪标签缓慢漂移，μ_hc_c 是
                    #       0% 污染样本的均值，直接作为 Phase 2 的起点更准确
                    hc_mu_now  = hc_acc.mu()       # [C, D] L2-normalized
                    hc_cnt_now = hc_acc.hc_cnt     # [C]
                    sigma0 = cfg['sigma'] * torch.eye(D, dtype=torch.float32, device=device)
                    for c_idx in range(C):
                        if hc_cnt_now[c_idx].item() >= min_hc:
                            dota_v2.mu[c_idx]    = hc_mu_now[c_idx].to(dota_v2.mu.dtype)
                            dota_v2.c[c_idx]     = hc_cnt_now[c_idx]
                            # Sigma 也重置：原 Sigma 是相对旧 mu 积累的，
                            # mu 变了后旧 Sigma 会引入虚假方差，故重置为各向同性先验
                            dota_v2.Sigma[c_idx] = sigma0
                    dota_v2.overall_Sigma = dota_v2.Sigma.mean(0)
                    dota_v2.update()   # 重算 Lambda
                    refined["refine_hc_v2"] = True
                if not refined["refine_hc_v3"]:
                    # v3 = M1+M3 only: 用 HC 质心重置 mu，但不限制后续 DOTA 更新
                    cw_hc_v3 = refine_weights(
                        dota_v3, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    hc_mu_v3  = hc_acc.mu()
                    hc_cnt_v3 = hc_acc.hc_cnt
                    sigma0_v3 = cfg['sigma'] * torch.eye(D, dtype=torch.float32, device=device)
                    for c_idx in range(C):
                        if hc_cnt_v3[c_idx].item() >= min_hc:
                            dota_v3.mu[c_idx]    = hc_mu_v3[c_idx].to(dota_v3.mu.dtype)
                            dota_v3.c[c_idx]     = hc_cnt_v3[c_idx]
                            dota_v3.Sigma[c_idx] = sigma0_v3
                    dota_v3.overall_Sigma = dota_v3.Sigma.mean(0)
                    dota_v3.update()
                    refined["refine_hc_v3"] = True
                if not refined["refine_hc_v4"]:
                    # v4 = 软混合重置 M1_soft+M3
                    # α[c] = clamp(hc_cnt[c] / hc_mix_n, 0, 1)
                    # mu_new[c] = normalize((1-α)*dota_v4.mu[c] + α*hc_mu[c])
                    cw_hc_v4 = refine_weights(
                        dota_v4, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    hc_mu_v4  = hc_acc.mu()            # [C, D] L2-normalized
                    hc_cnt_v4 = hc_acc.hc_cnt.float()  # [C]
                    alpha_v4  = (hc_cnt_v4 / hc_mix_n).clamp(max=1.0)  # [C]
                    sigma0_v4 = cfg['sigma'] * torch.eye(D, dtype=torch.float32, device=device)
                    for c_idx in range(C):
                        a = alpha_v4[c_idx].item()
                        if a > 0:
                            old_mu = dota_v4.mu[c_idx].float()
                            new_mu = (1.0 - a) * old_mu + a * hc_mu_v4[c_idx].float()
                            # L2 normalize 混合结果（特征都在单位球上）
                            norm = new_mu.norm()
                            if norm > 1e-8:
                                new_mu = new_mu / norm
                            dota_v4.mu[c_idx] = new_mu.to(dota_v4.mu.dtype)
                            # Sigma 按 α 线性插值：α 大时趋向各向同性先验，α 小时保留历史
                            dota_v4.Sigma[c_idx] = (1.0 - a) * dota_v4.Sigma[c_idx] + a * sigma0_v4
                    dota_v4.overall_Sigma = dota_v4.Sigma.mean(0)
                    dota_v4.update()
                    refined["refine_hc_v4"] = True
                if not refined["refine_hc_gap"]:
                    # TTMGE: 全局模态间隙估计与修正
                    # Step1: μ_t 来自 std 模板均值（最干净的文本原型）
                    mu_t_std = clip_weights_std.float().T.contiguous()  # [C, D] L2-normed
                    # Step2: 估计全局间隙 δ̄，修正所有类
                    delta_bar, mu_t_corr, beta_gap = estimate_modality_gap(
                        hc_acc, mu_t_std, min_hc=min_hc, n_mix=float(hc_mix_n)
                    )
                    n_reliable_gap = (hc_acc.hc_cnt >= min_hc).sum().item()
                    print(f"    [gap] reliable_classes={n_reliable_gap}/{C}  "
                          f"delta_norm={delta_bar.norm():.4f}  "
                          f"avg_beta={beta_gap.mean():.3f}")
                    # Step3: 用修正后的 μ̂_t 重置 DOTA gap 的 mu 起点和 Sigma
                    sigma0_gap = cfg['sigma'] * torch.eye(D, dtype=torch.float32, device=device)
                    for c_idx in range(C):
                        dota_gap.mu[c_idx]    = mu_t_corr[c_idx].to(dota_gap.mu.dtype)
                        dota_gap.Sigma[c_idx] = sigma0_gap
                        # 用 HC 累积的样本数作为伪观测数（越可靠的类权重越大）
                        if hc_acc.hc_cnt[c_idx].item() >= min_hc:
                            dota_gap.c[c_idx] = hc_acc.hc_cnt[c_idx].to(dota_gap.c.dtype)
                    dota_gap.overall_Sigma = dota_gap.Sigma.mean(0)
                    dota_gap.update()
                    # Step4: 用修正后的文本权重重建 E 步分类器
                    cw_hc_gap = mu_t_corr.T.half().contiguous()   # [D, C]
                    refined["refine_hc_gap"] = True
                if not refined["refine_gap_ms"]:
                    # 均值偏移 gap 估计：delta = mean(img_feats) - mean(text_prototypes)
                    # 完全无标签，对稀疏数据集更稳定
                    # 注意：传 raw mean（未归一化），在 embedding 空间做差才有意义
                    mean_vis_now = running_visual_sum / max(running_visual_n, 1)  # [D] raw
                    mu_t_std_ms = clip_weights_std.float().T.contiguous()   # [C, D]
                    delta_ms, mu_t_corr_ms = estimate_gap_meanshift(
                        mean_vis_now, mu_t_std_ms, hc_acc=hc_acc,
                        min_hc=min_hc, n_mix=float(hc_mix_n)
                    )
                    print(f"    [gap_ms] delta_norm={delta_ms.norm():.4f}  "
                          f"mean_vis_norm={mean_vis_now.norm():.4f}")
                    sigma0_ms = cfg['sigma'] * torch.eye(D, dtype=torch.float32, device=device)
                    for c_idx in range(C):
                        dota_gap_ms.mu[c_idx]    = mu_t_corr_ms[c_idx].to(dota_gap_ms.mu.dtype)
                        dota_gap_ms.Sigma[c_idx] = sigma0_ms
                        if hc_acc.hc_cnt[c_idx].item() >= min_hc:
                            dota_gap_ms.c[c_idx] = hc_acc.hc_cnt[c_idx].to(dota_gap_ms.c.dtype)
                    dota_gap_ms.overall_Sigma = dota_gap_ms.Sigma.mean(0)
                    dota_gap_ms.update()
                    cw_gap_ms = mu_t_corr_ms.T.half().contiguous()   # [D, C]
                    refined["refine_gap_ms"] = True
                if not refined["refine_pathA"]:
                    # Same one-shot selection as refine_hc; used as starting point
                    # for Line 2 dynamic updates (EM anchor stays this frozen cw_hc)
                    clip_weights_pa = cw_hc.clone()
                    refined["refine_pathA"] = True

                if not refined["refine_hc_full"]:
                    # Text side: identical to refine_hc (HC-centroid top-k CuPL selection)
                    # Visual side: HC samples get one-hot labels in DOTA update (see per-step block)
                    cw_hc_full = refine_weights(
                        dota_hcfull, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_hc_full"] = True

                if not refined["refine_hc_phase2"]:
                    # Text side: identical to refine_hc (HC-centroid top-k CuPL selection)
                    # Visual side: switch from full-sample adEM update to HC-only soft-label update
                    # HC filter preserves DOTA's soft EM structure while removing noisy samples
                    cw_hc_phase2 = refine_weights(
                        dota_phase2, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_hc_phase2"] = True

                if not refined["refine_hc_lambda"]:
                    # HC-Λ: Bayes-optimal LDA via HC-supervised within-class scatter.
                    # Compute Λ_HC once at refine_step; use it for BOTH template
                    # scoring (here) AND prediction (per-step block below).
                    Lambda_HC = hc_acc.lambda_hc(epsilon=cfg["epsilon"])
                    cw_hc_lambda = refine_weights(
                        dota_hc, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                        lambda_override=Lambda_HC,
                    )
                    print(f"    [HC-Λ] computed at step={step}  "
                          f"n_total_HC={int(hc_acc.n_total.item())}  "
                          f"||Σ_HC||_F={torch.norm(hc_acc.m2_shared / hc_acc.n_total.clamp(min=1)).item():.3f}")
                    refined["refine_hc_lambda"] = True

                if not refined["tda_refine_hc"]:
                    # TDA + HC-refined text: use refine_hc text instead of std-only.
                    # This is the key experiment asked by the user: can HC-refined text
                    # boost TDA's cache-based prediction?  Same scoring as refine_hc but
                    # the result is consumed by TDA's pos+neg cache (not by DOTA logits).
                    cw_tda_refined = refine_weights(
                        dota_hc, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["tda_refine_hc"] = True

                # ablation M1 only: warm-start DOTA Gaussian from HC centroids.
                # TDA M1: per-step HC-gated cache updates (no trigger injection needed).
                if not refined["tda_hc_proto"]:
                    hc_mu_now = hc_acc.mu()   # [C, D] L2-normalized
                    # Warm-start dota_hcproto's visual mean from HC centroids
                    with torch.no_grad():
                        for c_idx in range(C):
                            if hc_acc.hc_cnt[c_idx].item() >= max(1, min_hc // 2):
                                dota_hcproto.mu[c_idx] = hc_mu_now[c_idx].to(dota_hcproto.mu.dtype)
                    refined["tda_hc_proto"] = True
                    refined["dota_hc_proto"] = True

                # ablation M2 only: DOTA running mean centroid -> CuPL selection (no HC filter)
                if not refined["tda_centroid_guided"]:
                    cw_tda_centroid_guided = refine_weights(
                        dota_hc, None, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    cw_dota_cg = cw_tda_centroid_guided
                    refined["tda_centroid_guided"] = True
                    refined["dota_centroid_guided"] = True

                if not refined["refine_unified"]:
                    cw_unified = refine_weights_unified(
                        dota_unified, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_unified"] = True
                if not refined["refine_adaptive"]:
                    # Compute auto top_frac from Phase-1 CuPL accuracy gain over std.
                    # CuPL genuinely better (cupl_gain >= hi_gain) → aggressive selection
                    # CuPL not better      (cupl_gain <= lo_gain) → use all (≈ adEM)
                    if p1_total > 0:
                        acc_std  = 100.0 * p1_correct_std / p1_total
                        acc_cupl = 100.0 * p1_correct_cu  / p1_total
                        cupl_gain = acc_cupl - acc_std   # percentage points
                        lo_gain, hi_gain = 0.0, 2.0
                        t = max(0.0, min(1.0, (hi_gain - cupl_gain) / max(hi_gain - lo_gain, 1e-6)))
                        auto_top_frac = top_frac + (1.0 - top_frac) * t
                    else:
                        acc_std = acc_cupl = cupl_gain = float("nan")
                    avg_H_std = p1_H_std / max(p1_n, 1)
                    avg_H_cu  = p1_H_cu  / max(p1_n, 1)
                    h_ratio   = avg_H_cu / max(avg_H_std, 1e-6)
                    print(f"    Auto top_frac: acc_std={acc_std:.1f}% acc_cu={acc_cupl:.1f}% "
                          f"cupl_gain={cupl_gain:+.2f}% | H_ratio={h_ratio:.3f} "
                          f"→ auto_tf={auto_top_frac:.3f}")
                    # Use HC-centroid scoring (same as refine_hc): at tf=1.0 the
                    # selected = all CuPL ≈ adEM, while at tf=0.10 HC centroid gives
                    # better EuroSAT (+1.65%) than unified (+0.99%).
                    cw_adapt = refine_weights(
                        dota_adapt, hc_acc, std_embs, cupl_embs, auto_top_frac, min_hc,
                        use_text_fallback=use_text_fallback,
                    )
                    refined["refine_adaptive"] = True

                if not refined["refine_adapt2"]:
                    # Same auto_top_frac logic, but when cupl_gain <= 0 switch to
                    # scoring the std templates (e.g. 80 imagenet_r templates on
                    # imagenet_a) instead of falling back to all CuPL.
                    # Lambda selects the top-10% templates most aligned with the
                    # adversarial visual distribution without using CuPL at all.
                    if p1_total > 0:
                        acc_std2  = 100.0 * p1_correct_std / p1_total
                        acc_cupl2 = 100.0 * p1_correct_cu  / p1_total
                        cupl_gain2 = acc_cupl2 - acc_std2
                        lo_gain2, hi_gain2 = 0.0, 2.0
                        t2 = max(0.0, min(1.0, (hi_gain2 - cupl_gain2) / max(hi_gain2 - lo_gain2, 1e-6)))
                        auto_top_frac2 = top_frac + (1.0 - top_frac) * t2
                    else:
                        cupl_gain2 = float("nan")
                        auto_top_frac2 = top_frac
                    if not (cupl_gain2 != cupl_gain2) and cupl_gain2 <= 0.0:
                        # CuPL is harmful: score templates, pick top top_frac,
                        # no std anchor (selected templates only, no double-counting)
                        print(f"    adapt2: cupl_gain={cupl_gain2:+.2f}% → scoring {len(std_embs[0])} templates, tf={top_frac:.2f}")
                        cw_adapt2 = refine_weights(
                            dota_adapt2, hc_acc, std_embs, std_embs, top_frac, min_hc,
                            use_text_fallback=use_text_fallback,
                            include_std_anchor=False,
                        )
                    else:
                        # CuPL helps: same as refine_adaptive
                        print(f"    adapt2: cupl_gain={cupl_gain2:+.2f}% → CuPL scoring, tf={auto_top_frac2:.3f}")
                        cw_adapt2 = refine_weights(
                            dota_adapt2, hc_acc, std_embs, cupl_embs, auto_top_frac2, min_hc,
                            use_text_fallback=use_text_fallback,
                        )
                    refined["refine_adapt2"] = True

            # ── encode once, reuse features for all weight variants ───────
            raw_feats = encode_images(images, clip_model)  # [64, D]

            # 在线视觉均值累积（无标签，用于 mean-shift gap 估计）
            bs_cur = raw_feats.size(0)
            running_visual_sum = running_visual_sum + raw_feats.float().sum(0)
            running_visual_n  += bs_cur

            feats_std, logits_std, prob_std = logits_from_features(raw_feats, clip_weights_std)
            feats_cu,  logits_cu,  prob_cu  = logits_from_features(raw_feats, clip_weights_cupl)
            # Official DOTA uses original templates (7 for imagenet_a, same as std for others).
            feats_off, logits_off, prob_off = logits_from_features(raw_feats, clip_weights_official)
            # DBR: uses dynamically updated text weights
            feats_dbr, logits_dbr, prob_dbr = logits_from_features(raw_feats, cw_dbr)

            if refined["refine_noisy"]:
                _, logits_noisy, prob_noisy = logits_from_features(raw_feats, cw_noisy)
            else:
                logits_noisy, prob_noisy = logits_cu, prob_cu

            if refined["refine_hc"]:
                _, logits_hc, prob_hc = logits_from_features(raw_feats, cw_hc)
            else:
                logits_hc, prob_hc = logits_cu, prob_cu

            if refined["refine_hc_bayes"]:
                _, logits_hc_bayes, prob_hc_bayes = logits_from_features(raw_feats, cw_hc_bayes)
            else:
                logits_hc_bayes, prob_hc_bayes = logits_cu, prob_cu

            if refined["refine_hc_share"]:
                _, logits_hc_share, prob_hc_share = logits_from_features(raw_feats, cw_hc_share)
            else:
                logits_hc_share, prob_hc_share = logits_cu, prob_cu

            if refined["refine_unified"]:
                _, logits_uni, prob_uni = logits_from_features(raw_feats, cw_unified)
            else:
                logits_uni, prob_uni = logits_cu, prob_cu

            if refined["refine_adaptive"]:
                _, logits_adapt, prob_adapt = logits_from_features(raw_feats, cw_adapt)
            else:
                logits_adapt, prob_adapt = logits_cu, prob_cu

            if refined["refine_adapt2"]:
                _, logits_adapt2, prob_adapt2 = logits_from_features(raw_feats, cw_adapt2)
            else:
                logits_adapt2, prob_adapt2 = logits_cu, prob_cu

            if refined["refine_pathA"]:
                _, logits_pa, _ = logits_from_features(raw_feats, clip_weights_pa)
            else:
                logits_pa = logits_cu

            std_ent  = _batch_entropy(prob_std)
            cu_ent   = _batch_entropy(prob_cu)
            use_std  = std_ent < cu_ent

            # Accumulate Phase-1 stats for adaptive top_frac computation
            if step < refine_step:
                p1_H_std += std_ent
                p1_H_cu  += cu_ent
                p1_n     += 1
                p1_correct_std += (logits_std.argmax(1) == target).sum().item()
                p1_correct_cu  += (logits_cu.argmax(1)  == target).sum().item()
                p1_total       += target.size(0)

            n_sel = feats_std.size(0)
            mean_feat_std = feats_std.mean(0).unsqueeze(0)
            mean_feat_cu  = feats_cu.mean(0).unsqueeze(0)
            mean_feat_off = feats_off.mean(0).unsqueeze(0)
            mean_feat_dbr = feats_dbr.mean(0).unsqueeze(0)

            # ── dota_official (official DOTA reproduction) ────────────────
            # Faithfully mirrors official DOTA:
            #   clip_logits = cosine(feat, clip_weights_official)  [original templates]
            #   dota_weights = clamp(rho * c.mean() / batch, eta)
            #   final = clip_logits + dota_weights * dota.predict(feat)
            #   M-step: prob from standard-template softmax (same as clip_logits)
            # The only missing piece vs official is test-time augmentation.
            n_sel_off = feats_off.size(0)
            dw_off = torch.clamp(cfg["rho"] * dota_official.c.mean() / n_sel_off, max=cfg["eta"])
            dl_off = dota_official.predict(mean_feat_off)
            final_off = logits_off + dw_off * dl_off
            correct["dota_official"] += (final_off.argmax(1) == target).sum().item()
            total["dota_official"]   += target.size(0)
            dota_official.fit(feats_off, prob_off.half()); dota_official.update()

            # ── adEM ─────────────────────────────────────────────────────
            dw = torch.clamp(cfg["rho"] * dota_adem.c.mean() / n_sel, max=cfg["eta"])
            dl = dota_adem.predict(mean_feat_cu)
            final = logits_cu + dw * dl
            correct["adEM"] += (final.argmax(1) == target).sum().item()
            total["adEM"]   += target.size(0)
            update_prob_adem = prob_std if use_std else prob_cu
            dota_adem.fit(feats_cu, update_prob_adem.half()); dota_adem.update()

            # ── dota_hc_proto (ablation M1 only): HC centroids warm-start DOTA Gaussian ──
            # Same CuPL text logits; DOTA visual Gaussian pre-seeded with HC centroids
            # at trigger for better visual distribution estimate (M1 without CuPL selection).
            feats_hp, logits_hp, prob_hp = feats_cu, logits_cu, prob_cu
            mean_feat_hp = mean_feat_cu
            n_sel_hp = feats_hp.size(0)
            dw_hp = torch.clamp(cfg["rho"] * dota_hcproto.c.mean() / n_sel_hp, max=cfg["eta"])
            dl_hp = dota_hcproto.predict(mean_feat_hp)
            final_hp = logits_hp + dw_hp * dl_hp
            correct["dota_hc_proto"] += (final_hp.argmax(1) == target).sum().item()
            total["dota_hc_proto"]   += target.size(0)
            update_prob_hp = prob_std if use_std else prob_hp
            dota_hcproto.fit(feats_hp, update_prob_hp.half()); dota_hcproto.update()

            # ── dota_centroid_guided (ablation M2 only): DOTA mean → CuPL selection ─
            # No HC filter: uses DOTA running mean (soft-label EM, all samples) to score CuPL.
            # Tests: centroid-guided selection without consensus filtering of centroid quality.
            if refined["dota_centroid_guided"]:
                feats_cg, logits_cg, prob_cg = logits_from_features(raw_feats, cw_dota_cg)
                mean_feat_cg = feats_cg.mean(0).unsqueeze(0)
            else:
                feats_cg, logits_cg, prob_cg = feats_cu, logits_cu, prob_cu
                mean_feat_cg = mean_feat_cu
            n_sel_cg = feats_cg.size(0)
            dw_cg = torch.clamp(cfg["rho"] * dota_cg.c.mean() / n_sel_cg, max=cfg["eta"])
            dl_cg = dota_cg.predict(mean_feat_cg)
            final_cg = logits_cg + dw_cg * dl_cg
            correct["dota_centroid_guided"] += (final_cg.argmax(1) == target).sum().item()
            total["dota_centroid_guided"]   += target.size(0)
            update_prob_cg = prob_std if use_std else prob_cg
            dota_cg.fit(feats_cg, update_prob_cg.half()); dota_cg.update()

            # ── refine_noisy ─────────────────────────────────────────────
            dw = torch.clamp(cfg["rho"] * dota_noisy.c.mean() / n_sel, max=cfg["eta"])
            dl = dota_noisy.predict(mean_feat_cu)
            final = logits_cu + dw * dl
            correct["refine_noisy"] += (final.argmax(1) == target).sum().item()
            total["refine_noisy"]   += target.size(0)
            p_noisy_refined = prob_noisy if refined["refine_noisy"] else prob_cu
            noisy_ent = _batch_entropy(p_noisy_refined)
            update_prob_noisy = prob_std if std_ent < noisy_ent else p_noisy_refined
            dota_noisy.fit(feats_cu, update_prob_noisy.half()); dota_noisy.update()

            # ── refine_hc ────────────────────────────────────────────────
            dw_hc = torch.clamp(cfg["rho"] * dota_hc.c.mean() / n_sel, max=cfg["eta"])
            dl_hc = dota_hc.predict(mean_feat_cu)
            final = logits_cu + dw_hc * dl_hc
            correct["refine_hc"] += (final.argmax(1) == target).sum().item()
            total["refine_hc"]   += target.size(0)
            p_hc_refined = prob_hc if refined["refine_hc"] else prob_cu
            hc_ent = _batch_entropy(p_hc_refined)
            update_prob_hc = prob_std if std_ent < hc_ent else p_hc_refined
            dota_hc.fit(feats_cu, update_prob_hc.half()); dota_hc.update()

            # ── refine_hc_bayes ──────────────────────────────────────────
            # Mirrors refine_hc E-step (same cw_hc_bayes triggered at 25%, same
            # entropy-min pseudo-label gating) but uses DOTABayes for the M-step,
            # which adds a text-prior shrinkage term to mu_c updates.
            dw_hcb = torch.clamp(cfg["rho"] * dota_hc_bayes.c.mean() / n_sel, max=cfg["eta"])
            dl_hcb = dota_hc_bayes.predict(mean_feat_cu)
            final = logits_cu + dw_hcb * dl_hcb
            correct["refine_hc_bayes"] += (final.argmax(1) == target).sum().item()
            total["refine_hc_bayes"]   += target.size(0)
            p_hcb_refined = prob_hc_bayes if refined["refine_hc_bayes"] else prob_cu
            hcb_ent = _batch_entropy(p_hcb_refined)
            update_prob_hcb = prob_std if std_ent < hcb_ent else p_hcb_refined
            dota_hc_bayes.fit(feats_cu, update_prob_hcb.half()); dota_hc_bayes.update()

            # ── refine_hc_share ──────────────────────────────────────────
            # Direction B: same E-step as refine_hc but per-class Lambda_c built
            # from K_t-weighted Sigma sharing in update().
            dw_hcs = torch.clamp(cfg["rho"] * dota_hc_share.c.mean() / n_sel, max=cfg["eta"])
            dl_hcs = dota_hc_share.predict(mean_feat_cu)
            final = logits_cu + dw_hcs * dl_hcs
            correct["refine_hc_share"] += (final.argmax(1) == target).sum().item()
            total["refine_hc_share"]   += target.size(0)
            p_hcs_refined = prob_hc_share if refined["refine_hc_share"] else prob_cu
            hcs_ent = _batch_entropy(p_hcs_refined)
            update_prob_hcs = prob_std if std_ent < hcs_ent else p_hcs_refined
            dota_hc_share.fit(feats_cu, update_prob_hcs.half()); dota_hc_share.update()

            # ── refine_hc_v2 ─────────────────────────────────────────────
            # 修改3：文本基础用精选权重 cw_hc_v2（25%前退化为原始 W_cupl）
            if refined["refine_hc_v2"]:
                _, logits_hc_v2, prob_hc_v2 = logits_from_features(raw_feats, cw_hc_v2)
            else:
                logits_hc_v2, prob_hc_v2 = logits_cu, prob_cu

            dw_v2 = torch.clamp(cfg["rho"] * dota_v2.c.mean() / n_sel, max=cfg["eta"])
            dl_v2 = dota_v2.predict(mean_feat_cu)
            final_v2 = logits_hc_v2 + dw_v2 * dl_v2
            correct["refine_hc_v2"] += (final_v2.argmax(1) == target).sum().item()
            total["refine_hc_v2"]   += target.size(0)

            # 修改2：HC 双一致过滤——只有两套文本分类器都同意且高置信的样本
            # 才进入 DOTA Welford 更新，替代原来"任何样本都更新"的策略
            # 理由：HC 过滤后错误率 = 0%，而熵最小门控后错误率仍 30~70%
            pred_std_v2 = prob_std.argmax(1)
            max_std_v2  = prob_std.max(1).values
            pred_cu_v2  = prob_cu.argmax(1)
            max_cu_v2   = prob_cu.max(1).values
            hc_mask_v2  = (pred_std_v2 == pred_cu_v2) & \
                          (max_std_v2 > conf_thresh) & (max_cu_v2 > conf_thresh)
            if hc_mask_v2.any():
                dota_v2.fit(feats_cu[hc_mask_v2], prob_std[hc_mask_v2].half())
                dota_v2.update()

            # ── refine_hc_v3: M1+M3，不限制更新（去掉 M2）─────────────────
            if refined["refine_hc_v3"]:
                _, logits_hc_v3, prob_hc_v3 = logits_from_features(raw_feats, cw_hc_v3)
            else:
                logits_hc_v3, prob_hc_v3 = logits_cu, prob_cu

            dw_v3 = torch.clamp(cfg["rho"] * dota_v3.c.mean() / n_sel, max=cfg["eta"])
            dl_v3 = dota_v3.predict(mean_feat_cu)
            final_v3 = logits_hc_v3 + dw_v3 * dl_v3
            correct["refine_hc_v3"] += (final_v3.argmax(1) == target).sum().item()
            total["refine_hc_v3"]   += target.size(0)
            # v3 DOTA 更新：照常用熵门控（与 adEM 相同策略，但从 mu_hc 出发）
            v3_ent = _batch_entropy(prob_hc_v3 if refined["refine_hc_v3"] else prob_cu)
            update_prob_v3 = prob_std if std_ent < v3_ent else (prob_hc_v3 if refined["refine_hc_v3"] else prob_cu)
            dota_v3.fit(feats_cu, update_prob_v3.half()); dota_v3.update()

            # ── refine_hc_v4: 软混合重置（M1_soft+M3）────────────────────
            if refined["refine_hc_v4"]:
                _, logits_hc_v4, prob_hc_v4 = logits_from_features(raw_feats, cw_hc_v4)
            else:
                logits_hc_v4, prob_hc_v4 = logits_cu, prob_cu

            dw_v4 = torch.clamp(cfg["rho"] * dota_v4.c.mean() / n_sel, max=cfg["eta"])
            dl_v4 = dota_v4.predict(mean_feat_cu)
            final_v4 = logits_hc_v4 + dw_v4 * dl_v4
            correct["refine_hc_v4"] += (final_v4.argmax(1) == target).sum().item()
            total["refine_hc_v4"]   += target.size(0)
            v4_ent = _batch_entropy(prob_hc_v4 if refined["refine_hc_v4"] else prob_cu)
            update_prob_v4 = prob_std if std_ent < v4_ent else (prob_hc_v4 if refined["refine_hc_v4"] else prob_cu)
            dota_v4.fit(feats_cu, update_prob_v4.half()); dota_v4.update()

            # ── refine_hc_gap: TTMGE 全局间隙修正 ───────────────────────
            if refined["refine_hc_gap"]:
                _, logits_gap, prob_gap = logits_from_features(raw_feats, cw_hc_gap)
            else:
                logits_gap, prob_gap = logits_cu, prob_cu

            dw_gap = torch.clamp(cfg["rho"] * dota_gap.c.mean() / n_sel, max=cfg["eta"])
            dl_gap = dota_gap.predict(mean_feat_cu)
            final_gap = logits_gap + dw_gap * dl_gap
            correct["refine_hc_gap"] += (final_gap.argmax(1) == target).sum().item()
            total["refine_hc_gap"]   += target.size(0)
            # DOTA 更新：使用修正后的文本概率做熵门控
            gap_ent = _batch_entropy(prob_gap if refined["refine_hc_gap"] else prob_cu)
            update_prob_gap = prob_std if std_ent < gap_ent else (prob_gap if refined["refine_hc_gap"] else prob_cu)
            dota_gap.fit(feats_cu, update_prob_gap.half()); dota_gap.update()

            # ── refine_gap_ms: 均值偏移 gap 修正（无标签）───────────────────
            if refined["refine_gap_ms"]:
                _, logits_ms, prob_ms = logits_from_features(raw_feats, cw_gap_ms)
            else:
                logits_ms, prob_ms = logits_cu, prob_cu

            dw_ms = torch.clamp(cfg["rho"] * dota_gap_ms.c.mean() / n_sel, max=cfg["eta"])
            dl_ms = dota_gap_ms.predict(mean_feat_cu)
            final_ms = logits_ms + dw_ms * dl_ms
            correct["refine_gap_ms"] += (final_ms.argmax(1) == target).sum().item()
            total["refine_gap_ms"]   += target.size(0)
            ms_ent = _batch_entropy(prob_ms if refined["refine_gap_ms"] else prob_cu)
            update_prob_ms = prob_std if std_ent < ms_ent else (prob_ms if refined["refine_gap_ms"] else prob_cu)
            dota_gap_ms.fit(feats_cu, update_prob_ms.half()); dota_gap_ms.update()

            # ── refine_hc_full: 精选文本 + HC硬标签混合伪标签 ──────────────────
            # E-step (prediction): 25% 后用精选文本权重 cw_hc_full（同 refine_hc）
            if refined["refine_hc_full"]:
                _, logits_hcf, prob_hcf = logits_from_features(raw_feats, cw_hc_full)
            else:
                logits_hcf, prob_hcf = logits_cu, prob_cu

            dw_hcf = torch.clamp(cfg["rho"] * dota_hcfull.c.mean() / n_sel, max=cfg["eta"])
            dl_hcf = dota_hcfull.predict(mean_feat_cu)
            final_hcf = logits_hcf + dw_hcf * dl_hcf
            correct["refine_hc_full"] += (final_hcf.argmax(1) == target).sum().item()
            total["refine_hc_full"]   += target.size(0)

            # M-step: 分层伪标签
            # HC 样本（两个分类器一致且高置信）→ one-hot 硬标签（错误率≈0%）
            # 非 HC 样本 → 熵门控软标签（同 adEM）
            pred_std_hcf = prob_std.argmax(1)
            pred_cu_hcf  = prob_cu.argmax(1)
            hc_mask_full = (pred_std_hcf == pred_cu_hcf) & \
                           (prob_std.max(1).values > conf_thresh) & \
                           (prob_cu.max(1).values > conf_thresh)
            hcf_ent = _batch_entropy(prob_hcf if refined["refine_hc_full"] else prob_cu)
            base_prob_hcf = prob_std if std_ent < hcf_ent else \
                            (prob_hcf if refined["refine_hc_full"] else prob_cu)
            update_prob_hcf = base_prob_hcf.float().clone()
            if hc_mask_full.any():
                # one-hot override for HC samples
                update_prob_hcf[hc_mask_full] = 0.0
                update_prob_hcf[hc_mask_full, pred_std_hcf[hc_mask_full]] = 1.0
            dota_hcfull.fit(feats_cu, update_prob_hcf.half()); dota_hcfull.update()

            # ── refine_hc_phase2: 精选文本 + 两阶段 HC 过滤软标签 ──────────────────
            # E-step: 25% 后用精选文本权重 cw_hc_phase2
            if refined["refine_hc_phase2"]:
                _, logits_p2, prob_p2 = logits_from_features(raw_feats, cw_hc_phase2)
            else:
                logits_p2, prob_p2 = logits_cu, prob_cu

            dw_p2 = torch.clamp(cfg["rho"] * dota_phase2.c.mean() / n_sel, max=cfg["eta"])
            dl_p2 = dota_phase2.predict(mean_feat_cu)
            final_p2 = logits_p2 + dw_p2 * dl_p2
            correct["refine_hc_phase2"] += (final_p2.argmax(1) == target).sum().item()
            total["refine_hc_phase2"]   += target.size(0)

            # M-step: Phase 1 → adEM 全样本软标签；Phase 2 → HC 过滤软标签（仅 HC 样本）
            if not refined["refine_hc_phase2"]:
                # Phase 1: same as adEM (all samples, entropy-gated soft)
                p2_ent = _batch_entropy(prob_cu)
                update_prob_p2 = prob_std if std_ent < p2_ent else prob_cu
                dota_phase2.fit(feats_cu, update_prob_p2.half()); dota_phase2.update()
            else:
                # Phase 2: HC-filtered soft labels (error≈0%, preserves soft structure)
                pred_std_p2 = prob_std.argmax(1)
                pred_cu_p2  = prob_cu.argmax(1)
                hc_mask_p2  = (pred_std_p2 == pred_cu_p2) & \
                               (prob_std.max(1).values > conf_thresh) & \
                               (prob_cu.max(1).values > conf_thresh)
                if hc_mask_p2.any():
                    p2_ent_hc = _batch_entropy(prob_p2)
                    update_prob_p2_hc = prob_std if std_ent < p2_ent_hc else prob_p2
                    dota_phase2.fit(feats_cu[hc_mask_p2],
                                    update_prob_p2_hc[hc_mask_p2].half())
                    dota_phase2.update()

            # ── refine_hc_lambda: HC-Λ for TEMPLATE SCORING ─────────────────
            # Cleanest single-variable A/B test against refine_hc:
            #   - Both share dota_hc M-step → identical visual prediction term
            #   - refine_hc:      template scoring uses dota_hc.Lambda
            #   - refine_hc_lambda: template scoring uses Λ_HC (clean within-class
            #                       scatter from HC samples with ~0% label noise)
            # If clean Σ gives more discriminative scoring directions, the
            # selected templates better align with the visual class structure.
            if refined["refine_hc_lambda"]:
                _, logits_hcl, prob_hcl = logits_from_features(raw_feats, cw_hc_lambda)
            else:
                logits_hcl, prob_hcl = logits_cu, prob_cu
            dw_hcl   = torch.clamp(
                cfg["rho"] * dota_hc.c.mean() / n_sel, max=cfg["eta"]
            )
            dl_hcl   = dota_hc.predict(mean_feat_cu)
            final_hcl = logits_hcl + dw_hcl * dl_hcl
            correct["refine_hc_lambda"] += (final_hcl.argmax(1) == target).sum().item()
            total["refine_hc_lambda"]   += target.size(0)
            # No separate M-step: dota_hc is shared with refine_hc.  Λ_HC
            # affects only the one-shot template scoring at refine_step.

            # ── refine_hc_otw: HC-Validated Online Template Weighting ──────
            # NEW PARADIGM: avoid Σ_HC entirely by directly scoring each CuPL
            # template via HC validation (positive + negative signals).
            # Templates are continuously reweighted by softmax of accumulated
            # scores, then ensembled with std anchor.  Updates EVERY step.
            #
            # Step 1: update template scores from this batch's HC samples
            hc_mask_otw = ((prob_std.argmax(1) == prob_cu.argmax(1)) &
                           (prob_std.max(1).values > conf_thresh) &
                           (prob_cu.max(1).values > conf_thresh))
            if hc_mask_otw.any():
                hc_X_b = feats_cu[hc_mask_otw]                    # [n_hc_b, D]
                hc_y_b = prob_std.argmax(1)[hc_mask_otw]          # [n_hc_b]
                update_otw_scores(otw_scores, hc_X_b, hc_y_b, cupl_embs, beta=otw_beta)

            # Step 2: rebuild cw_otw periodically (when HC count grows enough)
            #          to avoid expensive softmax every batch
            total_hc_now_otw = hc_acc.hc_cnt.sum().item()
            if total_hc_now_otw - otw_last_hc_count >= otw_update_every_n_hc:
                cw_otw = otw_weights(otw_scores, cupl_embs, std_embs, tau=otw_tau)
                otw_last_hc_count = total_hc_now_otw
                refined["refine_hc_otw"] = True

            # Step 3: predict using OTW classifier + DOTA visual term
            if refined["refine_hc_otw"]:
                _, logits_otw, prob_otw = logits_from_features(raw_feats, cw_otw)
            else:
                logits_otw, prob_otw = logits_cu, prob_cu
            dw_otw   = torch.clamp(
                cfg["rho"] * dota_hc.c.mean() / n_sel, max=cfg["eta"]
            )
            dl_otw   = dota_hc.predict(mean_feat_cu)
            final_otw = logits_otw + dw_otw * dl_otw
            correct["refine_hc_otw"] += (final_otw.argmax(1) == target).sum().item()
            total["refine_hc_otw"]   += target.size(0)

            # ── tda_entropy_cache + tda_hc_cache ─────────────────────────────
            # Two TDA-style caches differing only in admission filter:
            #   * entropy filter (TDA baseline) — low entropy preferred
            #   * HC filter (ours) — only HC samples enter
            # Logits added on top of CuPL text + DOTA visual to mirror TDA.
            #
            # One decision per image (using mean logits, like TDA does).
            sample_feat   = mean_feat_cu.squeeze(0)               # [D] for entropy/hc caches
            p_cu_mean     = logits_cu.softmax(-1).squeeze(0)       # [C]
            p_std_mean    = logits_std.softmax(-1).squeeze(0)
            pred_cu_img   = int(p_cu_mean.argmax().item())
            pred_std_img  = int(p_std_mean.argmax().item())
            max_p_cu      = float(p_cu_mean.max().item())
            max_p_std     = float(p_std_mean.max().item())
            ent_image     = -(p_cu_mean * p_cu_mean.log().clamp(min=-100)).sum().item()
            # std-template scalars for pure-TDA caches
            ent_std_image   = -(p_std_mean * p_std_mean.log().clamp(min=-100)).sum().item()
            sample_feat_std = mean_feat_std.squeeze(0)             # [D]

            # Entropy cache: every image (priority = -entropy → low entropy = better)
            entropy_cache.update(
                sample_feat, pred=pred_cu_img, priority=-float(ent_image),
            )
            # HC cache: only HC-passing images
            hc_pass = (
                (pred_std_img == pred_cu_img) and
                (max_p_std > conf_thresh) and
                (max_p_cu  > conf_thresh)
            )
            if hc_pass:
                hc_score = min(max_p_std, max_p_cu)
                hc_cache.update(
                    sample_feat, pred=pred_std_img, priority=hc_score,
                )
            # Predict with each cache
            dw_cache = torch.clamp(
                cfg["rho"] * dota_hc.c.mean() / n_sel, max=cfg["eta"]
            )
            dl_cache = dota_hc.predict(mean_feat_cu)
            # entropy-cache variant: logits_cu + dw·DOTA + entropy_cache_logits
            ent_cache_logits = entropy_cache.logits(sample_feat).unsqueeze(0)  # [1,C]
            final_ent = logits_cu + dw_cache * dl_cache + ent_cache_logits.to(logits_cu.dtype)
            correct["tda_entropy_cache"] += (final_ent.argmax(1) == target).sum().item()
            total["tda_entropy_cache"]   += target.size(0)
            # HC-cache variant: logits_cu + dw·DOTA + hc_cache_logits
            hc_cache_logits = hc_cache.logits(sample_feat).unsqueeze(0)
            final_hcc = logits_cu + dw_cache * dl_cache + hc_cache_logits.to(logits_cu.dtype)
            correct["tda_hc_cache"] += (final_hcc.argmax(1) == target).sum().item()
            total["tda_hc_cache"]   += target.size(0)

            # ── tda_baseline + tda_refine_hc: faithful TDA with optional HC text ──
            # tda_baseline:   std logits + pos cache + neg cache       (faithful TDA)
            # tda_refine_hc:  HC-refined text logits + pos cache + neg cache
            #                 (the key experiment: refine_hc plugged into TDA framework)
            #
            # Caches use TDA's exact rules: positive admitted by entropy threshold,
            # negative admitted by entropy in [0.2, 0.5] × prop_entropy (TDA normalizes
            # entropy by log(C)). We use the same prop_entropy as TDA.
            # TDA normalizes by log2(C) to map natural-log entropy into [0, 1].
            log2_C = float(torch.log2(torch.tensor(float(C))).item())
            prop_entropy_std = ent_std_image / log2_C

            # tda_baseline: pure TDA — std features, std entropy, std pred throughout
            tda_pos_baseline.update(
                sample_feat_std, loss=ent_std_image, pred=pred_std_img,
            )
            if TDA_NEG_PARAMS["entropy_lower"] < prop_entropy_std < TDA_NEG_PARAMS["entropy_upper"]:
                tda_neg_baseline.update(
                    sample_feat_std, loss=ent_std_image, pred=pred_std_img,
                    prob_map=p_std_mean,
                )

            # tda_refine_hc: Phase 1 → std features; Phase 2 → refined-text features
            if refined["tda_refine_hc"]:
                feats_tda_ref, logits_tda_ref, _ = logits_from_features(raw_feats, cw_tda_refined)
                sample_feat_tda_ref = feats_tda_ref.mean(0)
                p_tda_ref   = logits_tda_ref.softmax(-1).squeeze(0)
                pred_tda_ref = int(p_tda_ref.argmax().item())
                ent_tda_ref  = -(p_tda_ref * p_tda_ref.log().clamp(min=-100)).sum().item()
            else:
                sample_feat_tda_ref = sample_feat_std
                logits_tda_ref = logits_std
                p_tda_ref      = p_std_mean
                pred_tda_ref   = pred_std_img
                ent_tda_ref    = ent_std_image
            tda_pos_refined.update(
                sample_feat_tda_ref, loss=ent_tda_ref, pred=pred_tda_ref,
            )
            prop_ent_ref = ent_tda_ref / log2_C
            if TDA_NEG_PARAMS["entropy_lower"] < prop_ent_ref < TDA_NEG_PARAMS["entropy_upper"]:
                tda_neg_refined.update(
                    sample_feat_tda_ref, loss=ent_tda_ref, pred=pred_tda_ref,
                    prob_map=p_tda_ref,
                )

            # ─ Predict tda_baseline: std logits + pos cache + neg cache (pure TDA) ─
            pos_logits_b = tda_pos_baseline.logits(sample_feat_std).unsqueeze(0)    # [1, C]
            neg_logits_b = tda_neg_baseline.logits(
                sample_feat_std,
                neg_mask_lower=TDA_NEG_PARAMS["mask_lower"],
                neg_mask_upper=TDA_NEG_PARAMS["mask_upper"],
            ).unsqueeze(0)
            final_tda_b = logits_std + pos_logits_b.to(logits_std.dtype) - neg_logits_b.to(logits_std.dtype)
            correct["tda_baseline"] += (final_tda_b.argmax(1) == target).sum().item()
            total["tda_baseline"]   += target.size(0)

            # ─ Predict tda_refine_hc: HC-refined text + pos cache + neg cache (pure TDA) ─
            pos_logits_r = tda_pos_refined.logits(sample_feat_tda_ref).unsqueeze(0)
            neg_logits_r = tda_neg_refined.logits(
                sample_feat_tda_ref,
                neg_mask_lower=TDA_NEG_PARAMS["mask_lower"],
                neg_mask_upper=TDA_NEG_PARAMS["mask_upper"],
            ).unsqueeze(0)
            final_tda_r = logits_tda_ref + pos_logits_r.to(logits_tda_ref.dtype) - neg_logits_r.to(logits_tda_ref.dtype)
            correct["tda_refine_hc"] += (final_tda_r.argmax(1) == target).sum().item()
            total["tda_refine_hc"]   += target.size(0)

            # ── tda_hc_proto (ablation M1 only): HC-gated cache updates ────
            # Std text logits; only HC-consensus samples enter pos cache with high priority.
            # M1 contribution: cleaner visual cache vs. baseline (all samples accepted).
            sample_feat_tda_hp = sample_feat_std
            logits_tda_hp      = logits_std
            p_tda_hp           = p_std_mean
            pred_tda_hp        = pred_std_img
            ent_tda_hp         = ent_std_image
            # HC consensus check: HC samples get high priority (loss=0.01), others enter normally
            _cu_prob_m1   = prob_cu.squeeze(0)
            _pred_cu_m1   = int(_cu_prob_m1.argmax().item())
            _conf_std_m1  = p_tda_hp.max().item()
            _conf_cu_m1   = _cu_prob_m1.max().item()
            _is_hc_m1     = (pred_tda_hp == _pred_cu_m1 and
                             _conf_std_m1 > conf_thresh and
                             _conf_cu_m1  > conf_thresh)
            _loss_m1 = 0.01 if _is_hc_m1 else ent_tda_hp
            tda_pos_hc_proto.update(sample_feat_tda_hp, loss=_loss_m1, pred=pred_tda_hp)
            prop_ent_hp = ent_tda_hp / log2_C
            if TDA_NEG_PARAMS["entropy_lower"] < prop_ent_hp < TDA_NEG_PARAMS["entropy_upper"]:
                tda_neg_hc_proto.update(
                    sample_feat_tda_hp, loss=ent_tda_hp, pred=pred_tda_hp, prob_map=p_tda_hp,
                )
            pos_logits_hp = tda_pos_hc_proto.logits(sample_feat_tda_hp).unsqueeze(0)
            neg_logits_hp = tda_neg_hc_proto.logits(
                sample_feat_tda_hp,
                neg_mask_lower=TDA_NEG_PARAMS["mask_lower"],
                neg_mask_upper=TDA_NEG_PARAMS["mask_upper"],
            ).unsqueeze(0)
            final_tda_hp = logits_tda_hp + pos_logits_hp.to(logits_tda_hp.dtype) - neg_logits_hp.to(logits_tda_hp.dtype)
            correct["tda_hc_proto"] += (final_tda_hp.argmax(1) == target).sum().item()
            total["tda_hc_proto"]   += target.size(0)

            # ── tda_centroid_guided (ablation M2 only): DOTA mean → CuPL selection ──
            # No HC filter: uses DOTA running mean centroid to select CuPL templates.
            if refined["tda_centroid_guided"]:
                feats_tda_cg, logits_tda_cg, _ = logits_from_features(raw_feats, cw_tda_centroid_guided)
                sample_feat_tda_cg = feats_tda_cg.mean(0)
                p_tda_cg    = logits_tda_cg.softmax(-1).squeeze(0)
                pred_tda_cg = int(p_tda_cg.argmax().item())
                ent_tda_cg  = -(p_tda_cg * p_tda_cg.log().clamp(min=-100)).sum().item()
            else:
                sample_feat_tda_cg = sample_feat_std
                logits_tda_cg      = logits_std
                p_tda_cg           = p_std_mean
                pred_tda_cg        = pred_std_img
                ent_tda_cg         = ent_std_image
            tda_pos_centroid_guided.update(sample_feat_tda_cg, loss=ent_tda_cg, pred=pred_tda_cg)
            prop_ent_cg = ent_tda_cg / log2_C
            if TDA_NEG_PARAMS["entropy_lower"] < prop_ent_cg < TDA_NEG_PARAMS["entropy_upper"]:
                tda_neg_centroid_guided.update(
                    sample_feat_tda_cg, loss=ent_tda_cg, pred=pred_tda_cg, prob_map=p_tda_cg,
                )
            pos_logits_cg2 = tda_pos_centroid_guided.logits(sample_feat_tda_cg).unsqueeze(0)
            neg_logits_cg2 = tda_neg_centroid_guided.logits(
                sample_feat_tda_cg,
                neg_mask_lower=TDA_NEG_PARAMS["mask_lower"],
                neg_mask_upper=TDA_NEG_PARAMS["mask_upper"],
            ).unsqueeze(0)
            final_tda_cg = logits_tda_cg + pos_logits_cg2.to(logits_tda_cg.dtype) - neg_logits_cg2.to(logits_tda_cg.dtype)
            correct["tda_centroid_guided"] += (final_tda_cg.argmax(1) == target).sum().item()
            total["tda_centroid_guided"]   += target.size(0)

            # ── refine_dbr: Dynamic Bidirectional Refinement ─────────────────
            # T-step: 当 HC 累积量超过阈值时重新评分模板，更新文本权重
            total_hc_now = hc_acc.hc_cnt.sum().item()
            if (total_hc_now >= dbr_min_hc_total and
                    total_hc_now - dbr_last_hc_count >= dbr_update_every):
                cw_dbr = refine_weights(
                    dota_dbr, hc_acc, std_embs, cupl_embs, top_frac, min_hc,
                    use_text_fallback=use_text_fallback,
                )
                # 更新后需要重新计算 feats_dbr（文本权重已变）
                feats_dbr, logits_dbr, prob_dbr = logits_from_features(raw_feats, cw_dbr)
                mean_feat_dbr = feats_dbr.mean(0).unsqueeze(0)
                dbr_last_hc_count = total_hc_now
                dbr_t_step_count += 1
                print(f"    [DBR T-step #{dbr_t_step_count}] step={step}  "
                      f"total_hc={int(total_hc_now)}  avg_hc/class={total_hc_now/C:.1f}")

            # E-step
            n_sel_dbr = feats_dbr.size(0)
            dw_dbr = torch.clamp(cfg["rho"] * dota_dbr.c.mean() / n_sel_dbr, max=cfg["eta"])
            dl_dbr = dota_dbr.predict(mean_feat_dbr)
            final_dbr = logits_dbr + dw_dbr * dl_dbr
            correct["refine_dbr"] += (final_dbr.argmax(1) == target).sum().item()
            total["refine_dbr"]   += target.size(0)

            # M-step: 全样本 + 熵门控（同 adEM，保留完整覆盖率）
            # 使用 prob_dbr（动态文本权重）而非固定 prob_cu：
            #   当 cw_dbr 改进 → prob_dbr 更准 → 伪标签质量提升 → Lambda 更好
            #   → T-step 评分更准 → cw_dbr 继续改进（真正的闭环）
            dbr_ent = _batch_entropy(prob_dbr)
            update_prob_dbr = prob_std if std_ent < dbr_ent else prob_dbr
            dota_dbr.fit(feats_dbr, update_prob_dbr.half()); dota_dbr.update()

            # ── refine_unified ───────────────────────────────────────────
            dw = torch.clamp(cfg["rho"] * dota_unified.c.mean() / n_sel, max=cfg["eta"])
            dl = dota_unified.predict(mean_feat_cu)
            final = logits_cu + dw * dl
            correct["refine_unified"] += (final.argmax(1) == target).sum().item()
            total["refine_unified"]   += target.size(0)
            p_uni_refined = prob_uni if refined["refine_unified"] else prob_cu
            uni_ent = _batch_entropy(p_uni_refined)
            update_prob_uni = prob_std if std_ent < uni_ent else p_uni_refined
            dota_unified.fit(feats_cu, update_prob_uni.half()); dota_unified.update()

            # ── refine_adaptive ──────────────────────────────────────────
            dw = torch.clamp(cfg["rho"] * dota_adapt.c.mean() / n_sel, max=cfg["eta"])
            dl = dota_adapt.predict(mean_feat_cu)
            final = logits_cu + dw * dl
            correct["refine_adaptive"] += (final.argmax(1) == target).sum().item()
            total["refine_adaptive"]   += target.size(0)
            p_adapt_refined = prob_adapt if refined["refine_adaptive"] else prob_cu
            adapt_ent = _batch_entropy(p_adapt_refined)
            update_prob_adapt = prob_std if std_ent < adapt_ent else p_adapt_refined
            dota_adapt.fit(feats_cu, update_prob_adapt.half()); dota_adapt.update()

            # ── refine_adapt2 ─────────────────────────────────────────────
            dw = torch.clamp(cfg["rho"] * dota_adapt2.c.mean() / n_sel, max=cfg["eta"])
            dl = dota_adapt2.predict(mean_feat_cu)
            final = logits_cu + dw * dl
            correct["refine_adapt2"] += (final.argmax(1) == target).sum().item()
            total["refine_adapt2"]   += target.size(0)
            p_adapt2_refined = prob_adapt2 if refined["refine_adapt2"] else prob_cu
            adapt2_ent = _batch_entropy(p_adapt2_refined)
            update_prob_adapt2 = prob_std if std_ent < adapt2_ent else p_adapt2_refined
            dota_adapt2.fit(feats_cu, update_prob_adapt2.half()); dota_adapt2.update()

            # ── refine_pathA ─────────────────────────────────────────────────
            # Prediction: logits_pa (dynamic base) + same dw/dl as refine_hc
            # dw_hc/dl_hc were computed BEFORE dota_hc.update() — consistent Lambda state
            # EM supervision: reuses dota_hc (same frozen cw_hc → prob_hc) — no separate fit()
            final_pa = logits_pa + dw_hc * dl_hc
            correct["refine_pathA"] += (final_pa.argmax(1) == target).sum().item()
            total["refine_pathA"]   += target.size(0)

            # ── HC accumulator update (consensus: std and cupl must agree) ──
            hc_acc.update(feats_cu, prob_std, prob_cu)

            # ── pathA: Line 2 dynamic scoring (uses stable dota_hc.Lambda) ──
            if refined["refine_pathA"]:
                Lambda_pa  = dota_hc.Lambda.float()
                dota_mu_pa = dota_hc.mu.float()
                hc_mu_cur  = hc_acc.mu()
                hc_cnt_cur = hc_acc.hc_cnt
                W_all_pa   = _build_W_all(
                    C, D, Lambda_pa, hc_mu_cur, hc_cnt_cur, dota_mu_pa,
                    min_hc, device, std_embs, use_text_fallback)

                for c in range(C):
                    if hc_cnt_cur[c].item() < min_hc:
                        continue
                    W_c  = W_all_pa[c]
                    embs = cupl_embs[c].float()
                    M_c  = embs.size(0)
                    pos  = embs @ W_c
                    neg  = (embs @ W_all_pa.T).sum(1).sub(pos).div(max(C - 1, 1))
                    raw  = pos - neg
                    if not pa_ema_init[c]:
                        pa_ema_score[c] = raw
                        pa_ema_init[c]  = True
                    else:
                        pa_ema_score[c] = ((1 - pa_ema_alpha) * pa_ema_score[c]
                                           + pa_ema_alpha * raw)
                    k       = max(1, int(M_c * top_frac))
                    new_sel = pa_ema_score[c].topk(k).indices.sort().values
                    if pa_selection[c] is None:
                        pa_selection[c] = new_sel
                        sel_embs = embs[new_sel]
                        combined = torch.cat([std_embs[c].float(), sel_embs], dim=0)
                        clip_weights_pa[:, c] = F.normalize(combined.mean(0), dim=-1).half()
                        pa_update_count[c] += 1
                    elif not torch.equal(new_sel, pa_selection[c]):
                        old_avg = pa_ema_score[c][pa_selection[c]].mean()
                        new_avg = pa_ema_score[c][new_sel].mean()
                        if new_avg > old_avg + pa_min_score_gain:
                            pa_selection[c] = new_sel
                            sel_embs = embs[new_sel]
                            combined = torch.cat([std_embs[c].float(), sel_embs], dim=0)
                            clip_weights_pa[:, c] = F.normalize(combined.mean(0), dim=-1).half()
                            pa_update_count[c] += 1

            step += len(target)

        # print HC coverage stats
        hc_filled = int((hc_acc.hc_cnt >= min_hc).sum().item())
        hc_mean   = float(hc_acc.hc_cnt.mean().item())
        print(f"    HC stats: {hc_filled}/{C} classes have >={min_hc} HC samples "
              f"(mean={hc_mean:.1f}, conf_thresh={conf_thresh})")
        pa_scored  = sum(1 for s in pa_selection if s is not None)
        pa_updates = sum(pa_update_count)
        print(f"    [pathA] classes updated: {pa_scored}/{C} | "
              f"total weight changes: {pa_updates} | "
              f"avg changes/class: {pa_updates / max(pa_scored, 1):.1f}")
        # ── refine_hc_bayes diagnostics ────────────────────────────────
        # alpha = kappa / (kappa + n_eff) = text-prior weight in mu_pred
        with torch.no_grad():
            n_eff_final  = (dota_hc_bayes.c - dota_hc_bayes.c_init).clamp(min=0.0)
            kappa_final  = bayes_kappa_max * torch.exp(-n_eff_final / bayes_n_decay)
            alpha_final  = kappa_final / (kappa_final + n_eff_final).clamp(min=1e-8)
        print(f"    [bayes] kappa_max={bayes_kappa_max}  n_decay={bayes_n_decay} | "
              f"n_eff: mean={n_eff_final.mean().item():.1f} "
              f"min={n_eff_final.min().item():.1f} max={n_eff_final.max().item():.1f} | "
              f"alpha (text weight in pred): mean={alpha_final.mean().item():.4f} "
              f"min={alpha_final.min().item():.4f} max={alpha_final.max().item():.4f}")
        # ── refine_hc_share diagnostics ────────────────────────────────
        # alpha_c = clamp(1 - n_eff_c / n_target, 0, 1) — text-shared weight
        with torch.no_grad():
            n_eff_share = (dota_hc_share.c - dota_hc_share.c_init).clamp(min=0.0)
            alpha_share = (1.0 - n_eff_share / share_n_target).clamp(min=0.0, max=1.0)
            # K_t-softmax row entropy (diversity of sharing)
            W_share = F.softmax(dota_hc_share.K_t / share_tau, dim=1)
            share_entropy = -(W_share * (W_share + 1e-12).log()).sum(dim=1)
            share_eff_classes = share_entropy.exp()  # effective number of neighbors averaged
        print(f"    [share] tau={share_tau}  n_target={share_n_target} | "
              f"alpha (shared weight): mean={alpha_share.mean().item():.3f} "
              f"min={alpha_share.min().item():.3f} max={alpha_share.max().item():.3f} | "
              f"effective neighbors per class: mean={share_eff_classes.mean().item():.1f} "
              f"(C={C})")

    elapsed = time.time() - t_start
    accs = {k: round(100.0 * correct[k] / total[k], 2) for k in correct if total[k] > 0}
    accs["_runtime_s"] = round(elapsed, 1)
    accs["_runtime_hms"] = "%dh%02dm%02ds" % (elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60)
    print(f"  [{dataset_name}] total time: {accs['_runtime_hms']} ({elapsed:.0f}s)")
    return accs


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the CGPS paper evaluation protocol."
    )
    p.add_argument("--data-root", dest="data_root", required=True)
    p.add_argument("--datasets", dest="datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument(
        "--output", dest="output", default="outputs/cgps_seed42.json"
    )
    p.add_argument("--seed", dest="seed", type=int, default=42)
    p.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--trigger-fraction", dest="refine_frac", type=float, default=0.25)
    p.add_argument("--selection-fraction", dest="top_frac", type=float, default=0.10)
    p.add_argument(
        "--confidence-threshold", dest="conf_thresh", type=float, default=0.30
    )
    p.add_argument("--min-consensus-count", dest="min_hc", type=int, default=5)
    p.add_argument("--bayes-kappa-max", dest="bayes_kappa_max", type=float, default=10.0,
                   help="refine_hc_bayes: max text-prior pseudo-count at cold start")
    p.add_argument("--bayes-decay", dest="bayes_n_decay", type=float, default=200.0,
                   help="refine_hc_bayes: per-class samples needed for kappa to decay to 1/e")
    p.add_argument("--share-temperature", dest="share_tau", type=float, default=0.05,
                   help="refine_hc_share: softmax temperature for K_t sharing weights")
    p.add_argument("--share-target", dest="share_n_target", type=float, default=200.0,
                   help="refine_hc_share: per-class n_eff at which alpha decays to 0")
    p.add_argument("--backbone", dest="backbone", type=str, default="ViT-B/16",
                   choices=["ViT-B/16", "RN50"],
                   help="CLIP visual backbone (ViT-B/16 default; RN50 supported).")
    return p.parse_args()


def main() -> None:
    global DATA_ROOT

    args    = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    DATA_ROOT = str(data_root)
    device  = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported paper dataset(s): {', '.join(unknown)}")
    if not 0.0 < args.conf_thresh <= 1.0:
        raise ValueError("--confidence-threshold must be in (0, 1].")
    if not 0.0 < args.refine_frac <= 1.0:
        raise ValueError("--trigger-fraction must be in (0, 1].")
    if not 0.0 < args.top_frac <= 1.0:
        raise ValueError("--selection-fraction must be in (0, 1].")
    if args.min_hc < 1:
        raise ValueError("--min-consensus-count must be positive.")

    print(f"Loading CLIP {args.backbone} on {device} ...")
    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model.eval()

    all_results: dict[str, dict[str, float]] = {}
    VARIANTS = ["dota_official", "refine_hc", "tda_baseline", "tda_refine_hc"]

    for ds in datasets:
        print(f"\n{'─'*60}\nDataset: {ds}\n{'─'*60}")
        try:
            res = run_dataset(
                ds, clip_model, preprocess,
                seed=args.seed, device=device,
                refine_frac=args.refine_frac,
                top_frac=args.top_frac,
                conf_thresh=args.conf_thresh,
                min_hc=args.min_hc,
                bayes_kappa_max=args.bayes_kappa_max,
                bayes_n_decay=args.bayes_n_decay,
                share_tau=args.share_tau,
                share_n_target=args.share_n_target,
            )
            all_results[ds] = res
            for v in VARIANTS:
                print(f"    {v:<18} {res[v]:.2f}%")
        except Exception:
            print(f"  [ERROR] {ds}:")
            traceback.print_exc()

    if not all_results:
        raise RuntimeError("No dataset completed successfully.")

    # ── table ────────────────────────────────────────────────────────────────
    datasets_done = list(all_results.keys())
    cw = 12
    header = f"{'dataset':<18}" + "".join(f"{v:>{cw}}" for v in VARIANTS)
    print(f"\n{'='*70}\n{header}\n{'─'*70}")
    for ds, res in all_results.items():
        base = res["adEM"]
        row  = f"{ds:<18}"
        for v in VARIANTS:
            val = res[v]
            delta = f"({val-base:+.2f})" if v != "adEM" else ""
            row  += f"{val:>{cw-7}.2f}{delta:>7}"
        print(row)
    print("─" * 70)

    avgs = {v: float(np.mean([r[v] for r in all_results.values()])) for v in VARIANTS}
    row  = f"{'average':<18}"
    base = avgs["adEM"]
    for v in VARIANTS:
        delta = f"({avgs[v]-base:+.2f})" if v != "adEM" else ""
        row  += f"{avgs[v]:>{cw-7}.2f}{delta:>7}"
    print(row)
    print("(delta vs adEM)")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump({"results": all_results, "averages": avgs, "config": vars(args)},
                  f, indent=2)
        f.write("\n")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
