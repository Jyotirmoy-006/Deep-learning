"""
ForenSightX - Deep Learning Models Module (Iteration 2 Enhanced)
Authoritative Reference: ForenSightX_Project_Architecture.docx
Course Mapping: Neural Network & Deep Learning (PCCCS587)

Syllabus Modules:
- Module 1: ANN / Deep Feedforward Network (7D Multimodal Feature Fusion)
- Module 2: Optimization (Adam, RMSProp, SGD), Initialization (Xavier, He, Default), Regularization (Dropout, BatchNorm)
- Module 3: Visual CNN with Grad-CAM Explainability & Shape Tracing
- Module 4: BiLSTM with Concat (Blueprint 5.2 default) and Attention Pooling options
- Module 5: Class-Conditional VAE for Synthetic Feature Augmentation
"""

import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device selection utility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Module 2: Initialization, Regularization & Optimizers
# =====================================================================

# =====================================================================
# Strict Checkpoint Loading & Migration Helpers (Iteration 4)
# =====================================================================

VALID_BILSTM_KEYS = {
    "embedding.weight",
    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",
    "lstm.weight_ih_l0_reverse",
    "lstm.weight_hh_l0_reverse",
    "lstm.bias_ih_l0_reverse",
    "lstm.bias_hh_l0_reverse",
    "lstm.weight_ih_l1",
    "lstm.weight_hh_l1",
    "lstm.bias_ih_l1",
    "lstm.bias_hh_l1",
    "lstm.weight_ih_l1_reverse",
    "lstm.weight_hh_l1_reverse",
    "lstm.bias_ih_l1_reverse",
    "lstm.bias_hh_l1_reverse",
    "fc.0.weight",
    "fc.0.bias",
    "fc.3.weight",
    "fc.3.bias",
    "attention.weight",
    "attention.bias",
}


def migrate_bilstm_state_dict(
    state_dict: Dict[str, torch.Tensor], target_pooling: str = "concat"
) -> Dict[str, torch.Tensor]:
    """
    Named migration function for ConversationBiLSTM state dicts.
    Explicit Allow-List:
      Under target_pooling == 'concat', keys 'attention.weight' and 'attention.bias'
      are safely pruned because the attention head is unallocated.
      Under target_pooling == 'attention', no keys may be dropped.
    Any unauthorized discrepancy or unknown key raises a RuntimeError.
    """
    for k in state_dict.keys():
        if k not in VALID_BILSTM_KEYS:
            raise RuntimeError(f"Unexpected key in state_dict not permitted by allow-list: {k}")

    migrated = dict(state_dict)
    allowed_drop_keys = {"attention.weight", "attention.bias"}
    if target_pooling.lower() == "concat":
        for k in list(migrated.keys()):
            if k in allowed_drop_keys:
                del migrated[k]
    return migrated


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    model_type: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Central strict checkpoint loader enforcing strict=True across all models.
    Logs missing and unexpected keys; raises if non-empty after migration.
    Verifies that every loaded model parameter exactly matches checkpoint tensor.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    raw_state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state_dict, dict) and "state_dict" in raw_state_dict:
        target_state_dict = raw_state_dict["state_dict"]
    elif isinstance(raw_state_dict, dict) and "model_state_dict" in raw_state_dict:
        target_state_dict = raw_state_dict["model_state_dict"]
    else:
        target_state_dict = raw_state_dict

    m_type = model_type.lower() if model_type else model.__class__.__name__.lower()
    if "bilstm" in m_type or isinstance(model, ConversationBiLSTM):
        target_pooling = getattr(model, "pooling", "concat")
        target_state_dict = migrate_bilstm_state_dict(target_state_dict, target_pooling=target_pooling)

    # Perform strict load (strict=True)
    incompatible = model.load_state_dict(target_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed for {model.__class__.__name__}! "
            f"Missing keys: {incompatible.missing_keys}, Unexpected keys: {incompatible.unexpected_keys}"
        )

    # Tensor equality assertion: verify every parameter equals the checkpoint tensor
    for name, param in model.named_parameters():
        if name in target_state_dict:
            expected = target_state_dict[name].to(param.device)
            if not torch.equal(param.data, expected):
                raise ValueError(f"Parameter tensor mismatch after load for {name}!")

    model.to(device)
    model.eval()
    return model


def init_weights(model: nn.Module, method: str = "he_normal") -> nn.Module:
    """
    Initializes module parameters per NNDL Module 2 syllabus.
    Supported: 'he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform', 'default'
    """
    method = method.lower()
    if method == "default":
        # Framework default initialization (leave untouched)
        return model

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            if method == "he_normal":
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif method == "he_uniform":
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            elif method == "xavier_normal":
                nn.init.xavier_normal_(param)
            elif method == "xavier_uniform":
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
        elif "bias" in name:
            nn.init.zeros_(param)
    return model


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Configures training optimizer per NNDL Module 2 syllabus.
    Supported: 'adam', 'rmsprop', 'sgd'.
    """
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =====================================================================
# Module 3: CNN - Visual Evidence Analyzer
# =====================================================================

class VisualEvidenceCNN(nn.Module):
    """
    CNN architecture for visual forensic evidence (screenshots, manipulated images, documents).
    Includes per-layer shape tracing and Grad-CAM explainability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate if use_dropout else 0.0

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()

        # Conv Block 3 (Target layer for Grad-CAM)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.flatten = nn.Flatten()
        dropout_layer = nn.Dropout(p=self.dropout_rate) if use_dropout else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        conv3_out = F.relu(self.bn3(self.conv3(x)))

        if x.requires_grad or self.training:
            self.activations = conv3_out
            if conv3_out.requires_grad:
                conv3_out.register_hook(self._save_gradient)

        x = self.adaptive_pool(conv3_out)
        out = self.classifier(x)
        return out

    def _save_gradient(self, grad: torch.Tensor):
        self.gradients = grad

    def trace_shapes(self, sample_input: torch.Tensor) -> List[Dict[str, Any]]:
        """Produces per-layer shape trace for diagnostics."""
        trace = []
        x = sample_input.clone()
        trace.append({"stage": "input", "shape": list(x.shape)})

        c1 = self.conv1(x)
        trace.append({"stage": "conv1", "shape": list(c1.shape)})
        b1 = self.bn1(c1)
        r1 = F.relu(b1)
        p1 = self.pool(r1)
        trace.append({"stage": "pool1", "shape": list(p1.shape)})

        c2 = self.conv2(p1)
        trace.append({"stage": "conv2", "shape": list(c2.shape)})
        p2 = self.pool(F.relu(self.bn2(c2)))
        trace.append({"stage": "pool2", "shape": list(p2.shape)})

        c3 = self.conv3(p2)
        trace.append({"stage": "conv3", "shape": list(c3.shape)})
        ap = self.adaptive_pool(F.relu(self.bn3(c3)))
        trace.append({"stage": "adaptive_pool", "shape": list(ap.shape)})

        fl = self.flatten(ap)
        trace.append({"stage": "flatten", "shape": list(fl.shape)})
        out = self.classifier(ap)
        trace.append({"stage": "output", "shape": list(out.shape)})
        return trace

    def get_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        self.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        prob = self.forward(input_tensor)
        self.zero_grad()
        prob.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros(target_size, dtype=np.float32)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).detach().cpu()
        heatmap = F.relu(heatmap)
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        heatmap_np = heatmap.numpy()
        heatmap_tensor = torch.tensor(heatmap_np).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            heatmap_tensor, size=target_size, mode="bilinear", align_corners=False
        )
        return resized.squeeze().numpy()


# =====================================================================
# Module 4: BiLSTM - Conversation Intelligence Analyzer
# =====================================================================

class ConversationBiLSTM(nn.Module):
    """
    Bidirectional LSTM for chat and email sequence analysis.
    Supports pooling: 'concat' (Blueprint 5.2 default) and 'attention'.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.25,
        pooling: str = "concat",  # 'concat' (Blueprint 5.2 default) or 'attention'
        use_attention: Optional[bool] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling.lower()
        assert self.pooling in ["concat", "attention"], f"Unknown pooling: {self.pooling}"

        # Per Iteration 3 directive: attention layer is conditional on config/pooling
        # If unused under pooling='concat', use_attention defaults to False (0 attention params).
        if use_attention is not None:
            self.use_attention = bool(use_attention)
        else:
            self.use_attention = (self.pooling == "attention")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Attention layer is conditional: None under concat pooling unless explicitly enabled
        if self.use_attention:
            self.attention = nn.Linear(hidden_dim * 2, 1)
        else:
            self.attention = None

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)

        # Compute or fallback attention weights
        if self.attention is not None:
            attn_scores = self.attention(lstm_out)
            attn_weights = F.softmax(attn_scores, dim=1)
        else:
            # Fallback uniform attention distribution over sequence when attention layer is disabled
            seq_len = x.size(1)
            attn_weights = torch.full(
                (x.size(0), seq_len), 1.0 / max(seq_len, 1), device=x.device, dtype=torch.float32
            )

        if self.pooling == "concat":
            # Blueprint 5.2 default: concatenate final forward and backward hidden states
            # h_n shape: (num_layers * 2, batch_size, hidden_dim)
            h_forward = h_n[-2, :, :]   # Top-layer forward hidden state
            h_backward = h_n[-1, :, :]  # Top-layer backward hidden state
            context = torch.cat([h_forward, h_backward], dim=1)  # (batch_size, hidden_dim * 2)
        elif self.pooling == "attention":
            if self.attention is None:
                raise ValueError("Pooling configured as 'attention' but attention layer is disabled.")
            context = torch.sum(lstm_out * attn_weights, dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        out = self.fc(context)

        if return_attention:
            ret_attn = attn_weights.squeeze(-1) if attn_weights.dim() == 3 else attn_weights
            return out, ret_attn
        return out

    def trace_shapes(self, sample_tokens: torch.Tensor) -> List[Dict[str, Any]]:
        trace = []
        trace.append({"stage": "input", "shape": list(sample_tokens.shape)})
        emb = self.embedding(sample_tokens)
        trace.append({"stage": "embedding", "shape": list(emb.shape)})
        lstm_out, (h_n, c_n) = self.lstm(emb)
        trace.append({"stage": "lstm_out", "shape": list(lstm_out.shape)})
        trace.append({"stage": "lstm_h_n", "shape": list(h_n.shape)})

        if self.pooling == "concat":
            ctx = torch.cat([h_n[-2], h_n[-1]], dim=1)
            trace.append({"stage": "context_concat", "shape": list(ctx.shape)})
        else:
            attn = F.softmax(self.attention(lstm_out), dim=1)
            ctx = torch.sum(lstm_out * attn, dim=1)
            trace.append({"stage": "context_attention", "shape": list(ctx.shape)})

        out = self.fc(ctx)
        trace.append({"stage": "output", "shape": list(out.shape)})
        return trace

    def get_token_importance(
        self, tokens: List[str], token_ids: torch.Tensor
    ) -> List[Tuple[str, float]]:
        self.eval()
        with torch.no_grad():
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            token_ids = token_ids.to(next(self.parameters()).device)
            _, attn_weights = self.forward(token_ids, return_attention=True)
            weights = attn_weights[0].cpu().numpy()

        results = []
        for token, w in zip(tokens, weights):
            if token != "<PAD>":
                results.append((token, float(w)))
        return sorted(results, key=lambda x: x[1], reverse=True)


# =====================================================================
# Module 1: ANN - Multimodal Feature Fusion & Risk Classifier
# =====================================================================

class ANNMultimodalRiskClassifier(nn.Module):
    """
    ANN / Deep Feedforward Network for 7D Multimodal Feature Fusion.
    Authoritative Spec Reference: Section 5.3 of ForenSightX_Project_Architecture.docx
    Supports Module 2 options: Dropout (on/off), BatchNorm (on/off).
    """

    FEATURE_NAMES = [
        "cnn_score",
        "bilstm_score",
        "url_score",
        "keyword_score",
        "attachment_score",
        "timeline_anomaly",
        "email_anomaly",
    ]

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate if use_dropout else 0.0

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=self.dropout_rate) if use_dropout else nn.Identity()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=self.dropout_rate) if use_dropout else nn.Identity()

        self.out = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # In train mode, BatchNorm requires batch size > 1; fallback to identity if batch size is 1
        if self.use_batchnorm and self.training and x.size(0) == 1:
            h1 = self.drop1(self.relu1(self.fc1(x)))
            h2 = self.drop2(self.relu2(self.fc2(h1)))
        else:
            h1 = self.drop1(self.relu1(self.bn1(self.fc1(x))))
            h2 = self.drop2(self.relu2(self.bn2(self.fc2(h1))))

        return self.sigmoid(self.out(h2))

    def trace_shapes(self, sample_input: torch.Tensor) -> List[Dict[str, Any]]:
        was_training = self.training
        self.eval()
        trace = []
        if sample_input.dim() == 1:
            sample_input = sample_input.unsqueeze(0)
        trace.append({"stage": "input", "shape": list(sample_input.shape)})

        h1 = self.fc1(sample_input)
        trace.append({"stage": "fc1", "shape": list(h1.shape)})
        if self.use_batchnorm:
            h1 = self.bn1(h1)
            trace.append({"stage": "bn1", "shape": list(h1.shape)})
        h1 = self.drop1(self.relu1(h1))

        h2 = self.fc2(h1)
        trace.append({"stage": "fc2", "shape": list(h2.shape)})
        if self.use_batchnorm:
            h2 = self.bn2(h2)
            trace.append({"stage": "bn2", "shape": list(h2.shape)})
        h2 = self.drop2(self.relu2(h2))

        out = self.sigmoid(self.out(h2))
        trace.append({"stage": "output", "shape": list(out.shape)})
        if was_training:
            self.train()
        return trace

    def explain_features(self, feature_vector: torch.Tensor) -> Dict[str, float]:
        self.eval()
        if feature_vector.dim() == 1:
            x = feature_vector.unsqueeze(0)
        else:
            x = feature_vector

        device = next(self.parameters()).device
        x = x.clone().detach().to(device).requires_grad_(True)
        output = self.forward(x)
        self.zero_grad()
        output.backward(torch.ones_like(output))

        if x.grad is not None:
            grad = x.grad.squeeze().cpu().numpy()
        else:
            grad = np.ones(self.input_dim, dtype=np.float32)

        feat = x.squeeze().detach().cpu().numpy()
        attribution = np.abs(grad * feat)
        total = np.sum(attribution)
        if total > 0:
            attribution = attribution / total
        else:
            attribution = np.ones_like(attribution) / len(attribution)

        return {
            self.FEATURE_NAMES[i]: float(attribution[i])
            for i in range(len(self.FEATURE_NAMES))
        }


# =====================================================================
# Module 5: VAE - Class-Conditional Synthetic Data Augmentation
# =====================================================================

def compute_vae_parameter_count(
    input_dim: int = 7,
    num_classes: int = 2,
    hidden_dim: int = 24,
    latent_dim: int = 4,
) -> int:
    """
    Closed-form mathematical formula for SyntheticDataVAE total parameter count:
      encoder_fc:
        Linear(input_dim + num_classes, hidden_dim)       -> (input_dim + num_classes + 1) * hidden_dim
        Linear(hidden_dim, hidden_dim // 2)              -> (hidden_dim + 1) * (hidden_dim // 2)
      fc_mu & fc_logvar:
        2 * Linear(hidden_dim // 2, latent_dim)          -> 2 * (hidden_dim // 2 + 1) * latent_dim
      decoder:
        Linear(latent_dim + num_classes, hidden_dim // 2) -> (latent_dim + num_classes + 1) * (hidden_dim // 2)
        Linear(hidden_dim // 2, hidden_dim)              -> (hidden_dim // 2 + 1) * hidden_dim
        Linear(hidden_dim, input_dim)                    -> (hidden_dim + 1) * input_dim
    """
    h1 = hidden_dim
    h2 = hidden_dim // 2
    enc_params = (input_dim + num_classes + 1) * h1 + (h1 + 1) * h2
    latent_params = 2 * (h2 + 1) * latent_dim
    dec_params = (latent_dim + num_classes + 1) * h2 + (h2 + 1) * h1 + (h1 + 1) * input_dim
    return enc_params + latent_params + dec_params


class SyntheticDataVAE(nn.Module):
    """
    Class-conditional Variational Autoencoder (CVAE) for forensic feature augmentation.
    Trained strictly on the TRAIN split only. Conditioned on binary class label y in {0, 1}.
    Enforces keyword-only instantiation via '*' to eliminate positional argument bugs.
    """

    def __init__(
        self,
        *,
        input_dim: int = 7,
        num_classes: int = 2,
        hidden_dim: int = 24,
        latent_dim: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder receives feature vector + one-hot class condition
        enc_in = input_dim + num_classes
        self.encoder_fc = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder receives latent vector z + one-hot class condition
        dec_in = latent_dim + num_classes
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    @classmethod
    def parameter_formula(
        cls,
        input_dim: int = 7,
        num_classes: int = 2,
        hidden_dim: int = 24,
        latent_dim: int = 4,
    ) -> int:
        return compute_vae_parameter_count(input_dim, num_classes, hidden_dim, latent_dim)

    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        elif y.dim() > 1:
            y = y.squeeze(-1)
        y_onehot = F.one_hot(y.long(), num_classes=self.num_classes).float()
        enc_input = torch.cat([x, y_onehot], dim=1)
        h = self.encoder_fc(enc_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is None:
            y = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        elif y.dim() > 1:
            y = y.squeeze(-1)
        y_onehot = F.one_hot(y.long(), num_classes=self.num_classes).float()
        dec_input = torch.cat([z, y_onehot], dim=1)
        return self.decoder(dec_input)

    def sample(self, num_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, y)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

    @staticmethod
    def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def generate(self, num_samples: int, target_class: int = 0) -> torch.Tensor:
        """Generates synthetic multimodal feature vectors conditioned on target class."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            y = torch.full((num_samples,), target_class, dtype=torch.long, device=device)
            samples = self.decode(z, y)
        return samples


# Re-export factory functions for unified model identity pipeline
from .factory import (
    build_model,
    get_default_config,
    save_model_checkpoint,
    load_model_from_checkpoint,
    detect_checkpoint_pooling,
    count_model_parameters,
    trace_model_layer_shapes,
)
