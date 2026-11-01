"""
ForenSightX - Deep Learning Models Module
Authoritative Reference: ForenSightX_Project_Architecture.docx
Course Mapping: Neural Network & Deep Learning (PCCCS587)

Modules Implemented:
- Module 1: ANN / Deep Feedforward Network (Multimodal Feature Fusion & Risk Classifier)
- Module 2: Initialization (He, Xavier), Regularization (Dropout, BatchNorm), Optimizers (Adam, RMSProp, SGD)
- Module 3: CNN (Visual Evidence Analyzer) with Grad-CAM Explainability
- Module 4: BiLSTM (Conversation & Email Sequence Intelligence) with Token Attribution
- Module 5: VAE (Variational Autoencoder) for Synthetic Data Augmentation
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Module 2: Initialization, Regularization & Optimizers
# =====================================================================

def init_weights(model: nn.Module, method: str = "he_normal") -> nn.Module:
    """
    Initializes module parameters per NNDL Module 2 syllabus.
    Supported methods:
      - 'he_normal' (Kaiming normal)
      - 'he_uniform' (Kaiming uniform)
      - 'xavier_normal' (Glorot normal)
      - 'xavier_uniform' (Glorot uniform)
    """
    method = method.lower()
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
    Includes forward hooks for Grad-CAM explainability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3 (Target layer for Grad-CAM)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Dense Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

        # Gradients and activations storage for Grad-CAM
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        conv3_out = F.relu(self.bn3(self.conv3(x)))

        if x.requires_grad:
            self.activations = conv3_out
            conv3_out.register_hook(self._save_gradient)

        x = self.adaptive_pool(conv3_out)
        out = self.classifier(x)
        return out

    def _save_gradient(self, grad: torch.Tensor):
        self.gradients = grad

    def get_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_size: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """
        Computes Grad-CAM saliency map for visual evidence explainability.
        Returns:
            np.ndarray of shape (H, W) with values normalized to [0, 1].
        """
        self.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        prob = self.forward(input_tensor)
        self.zero_grad()
        prob.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if gradients unavailable
            return np.zeros(target_size, dtype=np.float32)

        # Global average pooling of gradients over spatial dimensions
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

        # Resize to target size via bilinear interpolation
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
    Captures bidirectional contextual signals to detect phishing, social engineering,
    urgency tactics, and credential requests.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Attention layer for token-level importance extraction
        self.attention = nn.Linear(hidden_dim * 2, 1)

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
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)

        # Contextual attention weights
        attn_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)

        # Weighted context representation
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_dim * 2)

        out = self.fc(context)  # (batch_size, 1)

        if return_attention:
            return out, attn_weights.squeeze(-1)
        return out

    def get_token_importance(
        self, tokens: List[str], token_ids: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Extracts explainable token importance scores using the attention mechanism.
        """
        self.eval()
        with torch.no_grad():
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
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
    ANN / Deep Feedforward Network for multimodal feature fusion.
    Authoritative Spec Reference: Section 5.3 of ForenSightX_Project_Architecture.docx
    Input Feature Vector (7 dimensions):
      [0]: cnn_score (Visual suspicion probability)
      [1]: bilstm_score (Text conversation suspicion probability)
      [2]: url_score (Suspicious URL indicator score)
      [3]: keyword_score (Suspicious forensic keyword frequency/weight)
      [4]: attachment_score (Anomalous attachment indicator)
      [5]: timeline_anomaly (Timestamp/sequence anomaly score)
      [6]: email_anomaly (Email header/domain/structure anomaly score)

    Output:
      Overall risk probability in [0, 1].
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.out = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Standard forward pass with BatchNorm
        # In eval mode, BatchNorm uses running_mean and running_var
        # In train mode, BatchNorm requires batch size > 1
        if self.training and x.size(0) == 1:
            h1 = self.drop1(self.relu1(self.fc1(x)))
            h2 = self.drop2(self.relu2(self.fc2(h1)))
        else:
            h1 = self.drop1(self.relu1(self.bn1(self.fc1(x))))
            h2 = self.drop2(self.relu2(self.bn2(self.fc2(h1))))

        return self.sigmoid(self.out(h2))

    def explain_features(self, feature_vector: torch.Tensor) -> Dict[str, float]:
        """
        Calculates feature attributions for explainability using integrated gradients / input x weight.
        """
        self.eval()
        if feature_vector.dim() == 1:
            x = feature_vector.unsqueeze(0)
        else:
            x = feature_vector

        x = x.clone().detach().requires_grad_(True)
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
# Module 5: VAE - Synthetic Data Augmentation
# =====================================================================

class SyntheticDataVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for controlled synthetic multimodal evidence augmentation.
    Learns latent distribution over forensic feature vectors and samples realistic synthetic
    evidence vectors to evaluate model robustness when forensic data are scarce.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 24,
        latent_dim: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Bounds synthetic features to normalized range [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evidence Lower Bound (ELBO) Loss: Reconstruction Loss (MSE) + KL Divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generates synthetic multimodal feature vectors by sampling from prior z ~ N(0, I).
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.decode(z)
        return samples
