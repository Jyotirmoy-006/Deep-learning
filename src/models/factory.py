"""
src/models/factory.py
Iteration 5 Canonical Model Factory and Checkpoint Management Path.
Builds every model from a config dict only. Checkpoints store their config;
the loader rebuilds from it. All traces, counts, and audits go through this path.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Canonical Default Configurations
# =====================================================================

def get_default_config(model_type: str, **kwargs) -> Dict[str, Any]:
    """Returns canonical default configuration dictionary for a given model type."""
    t = model_type.lower()
    if t in ["cnn", "visualevidencecnn"]:
        cfg = {
            "model_type": "cnn",
            "architecture": "VisualEvidenceCNN",
            "in_channels": 3,
            "num_classes": 1,
            "dropout_rate": 0.3,
            "use_batchnorm": True,
            "use_dropout": True,
            "input_resolution": [128, 128],
        }
    elif t in ["bilstm", "conversationbilstm"]:
        cfg = {
            "model_type": "bilstm",
            "architecture": "ConversationBiLSTM",
            "vocab_size": 256,
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout_rate": 0.25,
            "pooling": "concat",  # Blueprint 5.2 default
            "max_length": 80,
        }
    elif t in ["ann", "annmultimodalriskclassifier"]:
        cfg = {
            "model_type": "ann",
            "architecture": "ANNMultimodalRiskClassifier",
            "input_dim": 7,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout_rate": 0.2,
            "use_batchnorm": True,
        }
    elif t in ["vae", "syntheticdatavae"]:
        cfg = {
            "model_type": "vae",
            "architecture": "SyntheticDataVAE",
            "input_dim": 7,
            "latent_dim": 4,
            "hidden_dim": 32,
            "num_classes": 2,
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cfg.update(kwargs)
    return cfg


# =====================================================================
# Factory Builder
# =====================================================================

def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Builds a model strictly from its configuration dictionary.
    All architectural hyperparameters originate from config.
    """
    m_type = config.get("model_type", "").lower()
    arch = config.get("architecture", "")

    if m_type == "cnn" or arch == "VisualEvidenceCNN":
        return VisualEvidenceCNN(
            in_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 1),
            dropout_rate=config.get("dropout_rate", 0.3),
            use_batchnorm=config.get("use_batchnorm", True),
            use_dropout=config.get("use_dropout", True),
        )
    elif m_type == "bilstm" or arch == "ConversationBiLSTM":
        return ConversationBiLSTM(
            vocab_size=config.get("vocab_size", 256),
            embed_dim=config.get("embed_dim", 64),
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.25),
            pooling=config.get("pooling", "concat"),
        )
    elif m_type == "ann" or arch == "ANNMultimodalRiskClassifier":
        return ANNMultimodalRiskClassifier(
            input_dim=config.get("input_dim", 7),
            hidden_dim1=config.get("hidden_dim1", 64),
            hidden_dim2=config.get("hidden_dim2", 32),
            dropout_rate=config.get("dropout_rate", 0.2),
            use_batchnorm=config.get("use_batchnorm", True),
        )
    elif m_type == "vae" or arch == "SyntheticDataVAE":
        return SyntheticDataVAE(
            input_dim=config.get("input_dim", 7),
            latent_dim=config.get("latent_dim", 4),
            hidden_dim=config.get("hidden_dim", 32),
            num_classes=config.get("num_classes", 2),
        )
    else:
        raise ValueError(f"Unsupported model specification in config: {config}")


# =====================================================================
# Checkpoint Saving & Loading
# =====================================================================

def save_model_checkpoint(
    model: nn.Module,
    config: Dict[str, Any],
    filepath: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves model checkpoint enclosing its configuration dictionary and state_dict.
    Guarantees that checkpoints are self-describing.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "metadata": extra_metadata or {},
    }
    torch.save(payload, filepath)


def detect_checkpoint_pooling(state_dict: Dict[str, torch.Tensor]) -> str:
    """Detects whether a BiLSTM state_dict was trained with attention or concat pooling."""
    if "attention.weight" in state_dict or any("attention" in k for k in state_dict):
        return "attention"
    return "concat"


def load_model_from_checkpoint(
    filepath: str,
    device: Optional[torch.device] = None,
    expected_pooling: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Rebuilds model from the configuration stored in the checkpoint.
    Loading into a different pooling mode raises ValueError (no silent key pruning).
    """
    if device is None:
        device = DEVICE

    raw_data = torch.load(filepath, map_location=device)

    if isinstance(raw_data, dict) and "config" in raw_data and "state_dict" in raw_data:
        config = raw_data["config"]
        state_dict = raw_data["state_dict"]
    elif isinstance(raw_data, dict) and "model_state_dict" in raw_data:
        config = raw_data.get("config", {})
        state_dict = raw_data["model_state_dict"]
    elif isinstance(raw_data, dict):
        # Legacy raw state_dict without config payload
        state_dict = raw_data
        config = infer_config_from_legacy_state_dict(state_dict, filepath)
    else:
        raise ValueError(f"Unrecognized checkpoint format at {filepath}: {type(raw_data)}")

    # Strict BiLSTM Pooling Invariance Check (Part 1c)
    if config.get("model_type") == "bilstm" or config.get("architecture") == "ConversationBiLSTM":
        ckpt_pooling = detect_checkpoint_pooling(state_dict)
        cfg_pooling = config.get("pooling", ckpt_pooling)

        if expected_pooling is not None and expected_pooling.lower() != ckpt_pooling:
            raise ValueError(
                f"FATAL POOLING MISMATCH: Checkpoint '{filepath}' was trained with pooling='{ckpt_pooling}', "
                f"but caller requested pooling='{expected_pooling}'. Cross-pooling loading is strictly prohibited "
                f"(no silent key pruning permitted)."
            )

        if cfg_pooling != ckpt_pooling:
            raise ValueError(
                f"FATAL CONFIG/CHECKPOINT POOLING MISMATCH: Config specifies pooling='{cfg_pooling}' "
                f"but checkpoint weights indicate pooling='{ckpt_pooling}'."
            )

    model = build_model(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, config


def infer_config_from_legacy_state_dict(
    state_dict: Dict[str, torch.Tensor], filepath: str
) -> Dict[str, Any]:
    """Infers canonical config for legacy checkpoints missing explicit 'config' dict."""
    fn = os.path.basename(filepath).lower()
    if "cnn" in fn or "conv1.weight" in state_dict:
        cfg = get_default_config("cnn")
    elif "bilstm" in fn or "embedding.weight" in state_dict:
        pooling = detect_checkpoint_pooling(state_dict)
        vocab_size = state_dict["embedding.weight"].shape[0]
        cfg = get_default_config("bilstm", vocab_size=vocab_size, pooling=pooling)
    elif "ann" in fn or "fc1.weight" in state_dict:
        input_dim = state_dict["fc1.weight"].shape[1]
        cfg = get_default_config("ann", input_dim=input_dim)
    elif "vae" in fn or "fc_mu.weight" in state_dict:
        cfg = get_default_config("vae")
    else:
        raise ValueError(f"Cannot infer model config from state_dict keys: {list(state_dict.keys())[:5]}")
    return cfg


# =====================================================================
# Canonical Tracing and Parameter Audits
# =====================================================================

def count_model_parameters(model_or_config: Union[nn.Module, Dict[str, Any]]) -> Dict[str, int]:
    """Computes total, trainable, and non-trainable parameter counts."""
    if isinstance(model_or_config, dict):
        model = build_model(model_or_config)
    else:
        model = model_or_config

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def trace_model_layer_shapes(
    model_or_config: Union[nn.Module, Dict[str, Any]], sample_input: torch.Tensor
) -> List[Dict[str, Any]]:
    """Generates per-layer shape traces using the unified model factory path."""
    if isinstance(model_or_config, dict):
        model = build_model(model_or_config)
    else:
        model = model_or_config

    if hasattr(model, "trace_shapes"):
        return model.trace_shapes(sample_input)

    # Generic forward hook tracer
    traces = [{"stage": "input", "shape": list(sample_input.shape)}]
    hooks = []

    def hook_fn(name):
        def _fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                s = list(out.shape)
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                s = list(out[0].shape)
            else:
                s = str(type(out))
            traces.append({"stage": name, "shape": s})
        return _fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        if isinstance(model, SyntheticDataVAE):
            model(sample_input)
        else:
            model(sample_input)

    for h in hooks:
        h.remove()

    return traces

