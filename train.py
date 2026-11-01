"""
ForenSightX - Model Training & Experimental Harness
Authoritative Spec Reference: ForenSightX_Project_Architecture.docx Section 13
Implements Experiments A through E on a strictly held-out test set:
  - Experiment A: Text only (BiLSTM)
  - Experiment B: Image only (CNN)
  - Experiment C: Text + Image (Fused)
  - Experiment D: Text + Image + Metadata (Full 7D Multimodal ANN)
  - Experiment E: Multimodal ANN + VAE Synthetic Data Augmentation

Zero illustrative numbers used: all metrics are derived through actual execution.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models import (
    VisualEvidenceCNN,
    ConversationBiLSTM,
    ANNMultimodalRiskClassifier,
    SyntheticDataVAE,
    init_weights,
    get_optimizer,
)
from src.data_loader import (
    ForensicTokenizer,
    ForensicDatasetGenerator,
    TextEvidenceDataset,
    ImageEvidenceDataset,
    MultimodalFeatureDataset,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_binary_classifier(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes standard forensic classification metrics:
    Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def train_bilstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    epochs: int = 10,
    optimizer_name: str = "adam",
    init_method: str = "xavier_uniform",
    lr: float = 0.002,
) -> Tuple[ConversationBiLSTM, List[float]]:
    model = ConversationBiLSTM(vocab_size=vocab_size, hidden_dim=64, num_layers=2)
    init_weights(model, method=init_method)
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            preds = model(token_ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / max(1, len(train_loader)))

    return model, losses


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    optimizer_name: str = "adam",
    init_method: str = "he_normal",
    lr: float = 0.001,
) -> Tuple[VisualEvidenceCNN, List[float]]:
    model = VisualEvidenceCNN(in_channels=3, num_classes=1)
    init_weights(model, method=init_method)
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / max(1, len(train_loader)))

    return model, losses


def train_ann(
    train_loader: DataLoader,
    input_dim: int = 7,
    epochs: int = 15,
    optimizer_name: str = "adam",
    init_method: str = "he_normal",
    lr: float = 0.005,
) -> Tuple[ANNMultimodalRiskClassifier, List[float]]:
    model = ANNMultimodalRiskClassifier(input_dim=input_dim, hidden_dim1=32, hidden_dim2=16)
    init_weights(model, method=init_method)
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / max(1, len(train_loader)))

    return model, losses


def train_vae(
    train_loader: DataLoader,
    input_dim: int = 7,
    latent_dim: int = 4,
    epochs: int = 15,
    lr: float = 0.003,
) -> Tuple[SyntheticDataVAE, List[float]]:
    model = SyntheticDataVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, _ in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(feats)
            loss, _, _ = SyntheticDataVAE.vae_loss(recon, feats, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / max(1, len(train_loader)))

    return model, losses


def run_full_experiments(
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 42,
    save_models: bool = True,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Executes Experiments A, B, C, D, E per authoritative spec Section 13.
    Evaluates on identical held-out test splits.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models_saved", exist_ok=True)

    print("Generating forensic multimodal dataset for controlled experiments...")
    gen = ForensicDatasetGenerator(seed=seed)
    num_samples = 400
    texts, images, features, labels = gen.generate_multimodal_dataset(num_samples=num_samples)

    # Tokenizer fit on all texts
    tokenizer = ForensicTokenizer(max_length=80)
    tokenizer.fit_on_texts(texts)

    # Split indices: 70% Train, 15% Val, 15% Test
    n_train = int(num_samples * 0.70)
    n_val = int(num_samples * 0.15)
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, num_samples))

    # Test labels (ground truth for all experiments)
    y_test = labels[test_idx]

    # Datasets
    text_dataset = TextEvidenceDataset(texts, [int(l) for l in labels], tokenizer)
    image_dataset = ImageEvidenceDataset(images, [int(l) for l in labels])
    multimodal_dataset = MultimodalFeatureDataset(features, labels)

    # DataLoaders
    train_text_loader = DataLoader(Subset(text_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_text_loader = DataLoader(Subset(text_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_text_loader = DataLoader(Subset(text_dataset, test_idx), batch_size=batch_size, shuffle=False)

    train_img_loader = DataLoader(Subset(image_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_img_loader = DataLoader(Subset(image_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_img_loader = DataLoader(Subset(image_dataset, test_idx), batch_size=batch_size, shuffle=False)

    train_multi_loader = DataLoader(Subset(multimodal_dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_multi_loader = DataLoader(Subset(multimodal_dataset, test_idx), batch_size=batch_size, shuffle=False)

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # Experiment A: Text Only (BiLSTM)
    # -------------------------------------------------------------
    print("\n[1/5] Running Experiment A: Text Only (BiLSTM)...")
    bilstm_model, _ = train_bilstm(
        train_text_loader, val_text_loader, vocab_size=tokenizer.vocab_size, epochs=epochs
    )
    bilstm_model.eval()
    y_prob_a = []
    with torch.no_grad():
        for token_ids, _ in test_text_loader:
            preds = bilstm_model(token_ids)
            y_prob_a.extend(preds.squeeze(-1).cpu().numpy().tolist())
    y_prob_a_np = np.array(y_prob_a)
    results["Experiment_A_Text_Only"] = evaluate_binary_classifier(y_test, y_prob_a_np)

    # -------------------------------------------------------------
    # Experiment B: Image Only (CNN)
    # -------------------------------------------------------------
    print("\n[2/5] Running Experiment B: Image Only (CNN)...")
    cnn_model, _ = train_cnn(train_img_loader, val_img_loader, epochs=epochs)
    cnn_model.eval()
    y_prob_b = []
    with torch.no_grad():
        for imgs, _ in test_img_loader:
            preds = cnn_model(imgs)
            y_prob_b.extend(preds.squeeze(-1).cpu().numpy().tolist())
    y_prob_b_np = np.array(y_prob_b)
    results["Experiment_B_Image_Only"] = evaluate_binary_classifier(y_test, y_prob_b_np)

    # -------------------------------------------------------------
    # Experiment C: Text + Image (Fused 2-Feature ANN)
    # -------------------------------------------------------------
    print("\n[3/5] Running Experiment C: Text + Image (Fused)...")
    # Features [cnn_score, bilstm_score] (columns 0 and 1)
    train_feats_c = features[train_idx, :2]
    test_feats_c = features[test_idx, :2]
    train_c_loader = DataLoader(
        MultimodalFeatureDataset(train_feats_c, labels[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    ann_c_model, _ = train_ann(train_c_loader, input_dim=2, epochs=epochs + 5)
    ann_c_model.eval()
    with torch.no_grad():
        preds_c = ann_c_model(torch.tensor(test_feats_c, dtype=torch.float32))
        y_prob_c_np = preds_c.squeeze(-1).cpu().numpy()
    results["Experiment_C_Text_Plus_Image"] = evaluate_binary_classifier(y_test, y_prob_c_np)

    # -------------------------------------------------------------
    # Experiment D: Text + Image + Metadata (Full 7D ANN)
    # -------------------------------------------------------------
    print("\n[4/5] Running Experiment D: Text + Image + Metadata (Full 7D ANN)...")
    ann_d_model, _ = train_ann(train_multi_loader, input_dim=7, epochs=epochs + 5)
    ann_d_model.eval()
    with torch.no_grad():
        test_feats_d = features[test_idx]
        preds_d = ann_d_model(torch.tensor(test_feats_d, dtype=torch.float32))
        y_prob_d_np = preds_d.squeeze(-1).cpu().numpy()
    results["Experiment_D_Text_Image_Metadata"] = evaluate_binary_classifier(y_test, y_prob_d_np)

    # -------------------------------------------------------------
    # Experiment E: Multimodal + VAE Synthetic Augmentation
    # -------------------------------------------------------------
    print("\n[5/5] Running Experiment E: Multimodal + VAE Synthetic Augmentation...")
    vae_model, _ = train_vae(train_multi_loader, input_dim=7, latent_dim=4, epochs=epochs + 5)
    vae_model.eval()

    # Generate synthetic training samples (e.g., 50% augmentation of training set)
    num_synthetic = int(len(train_idx) * 0.5)
    synthetic_feats = vae_model.generate(num_samples=num_synthetic).cpu().numpy()
    # Pseudo-label synthetic samples using the current best model (or high confidence threshold)
    with torch.no_grad():
        pseudo_labels = (ann_d_model(torch.tensor(synthetic_feats, dtype=torch.float32)).squeeze().numpy() >= 0.5).astype(np.float32)

    # Augmented training set
    augmented_train_feats = np.vstack([features[train_idx], synthetic_feats])
    augmented_train_labels = np.concatenate([labels[train_idx], pseudo_labels])

    aug_train_loader = DataLoader(
        MultimodalFeatureDataset(augmented_train_feats, augmented_train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    ann_e_model, _ = train_ann(aug_train_loader, input_dim=7, epochs=epochs + 5)
    ann_e_model.eval()
    with torch.no_grad():
        preds_e = ann_e_model(torch.tensor(test_feats_d, dtype=torch.float32))
        y_prob_e_np = preds_e.squeeze(-1).cpu().numpy()
    results["Experiment_E_Multimodal_Augmented"] = evaluate_binary_classifier(y_test, y_prob_e_np)

    # Save trained models
    if save_models:
        torch.save(bilstm_model.state_dict(), "models_saved/bilstm_model.pt")
        torch.save(cnn_model.state_dict(), "models_saved/cnn_model.pt")
        torch.save(ann_d_model.state_dict(), "models_saved/ann_fusion_model.pt")
        torch.save(ann_e_model.state_dict(), "models_saved/ann_augmented_model.pt")
        torch.save(vae_model.state_dict(), "models_saved/vae_model.pt")
        # Save tokenizer vocabulary
        with open("models_saved/tokenizer_vocab.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer.word2idx, f, indent=2)

    # Save results
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Format summary table
    summary_txt = os.path.join(output_dir, "experiment_results.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        header = f"{'Experiment':<35} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for exp_name, m in results.items():
            line = f"{exp_name:<35} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['roc_auc']:<10.4f}\n"
            f.write(line)

    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION SUMMARY (Strictly Evaluated on Held-Out Test Set):")
    print("=" * 80)
    with open(summary_txt, "r", encoding="utf-8") as f:
        print(f.read())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForenSightX Training & Evaluation Suite")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for experiment outputs")
    args = parser.parse_args()

    run_full_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

