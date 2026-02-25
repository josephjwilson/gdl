
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from tqdm import tqdm
import numpy as np

# Adjust Path: minimal/ is one level below the repo root where fairseq/ lives.
root_dir = os.path.dirname(os.path.abspath(__file__))  # repo root
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "fairseq"))

from graphormer.models.graphormer import GraphormerModel, GraphormerEncoder
from graphormer.data.collator import collator
from config import ModelConfig, load_train_config


def train(model_cfg, train_cfg):
    # Device selection
    if train_cfg.device:
        device = torch.device(train_cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Pure Task: {train_cfg.task} on {device}")

    # TODO: Add task implementation

    # 2. Model Config


    model_args = model_cfg.to_namespace()

    # 3. Initialize Model
    encoder = GraphormerEncoder(model_args)
    model = GraphormerModel(model_args, encoder)
    model.to(device)
    model.train()

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # 5. Loss Function & Task Head
    is_regression = (train_cfg.loss_type == "mse")
    if is_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Dynamic output dimension for mixed task (distance prediction)
    # Max SPD is N-1. Label is SPD-1 (0 to N-2). We need at least N-1 classes.
    if train_cfg.task == "mixed":
        out_dim = max(out_dim, n_nodes)

    class TaskHead(nn.Module):
        def __init__(self, dim, out_dim, readout="cls", n_nodes=12, no_cls=False):
            super().__init__()
            self.proj = nn.Linear(dim, out_dim)
            self.readout = readout
            self.n_nodes = n_nodes
            self.no_cls = no_cls
        def forward(self, x):
            # x is [B, T, C]
            # With CLS: position 0 = CLS, 1..N = nodes, rest = padding
            # Without CLS: position 0..N-1 = nodes, rest = padding
            start = 0 if self.no_cls else 1
            if self.readout == "node":
                return self.proj(x[:, start:start + self.n_nodes, :])  # [B, N, C]
            elif self.readout == "mean_pool":
                node_repr = x[:, start:start + self.n_nodes, :]
                return self.proj(node_repr.mean(dim=1))
            else:
                return self.proj(x[:, 0, :])

    # Auto-set readout for node-level tasks
    readout = train_cfg.readout
    if train_cfg.task == "broadcast" and readout == "cls":
        readout = "node"

    # Validate no_cls compatibility
    if model_cfg.no_cls and readout == "cls":
        raise ValueError("Cannot use CLS readout with no_cls=True. Use readout: mean_pool")

    head = TaskHead(model_cfg.encoder_embed_dim, out_dim, readout=readout,
                    n_nodes=n_nodes, no_cls=model_cfg.no_cls).to(device)

    optimizer.add_param_group({'params': head.parameters()})

    # DataLoader
    from types import SimpleNamespace
    def graphormer_collator(items):
        # Convert dict items to objects for collator
        obj_items = []
        for i, item in enumerate(items):
            obj = SimpleNamespace(**item)
            obj.idx = i # Collator requires 'idx'
            obj_items.append(obj)
        return collator(obj_items, max_node=model_args.max_nodes, multi_hop_max_dist=model_args.multi_hop_max_dist, spatial_pos_max=model_args.spatial_pos_max)

    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=graphormer_collator)

    # LR Scheduler (warmup + cosine decay)
    scheduler = None
    if train_cfg.warmup_frac > 0:
        from torch.optim.lr_scheduler import LambdaLR
        import math
        total_steps = len(dataloader) * train_cfg.epochs
        warmup_steps = int(total_steps * train_cfg.warmup_frac)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda)

    # Log file setup
    log_fh = None
    if train_cfg.log_file:
        log_dir = os.path.dirname(train_cfg.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fh = open(train_cfg.log_file, "a")
        log_fh.write(f"# Task={train_cfg.task} Layers={model_cfg.encoder_layers} "
                     f"Heads={model_cfg.encoder_attention_heads} Dim={model_cfg.encoder_embed_dim} "
                     f"LR={train_cfg.lr} WD={train_cfg.weight_decay} "
                     f"Samples={train_cfg.samples} BS={train_cfg.batch_size} "
                     f"Readout={readout} N={n_nodes}\n")
        log_fh.flush()

    # 9. BF16 mixed precision setup
    use_bf16 = getattr(train_cfg, 'use_bf16', False) and torch.cuda.is_available()
    if use_bf16:
        print(f"S9: BF16 mixed precision enabled")

    # 10. Training Loop
    best_loss = float('inf')

    for epoch in range(train_cfg.epochs):
        # S8: Online dataset regeneration — new graphs every epoch
        if train_cfg.online_regen and epoch > 0 and hasattr(dataset, 'regenerate'):
            dataset.regenerate(seed=train_cfg.seed + epoch)
            dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size,
                                    shuffle=True, collate_fn=graphormer_collator)

        total_loss = 0
        correct = 0
        total_samples = 0
        regression_mae = None
        if is_regression:
            regression_mae = np.zeros(out_dim)

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            # Forward (with optional BF16 autocast)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                output = model(
                    batched_data=batch,
                    perturb=None
                )

                enc_out = output # [B, T, C]
                logits = head(enc_out) # [B, Out]

                y = batch["y"]

                if is_regression:
                    # Collator does torch.cat(ys) which flattens [B, 4] to [B*4]
                    # Reshape back to [B, out_dim] for MSE loss
                    y = y.view(-1, out_dim)
                    loss = criterion(logits.float(), y)
                elif train_cfg.task == "broadcast":
                    # logits: [B, N, C], y: [B*N] from collator
                    B_actual, N_actual, C_actual = logits.shape
                    loss = criterion(logits.reshape(-1, C_actual), y)
                else:
                    loss = criterion(logits.float(), y.view(-1))

            loss.backward()

            # Metrics (outside autocast)
            with torch.no_grad():
                if is_regression:
                    y_metric = y.view(-1, out_dim)
                    mae = (logits.float() - y_metric).abs().mean(dim=0).cpu().numpy()
                    regression_mae += mae * y_metric.shape[0]
                elif train_cfg.task == "broadcast":
                    preds = logits.argmax(dim=-1).reshape(-1)
                    correct += (preds == y).sum().item()
                else:
                    preds = logits.argmax(dim=1)
                    correct += (preds == y.view(-1)).sum().item()

            if train_cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()),
                    train_cfg.gradient_clip
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n_items = y.shape[0]  # batch size
            total_loss += loss.item() * n_items
            total_samples += n_items

            avg_loss = total_loss / total_samples
            if is_regression:
                avg_mae = regression_mae / total_samples
                progress.set_postfix({"Loss": avg_loss, "MAE": avg_mae.mean()})
            else:
                acc = correct / max(1, total_samples * (n_nodes if train_cfg.task == "broadcast" else 1))
                progress.set_postfix({"Loss": avg_loss, "Acc": acc})

        # Log epoch summary
        if log_fh is not None:
            current_lr = optimizer.param_groups[0]['lr']
            if is_regression:
                avg_mae = regression_mae / total_samples
                mae_str = " ".join(f"{m:.4f}" for m in avg_mae)
                log_fh.write(f"Epoch {epoch+1}/{train_cfg.epochs}  "
                             f"Loss: {avg_loss:.6f}  MAE: [{mae_str}]  LR: {current_lr:.6f}\n")
            else:
                acc = correct / max(1, total_samples * (n_nodes if train_cfg.task == "broadcast" else 1))
                log_fh.write(f"Epoch {epoch+1}/{train_cfg.epochs}  "
                             f"Loss: {avg_loss:.4f}  Acc: {acc:.4f}  LR: {current_lr:.6f}\n")
            log_fh.flush()

        # Validation (MolHIV: track ROC-AUC each epoch)
        if val_dataset is not None:
            model.eval()
            head.eval()
            val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size,
                                     shuffle=False, collate_fn=graphormer_collator)
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    enc_out = model(batched_data=batch, perturb=None)
                    logits = head(enc_out)
                    probs = torch.softmax(logits, dim=1)[:, 1]  # P(active)
                    all_preds.append(probs.cpu())
                    all_labels.append(batch["y"].view(-1).cpu())
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
                print(f"  Epoch {epoch+1} — Val ROC-AUC: {val_auc:.4f}")
            except ValueError:
                print(f"  Epoch {epoch+1} — Val ROC-AUC: N/A (single class in batch)")
            model.train()
            head.train()

    if log_fh is not None:
        log_fh.close()

    print(f"Training Complete. Saving to {train_cfg.save_path}")

    # Ensure directory exists
    save_dir = os.path.dirname(train_cfg.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save combined checkpoint (model + head + metadata)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'model_config': asdict(model_cfg),
        'task': train_cfg.task,
        'num_classes': out_dim,
        'num_layers': model_cfg.encoder_layers,
        'num_heads': model_cfg.encoder_attention_heads,
        'pathway_mode': pathway_mode,
        'n_nodes': train_cfg.n_nodes,
        'fixed_spd_bias': model_cfg.fixed_spd_bias,
        'use_ffn': model_cfg.use_ffn,
        'readout': readout,
        'causal_mask': model_cfg.causal_mask,
        'pre_layernorm': model_cfg.pre_layernorm,
        'use_virtual_distance': model_cfg.use_virtual_distance,
        'no_cls': model_cfg.no_cls,
        'feature_vocab_size': train_cfg.feature_vocab_size,
        'online_regen': train_cfg.online_regen,
    }

    torch.save(checkpoint, train_cfg.save_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI args override YAML values)")
    parser.add_argument("--task", type=str, default="semantic",
                        choices=["positional", "semantic", "mixed", "positional_regular", "complex_mixed", "sbm", "molhiv", "broadcast", "graph_regression", "feature_regression"])
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (default 0.01, use 0.5 for sink experiments)")
    parser.add_argument("--cache_dataset", type=str, default="",
                        help="Path to save/load cached preprocessed dataset (.pt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (e.g. cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_nodes", type=int, default=None,
                        help="Fixed number of nodes per graph (default: task-specific)")
    parser.add_argument("--homophily", type=float, default=0.5,
                        help="SBM homophily level (0=ER, 1=strong communities)")
    parser.add_argument("--feature_info", type=float, default=0.75,
                        help="SBM feature informativeness (0.5=noise, 1.0=perfect)")
    parser.add_argument("--fixed_spd_bias", action="store_true",
                        help="H2: Use fixed 1/SPD bias instead of learned spatial_pos_encoder")
    parser.add_argument("--use_bf16", action="store_true",
                        help="S9: Enable BF16 mixed precision training")
    
    model_cfg, train_cfg = load_train_config(parser)

    if not train_cfg.save_path:
        parser.error("save_path must be set (via --save_path or in YAML config)")

    train(model_cfg, train_cfg)