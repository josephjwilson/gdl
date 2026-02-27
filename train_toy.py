
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
from graphormer.modules.multihead_attention import MultiheadAttention
from data.pure_tasks import get_pure_dataset
from config import ModelConfig, load_train_config

# Added the hooks
def install_zero_qkt_hooks(model, num_layers):
    hooks = []
    def zero_output_hook(module, inp, out):
        return torch.zeros_like(out)
    for module in model.modules():
        if isinstance(module, MultiheadAttention):
            hooks.append(module.q_proj.register_forward_hook(zero_output_hook))
            hooks.append(module.k_proj.register_forward_hook(zero_output_hook))
    return hooks

def install_zero_bias_hooks(model):
    import torch as _torch
    assert int(_torch.__version__.split('.')[0]) >= 2, \
        "install_zero_bias_hooks requires PyTorch >= 2.0 (with_kwargs=True support)"
    hooks = []
    def make_hook():
        def hook(m, args, kwargs):
            if kwargs.get('attn_bias') is not None:
                kwargs['attn_bias'] = torch.zeros_like(kwargs['attn_bias'])
            return args, kwargs
        return hook
    for module in model.modules():
        if isinstance(module, MultiheadAttention):
            hooks.append(module.register_forward_pre_hook(make_hook(), with_kwargs=True))
    return hooks


def train(model_cfg, train_cfg):
    # Device selection
    if train_cfg.device:
        device = torch.device(train_cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Pure Task: {train_cfg.task} on {device}")

    if train_cfg.task == "semantic":
        feature_dim = 2
    else:
        feature_dim = 20

    dataset_kwargs = dict(
        num_samples=train_cfg.samples,
        feature_dim=feature_dim,
        seed=train_cfg.seed,
    )

    # Setup a bunch of dataset kwargs
    if train_cfg.n_nodes is not None:
        dataset_kwargs['num_nodes'] = train_cfg.n_nodes
    elif train_cfg.task == "broadcast":
        dataset_kwargs['num_nodes'] = 80

    # Ensure dataset caching
    if train_cfg.task == "positional":
        if train_cfg.cache_dataset:
            dataset_kwargs['cache_path'] = train_cfg.cache_dataset
    if train_cfg.task == "semantic":
        dataset_kwargs['feature_vocab_size'] = train_cfg.feature_vocab_size
        dataset_kwargs['topology'] = train_cfg.topology
        dataset_kwargs['target_position'] = train_cfg.target_position
        if train_cfg.cache_dataset:
            dataset_kwargs['cache_path'] = train_cfg.cache_dataset
    if train_cfg.task == "mixed":
        if train_cfg.cache_dataset:
            dataset_kwargs['cache_path'] = train_cfg.cache_dataset

    _self_caching = {"semantic", "positional", "mixed"}

    if train_cfg.task in _self_caching:
        dataset = get_pure_dataset(train_cfg.task, **dataset_kwargs)
    elif train_cfg.cache_dataset and os.path.exists(train_cfg.cache_dataset):
        print(f"Loading cached dataset from {train_cfg.cache_dataset}")
        cached = torch.load(train_cfg.cache_dataset, map_location='cpu')
        dataset = get_pure_dataset(train_cfg.task, **dataset_kwargs)
        dataset.data_list = cached['data_list']
    else:
        dataset = get_pure_dataset(train_cfg.task, **dataset_kwargs)
        if train_cfg.cache_dataset:
            os.makedirs(os.path.dirname(train_cfg.cache_dataset) or '.', exist_ok=True)
            print(f"Saving dataset cache to {train_cfg.cache_dataset}")
            torch.save({'data_list': dataset.data_list, 'kwargs': dataset_kwargs}, train_cfg.cache_dataset)

    test_kwargs = dict(dataset_kwargs)
    test_kwargs['seed'] = train_cfg.seed + 10000 # Created a new OOD testset

    if test_kwargs.get('cache_path'):
        base, ext = os.path.splitext(test_kwargs['cache_path'])
        test_kwargs['cache_path'] = f"{base}_test{ext}"
    if train_cfg.task in _self_caching:
        test_dataset = get_pure_dataset(train_cfg.task, **test_kwargs)
    elif train_cfg.cache_dataset:
        base, ext = os.path.splitext(train_cfg.cache_dataset)
        test_cache = f"{base}_test{ext}"
        if os.path.exists(test_cache):
            print(f"Loading cached test dataset from {test_cache}")
            cached = torch.load(test_cache, map_location='cpu')
            test_dataset = get_pure_dataset(train_cfg.task, **test_kwargs)
            test_dataset.data_list = cached['data_list']
        else:
            test_dataset = get_pure_dataset(train_cfg.task, **test_kwargs)
            os.makedirs(os.path.dirname(test_cache) or '.', exist_ok=True)
            print(f"Saving test dataset cache to {test_cache}")
            torch.save({'data_list': test_dataset.data_list, 'kwargs': test_kwargs}, test_cache)
    else:
        test_dataset = get_pure_dataset(train_cfg.task, **test_kwargs)

    # Add for Vocab style, beyond regression
    if train_cfg.task == "semantic" and train_cfg.feature_vocab_size > 2:
        model_cfg.num_atoms = train_cfg.feature_vocab_size + 515

    model_args = model_cfg.to_namespace()

    # Initialise Model
    encoder = GraphormerEncoder(model_args)
    model = GraphormerModel(model_args, encoder)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Loss Function
    is_regression = (train_cfg.loss_type == "mse")
    if is_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    num_classes_map = {
        "positional": 12,
        "semantic": 2,
    }

    out_dim = num_classes_map[train_cfg.task]

    if train_cfg.num_classes is not None:
        out_dim = train_cfg.num_classes

    # For expanded semantic
    if train_cfg.task == "semantic" and train_cfg.feature_vocab_size > 2:
        out_dim = train_cfg.feature_vocab_size

    # Determine number of graph nodes for mean-pool readout
    if train_cfg.n_nodes is not None:
        n_nodes = train_cfg.n_nodes
    else:
        n_nodes = 12

    class TaskHead(nn.Module):
        def __init__(self, dim, out_dim, readout="cls", n_nodes=12, no_cls=False):
            super().__init__()
            self.proj = nn.Linear(dim, out_dim)
            self.readout = readout
            self.n_nodes = n_nodes
            self.no_cls = no_cls
        def forward(self, x):
            # x is [B, T, C]
            # With CLS: position 0 = CLS, 1..N = nodes
            # Without CLS: position 0..N-1 = nodes
            start = 0 if self.no_cls else 1
            if self.readout == "node":
                return self.proj(x[:, start:start + self.n_nodes, :])
            elif self.readout == "mean_pool": # Added an optional mean-pooling readout (not used now)
                node_repr = x[:, start:start + self.n_nodes, :]
                return self.proj(node_repr.mean(dim=1))
            else:
                return self.proj(x[:, 0, :])

    readout = train_cfg.readout

    # Validate no_cls compatibility
    if model_cfg.no_cls and readout == "cls":
        raise ValueError("Cannot use CLS readout with no_cls=True. Use readout: mean_pool")

    head = TaskHead(model_cfg.encoder_embed_dim, out_dim, readout=readout,
                    n_nodes=n_nodes, no_cls=model_cfg.no_cls).to(device)

    optimizer.add_param_group({'params': head.parameters()})

    pathway_hooks = []
    pathway_mode = "full"
    if train_cfg.zero_qkt:
        pathway_hooks = install_zero_qkt_hooks(model, model_cfg.encoder_layers)
        pathway_mode = "bias_only"
        print(f"P10b: Installed zero-QK^T hooks ({len(pathway_hooks)} hooks). "
              f"Training with bias-only attention.")
    elif train_cfg.zero_bias:
        pathway_hooks = install_zero_bias_hooks(model)
        pathway_mode = "qkt_only"
        print(f"P10c: Installed zero-bias hooks ({len(pathway_hooks)} hooks). "
              f"Training with QK^T-only attention.")

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

    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=train_cfg.batch_size,
                                     shuffle=False, collate_fn=graphormer_collator)

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

    # Important for training speed: BF16 mixed precision setup
    use_bf16 = getattr(train_cfg, 'use_bf16', False) and torch.cuda.is_available()
    if use_bf16:
        print(f"BF16 mixed precision enabled")

    # Training Loop
    best_loss = float('inf')

    for epoch in range(train_cfg.epochs):
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
                    y = y.view(-1, out_dim).float()
                    loss = criterion(logits.float(), y)
                else:
                    loss = criterion(logits.float(), y.view(-1).long())

            loss.backward()

            # Metrics (outside autocast)
            with torch.no_grad():
                if is_regression:
                    y_metric = y.view(-1, out_dim)
                    mae = (logits.float() - y_metric).abs().mean(dim=0).cpu().numpy()
                    regression_mae += mae * y_metric.shape[0]
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
                acc = correct / max(1, total_samples)
                progress.set_postfix({"Loss": avg_loss, "Acc": acc})

        # Test evaluation
        test_acc = None
        test_mae = None
        if test_dataloader is not None:
            model.eval()
            head.eval()
            test_correct = 0
            test_total = 0
            test_regression_mae = np.zeros(out_dim) if is_regression else None
            with torch.no_grad():
                for tbatch in test_dataloader:
                    for k, v in tbatch.items():
                        if isinstance(v, torch.Tensor):
                            tbatch[k] = v.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        t_out = model(batched_data=tbatch, perturb=None)
                        t_logits = head(t_out)
                    t_y = tbatch["y"]
                    if is_regression:
                        t_y = t_y.view(-1, out_dim).float()
                        t_mae = (t_logits.float() - t_y).abs().mean(dim=0).cpu().numpy()
                        test_regression_mae += t_mae * t_y.shape[0]
                        test_total += t_y.shape[0]
                    else:
                        t_preds = t_logits.argmax(dim=1)
                        test_correct += (t_preds == t_y.view(-1)).sum().item()
                        test_total += t_y.shape[0]
            if is_regression:
                test_mae = test_regression_mae / test_total
            else:
                test_acc = test_correct / max(1, test_total * (n_nodes if train_cfg.task == "broadcast" else 1))
            model.train()
            head.train()

        # Log epoch summary
        if log_fh is not None:
            current_lr = optimizer.param_groups[0]['lr']
            if is_regression:
                avg_mae = regression_mae / total_samples
                mae_str = " ".join(f"{m:.4f}" for m in avg_mae)
                test_mae_str = ""
                if test_mae is not None:
                    test_mae_str = "  TestMAE: [" + " ".join(f"{m:.4f}" for m in test_mae) + "]"
                log_fh.write(f"Epoch {epoch+1}/{train_cfg.epochs}  "
                             f"Loss: {avg_loss:.6f}  MAE: [{mae_str}]{test_mae_str}  LR: {current_lr:.6f}\n")
            else:
                acc = correct / max(1, total_samples * (n_nodes if train_cfg.task == "broadcast" else 1))
                test_str = ""
                if test_acc is not None:
                    test_str = f"  TestAcc: {test_acc:.4f}"
                log_fh.write(f"Epoch {epoch+1}/{train_cfg.epochs}  "
                             f"Loss: {avg_loss:.4f}  Acc: {acc:.4f}{test_str}  LR: {current_lr:.6f}\n")
            log_fh.flush()

    if log_fh is not None:
        log_fh.close()

    print(f"Training Complete. Saving to {train_cfg.save_path}")

    save_dir = os.path.dirname(train_cfg.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save combined checkpoint (model + head + metadata)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'model_config': asdict(model_cfg),
        # Legacy fields for backward compat with older analysis scripts
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
        'test_acc': test_acc,
        'test_mae': test_mae.tolist() if test_mae is not None else None,
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
    parser.add_argument("--fixed_spd_bias", action="store_true",
                        help="Use fixed 1/SPD bias instead of learned spatial_pos_encoder")
    parser.add_argument("--zero_qkt", action="store_true",
                        help="Zero QK^T during training (bias-only pathway)")
    parser.add_argument("--zero_bias", action="store_true",
                        help="Zero attn_bias during training (QK^T-only pathway)")
    parser.add_argument("--no_ffn", action="store_true",
                        help="Disable FFN blocks in encoder layers")
    parser.add_argument("--readout", type=str, default="cls", choices=["cls", "mean_pool", "node"],
                        help="Readout mode: cls (graph token), mean_pool (average node representations), or node (per-node prediction)")
    parser.add_argument("--gradient_clip", type=float, default=0.0,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--warmup_frac", type=float, default=0.0,
                        help="Fraction of total steps for linear LR warmup (0 = disabled)")
    parser.add_argument("--causal_mask", action="store_true",
                        help="Enable causal (upper-triangular) attention mask")
    parser.add_argument("--pre_layernorm", action="store_true",
                        help="Use pre-LayerNorm (unconstrained residual stream)")
    parser.add_argument("--no_virtual_distance", action="store_true",
                        help="Disable graph_token_virtual_distance CLS bias")
    parser.add_argument("--no_spd_bias", action="store_true",
                        help="Ablate spatial_pos_encoder: zero out node-node SPD attention bias")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to epoch-level log file (real-time monitoring via tail -f)")
    parser.add_argument("--topology", type=str, default="er",
                        choices=["er", "ring", "path", "barbell", "random_tree"],
                        help="Topology (default: er)")
    parser.add_argument("--mark_source", action="store_true",
                        help="Broadcast: use random marked source node (vs fixed node 0)")
    parser.add_argument("--online_regen", action="store_true",
                        help="Regenerate dataset each epoch (online data diversity)")
    parser.add_argument("--feature_vocab_size", type=int, default=2,
                        help="Number of distinct c0 feature values for semantic task (default: 2)")
    parser.add_argument("--loss_type", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"],
                        help="Loss type (cross_entropy or mse)")
    parser.add_argument("--classification_bins", type=int, default=None,
                        help="Discretize each target into this many bins (bins^4 total classes)")
    parser.add_argument("--num_feature_cols", type=int, default=1,
                        help="Number of random feature columns per node")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Enable BF16 mixed precision training")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override output head size (positional scaling)")

    # First pass: check for mutual exclusion before config loading
    args = parser.parse_args()
    if args.zero_qkt and args.zero_bias:
        parser.error("Cannot use --zero_qkt and --zero_bias together")
    if not args.save_path and not args.config:
        parser.error("--save_path is required (or use --config with save_path in YAML)")

    # Load config with YAML + CLI merging
    model_cfg, train_cfg = load_train_config(parser)

    if not train_cfg.save_path:
        parser.error("save_path must be set (via --save_path or in YAML config)")

    train(model_cfg, train_cfg)
