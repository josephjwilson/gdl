"""
Config dataclasses for Graphormer models, training, and analysis.

Field names in ModelConfig match the Namespace keys expected by
GraphormerEncoder, so `to_namespace()` is a direct translation.
"""

from argparse import Namespace
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    """Architecture parameters — maps 1:1 to GraphormerEncoder's Namespace contract."""

    # Core architecture
    encoder_layers: int = 2
    encoder_embed_dim: int = 128
    encoder_ffn_embed_dim: int = 128
    encoder_attention_heads: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    act_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_normalize_before: bool = True
    pre_layernorm: bool = False
    apply_graphormer_init: bool = True
    activation_fn: str = "gelu"
    encoder_layerdrop: float = 0.0

    # Embedding table sizes
    max_nodes: int = 128
    num_atoms: int = 10240       # 512*20
    num_in_degree: int = 128
    num_out_degree: int = 128
    num_edges: int = 128
    num_spatial: int = 512
    num_edge_dis: int = 128
    edge_type: str = "multi_hop"
    multi_hop_max_dist: int = 5
    spatial_pos_max: int = 512

    # Experiment-specific toggles
    fixed_spd_bias: bool = False
    alibi_spd_bias: bool = False
    alibi_custom_slopes: Optional[List[float]] = None
    use_ffn: bool = True
    causal_mask: bool = False
    use_virtual_distance: bool = True
    use_spd_bias: bool = True       # Set False to ablate spatial_pos_encoder contribution
    no_cls: bool = False            # Remove CLS/VNode from attention
    no_degree_embedding: bool = False  # Skip degree encoders (isolate QK^T from structural info)
    use_ecc_cls_bias: bool = False      # Replace constant CLS→node bias with eccentricity-based penalty
    ecc_cls_slope: float = 1.0          # Slope m in CLS bias: t_h - m * ecc(j)
    use_farness_cls_bias: bool = False  # ALiBi-style CLS→node bias using mean distance
    use_ecc_alibi_cls_bias: bool = False  # Diameter-normalized ecc with per-head ALiBi slopes
    use_raw_ecc_alibi_cls_bias: bool = False  # Unnormalized ecc with per-head ALiBi slopes
    raw_ecc_alibi_slope_rate: float = 8.0  # Exponent in 2^(-rate/H); 8=standard, 2=shallow
    alibi_spd_slope_rate: float = 8.0  # ALiBi SPD exponent: 2^(-rate/H); 8=standard
    use_random_cls_bias: bool = False     # Control: random per-node CLS bias (same ALiBi schedule as raw_ecc)

    # Misc (required by encoder but rarely changed)
    share_encoder_input_output_embed: bool = False
    no_token_positional_embeddings: bool = False
    share_input_output_embed: bool = False
    num_classes: int = 1
    remove_head: bool = True
    export: bool = False
    traceable: bool = False
    pretrained_model_name: str = "none"

    def to_namespace(self) -> Namespace:
        """Convert to argparse.Namespace for GraphormerEncoder compatibility."""
        return Namespace(**asdict(self))

    @classmethod
    def base_preset(cls) -> "ModelConfig":
        """12L/768d/32H base model (matches pretrained Graphormer-base)."""
        return cls(
            encoder_layers=12,
            encoder_embed_dim=768,
            encoder_ffn_embed_dim=768,
            encoder_attention_heads=32,
            max_nodes=512,
            num_atoms=4608,
            num_in_degree=512,
            num_out_degree=512,
            num_edges=1536,
            spatial_pos_max=1024,
        )

    @classmethod
    def from_checkpoint(cls, raw: dict) -> "ModelConfig":
        """Reconstruct ModelConfig from a saved checkpoint dict.

        Prefers the 'model_config' key (new format). Falls back to
        legacy individual keys for older checkpoints.
        """
        if "model_config" in raw:
            # New-format checkpoint — direct reconstruction
            saved = raw["model_config"]
            # Only pass keys that are valid ModelConfig fields
            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in saved.items() if k in valid_keys}
            return cls(**filtered)

        # Legacy format — reconstruct from individual keys
        kwargs = {}
        if "num_layers" in raw:
            kwargs["encoder_layers"] = raw["num_layers"]
        if "num_heads" in raw:
            kwargs["encoder_attention_heads"] = raw["num_heads"]
        if raw.get("fixed_spd_bias", False):
            kwargs["fixed_spd_bias"] = True
        if "use_ffn" in raw:
            kwargs["use_ffn"] = raw["use_ffn"]
        return cls(**kwargs)


@dataclass
class TrainConfig:
    """Training hyperparameters and task settings."""

    task: str = "semantic"
    epochs: int = 10
    batch_size: int = 32
    samples: int = 2000
    lr: float = 1e-4
    seed: int = 42
    save_path: str = ""
    device: str = ""             # "" = auto-detect
    readout: str = "cls"         # "cls" or "mean_pool"
    n_nodes: Optional[int] = None
    homophily: float = 0.5
    feature_info: float = 0.75
    weight_decay: float = 0.01
    cache_dataset: str = ""      # Path to save/load preprocessed dataset
    zero_qkt: bool = False
    zero_bias: bool = False
    gradient_clip: float = 0.0    # Max grad norm (0 = disabled)
    warmup_frac: float = 0.0      # Fraction of total steps for linear warmup
    log_file: str = ""             # Path to epoch-level log file (empty = disabled)
    topology: str = "er"          # Topology: "er" (default), "ring", "path", "barbell"
    mark_source: bool = True        # Broadcast: random marked source (vs fixed node 0)
    online_regen: bool = False      # Regenerate dataset each epoch (S8)
    feature_vocab_size: int = 2     # Number of distinct c0 feature values (S8)
    loss_type: str = "cross_entropy"  # "cross_entropy" or "mse"
    classification_bins: Optional[int] = None  # S9b: discretize targets into bins^4 classes
    num_classes: Optional[int] = None   # Override output head size (positional scaling)
    num_feature_cols: int = 1       # Number of random feature columns per node (S9)
    use_bf16: bool = False          # BF16 mixed precision training (S9)
    graph_families: Optional[List[str]] = None  # Subset of ['er','ba','ws','regular','tree']
    target_position: str = "random"   # "random", "highest_degree", "lowest_degree"
    target_metric: str = "apl"       # "apl", "global_efficiency", or "ge_sqrt_n" (positional task)
    distance_bins: object = None  # Mixed task: [2,3,4] = stratified, "uncapped" = raw SPD, None = dataset default
    num_colors: int = 2             # Retrieval task: number of color categories
    dataset_name: str = ""           # TUDataset name (e.g., "PROTEINS", "NCI1")


@dataclass
class AnalysisConfig:
    """Analysis pipeline parameters."""

    checkpoint: str = ""
    task: str = "semantic"
    model_type: str = "toy"
    device: str = ""
    save_results: Optional[str] = None
    n_permutations: int = 10
    batch_size: int = 16
    num_samples: int = 200
    max_batches: Optional[int] = None
    degree_policy: str = "dual"
    full_matrix_metrics: bool = False
    two_axis_classification: bool = False
    bootstrap_n: int = 200
    bootstrap_alpha: float = 0.05
    ratio_thresholds: List[float] = field(default_factory=lambda: [1.1, 1.2, 1.3])
    diff_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    entropy_threshold: float = 2.0
    ablate_degrees: bool = False
    num_swaps: int = 0
    swap_weighting: bool = False
    n_nodes: Optional[int] = None
    homophily: float = 0.5
    feature_info: float = 0.75
    distance_profiles: bool = False
    readout: str = "cls"
    use_v2_classification: bool = False


@dataclass
class ExperimentConfig:
    """Top-level config combining all three sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
