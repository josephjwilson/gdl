# Stripped-down graphormer package init.
# Removes criterions/tasks imports (require fairseq on sys.path at import time).

try:
    import torch
    torch.multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass
