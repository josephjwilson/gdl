# Stub: pretrained model loading is not used in toy/minimal training.
# load_pretrained_model is only called when args.pretrained_model_name != "none".

def load_pretrained_model(name: str):
    raise NotImplementedError(
        f"Pretrained model loading not supported in minimal/. "
        f"Requested: {name}"
    )
