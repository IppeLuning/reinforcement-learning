import torch


def get_device(preference: str = "auto") -> str:
    """
    preference: "auto" | "cuda" | "mps" | "cpu"
    Returns a device string usable in torch.
    """
    preference = preference.lower()
    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available.")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available.")
    if preference == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
