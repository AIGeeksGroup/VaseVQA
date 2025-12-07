import torch
import platform

def get_device():
    # Priority: detect CUDA (Linux or Windows)
    if torch.cuda.is_available():
        return "auto", torch.device("cuda")
    # macOS specific MPS detection (only when system is macOS and MPS is available)
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        # return torch.device("mps")
        # mps not available
        # return "cpu", torch.device("cpu")
        # mps available
        return "mps", torch.device("mps")
    # Default fallback to CPU
    else:
        return torch.device("cpu")

if __name__=="__main__":
    device_map, device = get_device()
    print(f"Current computing device: {device_map, device}")
