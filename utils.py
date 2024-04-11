import random, torch, os, numpy as np
import config

# utils.py contains utility functions that are used throughout the training process.
# These include saving/loading checkpoints, seeding for reproducibility, and more.

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Saves the current state of the model and optimizer to a file.
    
    Parameters:
    - model: The model being trained.
    - optimizer: The optimizer used during training.
    - filename: The path where the checkpoint will be saved.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),  # Save the model state
        "optimizer": optimizer.state_dict(),  # Save the optimizer state
    }
    torch.save(checkpoint, filename)  # Write the checkpoint to a file

def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    """
    Loads the model and optimizer state from a checkpoint file.
    
    Parameters:
    - checkpoint_file: The path to the checkpoint file.
    - model: The model that the state will be loaded into.
    - optimizer: The optimizer that the state will be loaded into (if provided).
    - lr: The new learning rate to set for the optimizer (if provided).
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)  # Load the checkpoint
    model.load_state_dict(checkpoint["state_dict"])  # Restore the model state

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])  # Restore the optimizer state
        if lr:
            # If a new learning rate was provided, update the optimizer with this new lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

def seed_everything(seed=42):
    """
    Seeds all the necessary components to ensure reproducibility. By fixing the seed,
    we ensure that random operations produce the same results each run, which is essential
    for debugging and comparing model performance across different training sessions.
    
    Parameters:
    - seed: The seed value to use for all random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set PYTHONHASHSEED environment variable
    random.seed(seed)  # Seed the random number generator from the standard Python library
    np.random.seed(seed)  # Seed the random number generator from NumPy
    torch.manual_seed(seed)  # Seed the random number generator from PyTorch for CPU
    torch.cuda.manual_seed(seed)  # Seed the random number generator from PyTorch for current CUDA device
    torch.cuda.manual_seed_all(seed)  # Seed the random number generator from PyTorch for all CUDA devices
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disabling the benchmarking feature that can speed up training
