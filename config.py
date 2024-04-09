import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# config.py is used to set up the configuration and hyperparameters for the CycleGAN training.
# It centralizes all settings so that they can be easily modified and accessed throughout the project.

# DEVICE configuration will use GPU if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories for training and validation datasets.
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# Batch size for training; you might need to adjust this depending on your GPU memory.
BATCH_SIZE = 1

# Learning rate for optimizers.
LEARNING_RATE = 1e-5

# Lambda identity and lambda cycle are hyperparameters for the loss calculations in CycleGAN.
# They control the relative importance of different loss components.
LAMBDA_IDENTITY = 0.0  # Typically, identity loss is not used for pixel-level tasks.
LAMBDA_CYCLE = 10      # Cycle consistency loss weight.

# Number of workers for PyTorch DataLoader.
NUM_WORKERS = 8

# Number of epochs for training.
NUM_EPOCHS = 10

# Flags to determine whether to load and save the models.
LOAD_MODEL = True
SAVE_MODEL = True

# Filenames for the generator and discriminator checkpoints.
# Naming convention has been updated to match the project context.
CHECKPOINT_GEN_SPRITE = "gen_Sprite.pth.tar"
CHECKPOINT_GEN_3D = "gen_3D.pth.tar"
CHECKPOINT_CRITIC_SPRITE = "critic_Sprite.pth.tar"
CHECKPOINT_CRITIC_3D = "critic_3D.pth.tar"

# Albumentations is a library for image augmentation. Here we define the transformation
# pipeline that will be applied to each image as it is loaded.
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),  # Resize the image to a square of 256x256 pixels.
        A.HorizontalFlip(p=0.5),          # Randomly flip the image horizontally 50% of the time.
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        # Normalize pixel values to have a mean of 0.5 and standard deviation of 0.5.
        ToTensorV2(),  # Convert the image to a PyTorch tensor.
    ],
    additional_targets={"image0": "image"},
    # 'additional_targets' allows for the same transformation to be applied to multiple targets.
)
