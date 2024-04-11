from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    # Set up a transformation pipeline that will be used to preprocess the images.
    # This includes resizing the images, randomly flipping them horizontally (for data augmentation),
    # normalizing pixel values, and converting the numpy array into a PyTorch tensor.
    return A.Compose([
        A.Resize(width=256, height=256),  # Resize images to the desired input size for the model.
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip to 50% of the images.
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        # Normalize images by scaling pixel values to have a mean of 0.5 and a std of 0.5.
        ToTensorV2(),  # Convert the image to a PyTorch tensor.
    ])

class PokemonDataset(Dataset):
    def __init__(self, root_3d_pokemon, root_sprite_pokemon):
        # Initialize the dataset with the paths to 3D and sprite Pokémon images.
        self.root_3d_pokemon = root_3d_pokemon
        self.root_sprite_pokemon = root_sprite_pokemon
        self.transform = get_transform()  # Initialize the transform.

        # List all the image filenames in the respective directories.
        self.pokemon_3d_images = os.listdir(root_3d_pokemon)
        self.pokemon_sprite_images = os.listdir(root_sprite_pokemon)
        
        # Set the dataset length to the maximum number of images available between both sets.
        self.length_dataset = max(len(self.pokemon_3d_images), len(self.pokemon_sprite_images))

    def __len__(self):
        # The length is the maximum number of images in either dataset.
        return self.length_dataset

    def __getitem__(self, index):
        # Get the image file names from both datasets using the index provided.
        # The modulo operator (%) ensures that we loop back to the start if the index goes beyond the dataset size,
        # which is a common technique used in CycleGAN implementations.
        pokemon_3d_img_name = self.pokemon_3d_images[index % len(self.pokemon_3d_images)]
        pokemon_sprite_img_name = self.pokemon_sprite_images[index % len(self.pokemon_sprite_images)]

        # Construct the full file paths for the 3D and sprite images.
        pokemon_3d_path = os.path.join(self.root_3d_pokemon, pokemon_3d_img_name)
        pokemon_sprite_path = os.path.join(self.root_sprite_pokemon, pokemon_sprite_img_name)

        # Load the images using PIL and convert them to RGB to ensure 3 channels.
        pokemon_3d_img = np.array(Image.open(pokemon_3d_path).convert("RGB"))
        pokemon_sprite_img = np.array(Image.open(pokemon_sprite_path).convert("RGB"))
        
        # Apply transformations to both images.
        if self.transform:
            augmented_pokemon_3d = self.transform(image=pokemon_3d_img)
            pokemon_3d_img = augmented_pokemon_3d["image"]
            
            augmented_pokemon_sprite = self.transform(image=pokemon_sprite_img)
            pokemon_sprite_img = augmented_pokemon_sprite["image"]

        # Return the preprocessed images.
        return pokemon_3d_img, pokemon_sprite_img

# Usage example:
# dataset = PokemonDataset(root_3d_pokemon='path/to/3d_pokemon', root_sprite_pokemon='path/to/sprite_pokemon')
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
