"""
Training for CycleGAN
"""

import os
import csv
import time
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import config  # Ensure this contains all the necessary configurations
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import PokemonDataset  # Your dataset class
from utils import save_checkpoint, load_checkpoint  # Your utility functions for saving and loading checkpoints


def generate_and_save_images(generator, training_session_start, input_dir="model_images/Input_Images", output_dir="model_images/Output_Images"):
    """Generates and saves images using the generator model."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    os.makedirs(output_dir, exist_ok=True)

    input_images = sorted(os.listdir(input_dir))[:3]  # Process the first 3 images
    input_image_paths = []
    output_image_paths = []

    for img_name in input_images:
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, f"{training_session_start}_{img_name}")
        input_image_paths.append(input_path)
        output_image_paths.append(output_path)

        input_image = Image.open(input_path).convert("RGB")
        input_tensor = transform(input_image).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            output_tensor = generator(input_tensor).squeeze(0)
            output_tensor = (output_tensor + 1) / 2  # Denormalize

        output_image = transforms.ToPILImage()(output_tensor.cpu())
        output_image.save(output_path)

    return input_image_paths, output_image_paths


def log_metrics(epoch, train_loss, val_loss, learning_rate, lambda_identity, lambda_cycle, avg_G_Sprite_loss, avg_G_3D_loss, avg_D_Sprite_loss, avg_D_3D_loss, training_session, start_time, end_time, total_time, input_image_paths, output_image_paths, file_name='training_progress.csv'):
    # Debugging: Print the values of all parameters received
    print(f"Debug: log_metrics() called with parameters:\nepoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, learning_rate: {learning_rate}, lambda_identity: {lambda_identity}, lambda_cycle: {lambda_cycle}, avg_G_Sprite_loss: {avg_G_Sprite_loss}, avg_G_3D_loss: {avg_G_3D_loss}, avg_D_Sprite_loss: {avg_D_Sprite_loss}, avg_D_3D_loss: {avg_D_3D_loss}, training_session: {training_session}, start_time: {start_time}, end_time: {end_time}, total_time: {total_time}, input_image_paths: {input_image_paths}, output_image_paths: {output_image_paths}, file_name: {file_name}")
    
    # Ensure the file_name variable is not empty
    if not file_name:
        raise ValueError("The file_name parameter is empty. Please provide a valid file name.")

    # Ensure the directory for file_name exists, if not create it
    directory = os.path.dirname(file_name)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("Debug: Directory did not exist, created new directory.")

    # Convert tensors to floats if necessary
    train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
    val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
    avg_G_Sprite_loss = avg_G_Sprite_loss.item() if isinstance(avg_G_Sprite_loss, torch.Tensor) else avg_G_Sprite_loss
    avg_G_3D_loss = avg_G_3D_loss.item() if isinstance(avg_G_3D_loss, torch.Tensor) else avg_G_3D_loss
    avg_D_Sprite_loss = avg_D_Sprite_loss.item() if isinstance(avg_D_Sprite_loss, torch.Tensor) else avg_D_Sprite_loss
    avg_D_3D_loss = avg_D_3D_loss.item() if isinstance(avg_D_3D_loss, torch.Tensor) else avg_D_3D_loss

    # Convert list of image paths to a single string to store in the CSV
    input_image_paths_str = ";".join(input_image_paths)
    output_image_paths_str = ";".join(output_image_paths)

    # Check if the CSV file already exists
    file_exists = os.path.isfile(file_name)
    
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the headers if the file does not exist
        if not file_exists:
            print("Debug: File does not exist, writing headers.")
            headers = ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'lambda_identity', 'lambda_cycle', 'avg_G_Sprite_loss', 'avg_G_3D_loss', 'avg_D_Sprite_loss', 'avg_D_3D_loss', 'training_session', 'start_time', 'end_time', 'total_time', 'input_image_paths', 'output_image_paths']
            writer.writerow(headers)
        else:
            print("Debug: File exists, appending to file.")

        # Writing the metrics and paths
        row = [epoch, train_loss, val_loss, learning_rate, lambda_identity, lambda_cycle, avg_G_Sprite_loss, avg_G_3D_loss, avg_D_Sprite_loss, avg_D_3D_loss, training_session, start_time, end_time, total_time, input_image_paths_str, output_image_paths_str]
        writer.writerow(row)
        print("Debug: Metrics and paths written to file successfully.")


def validate_fn(val_loader, gen_Sprite, gen_3D, l1, mse):
    """Validation function to evaluate the model on a validation set."""
    val_loss = 0.0
    for sprite_pokemon, _3D_pokemon in val_loader:
        sprite_pokemon, _3D_pokemon = sprite_pokemon.to(config.DEVICE), _3D_pokemon.to(config.DEVICE)
        with torch.no_grad():
            fake_sprite_pokemon, fake_3D_pokemon = gen_Sprite(_3D_pokemon), gen_3D(sprite_pokemon)
            cycle_sprite_pokemon, cycle_3D_pokemon = gen_Sprite(fake_3D_pokemon), gen_3D(fake_sprite_pokemon)
            identity_sprite_pokemon, identity_3D_pokemon = gen_Sprite(sprite_pokemon), gen_3D(_3D_pokemon)
            val_loss += (mse(fake_sprite_pokemon, sprite_pokemon) + mse(fake_3D_pokemon, _3D_pokemon) + l1(cycle_sprite_pokemon, sprite_pokemon) * config.LAMBDA_CYCLE + l1(cycle_3D_pokemon, _3D_pokemon) * config.LAMBDA_CYCLE + l1(identity_sprite_pokemon, sprite_pokemon) * config.LAMBDA_IDENTITY + l1(identity_3D_pokemon, _3D_pokemon) * config.LAMBDA_IDENTITY)
    return val_loss / len(val_loader)


def train_fn(disc_Sprite, disc_3D, gen_3D, gen_Sprite, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    """Training function for a single epoch."""
    total_loss = 0.0
    G_Sprite_loss_accum, G_3D_loss_accum, D_Sprite_loss_accum, D_3D_loss_accum = 0.0, 0.0, 0.0, 0.0
    Sprite_reals, Sprite_fakes = 0, 0

    loop = tqdm(loader, leave=True)
    for _3D_pokemon, sprite_pokemon in loop:
        _3D_pokemon, sprite_pokemon = _3D_pokemon.to(config.DEVICE), sprite_pokemon.to(config.DEVICE)
        # LEARNING RATE SCHEDULER GOOGLE IT

        # Train Discriminators
        fake_sprite_pokemon, fake_3D_pokemon = gen_Sprite(_3D_pokemon), gen_3D(sprite_pokemon)
        D_Sprite_real, D_Sprite_fake = disc_Sprite(sprite_pokemon), disc_Sprite(fake_sprite_pokemon.detach())
        D_3D_real, D_3D_fake = disc_3D(_3D_pokemon), disc_3D(fake_3D_pokemon.detach())
        D_Sprite_loss = mse(D_Sprite_real, torch.ones_like(D_Sprite_real)) + mse(D_Sprite_fake, torch.zeros_like(D_Sprite_fake))
        D_3D_loss = mse(D_3D_real, torch.ones_like(D_3D_real)) + mse(D_3D_fake, torch.zeros_like(D_3D_fake))
        D_loss = (D_Sprite_loss + D_3D_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        D_Sprite_fake, D_3D_fake = disc_Sprite(fake_sprite_pokemon), disc_3D(fake_3D_pokemon)
        loss_G_Sprite, loss_G_3D = mse(D_Sprite_fake, torch.ones_like(D_Sprite_fake)), mse(D_3D_fake, torch.ones_like(D_3D_fake))
        cycle_sprite_pokemon, cycle_3D_pokemon = gen_Sprite(fake_3D_pokemon), gen_3D(fake_sprite_pokemon)
        cycle_sprite_loss, cycle_3D_loss = l1(sprite_pokemon, cycle_sprite_pokemon) * config.LAMBDA_CYCLE, l1(_3D_pokemon, cycle_3D_pokemon) * config.LAMBDA_CYCLE
        identity_sprite_loss, identity_3D_loss = l1(sprite_pokemon, gen_Sprite(sprite_pokemon)) * config.LAMBDA_IDENTITY, l1(_3D_pokemon, gen_3D(_3D_pokemon)) * config.LAMBDA_IDENTITY
        G_loss = loss_G_Sprite + loss_G_3D + cycle_sprite_loss + cycle_3D_loss + identity_sprite_loss + identity_3D_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Accumulate and log losses
        G_Sprite_loss_accum += loss_G_Sprite.item()
        G_3D_loss_accum += loss_G_3D.item()
        D_Sprite_loss_accum += D_Sprite_loss.item()
        D_3D_loss_accum += D_3D_loss.item()
        total_loss += D_loss.item() + G_loss.item()
        loop.set_postfix(G_Sprite_loss=G_Sprite_loss_accum / len(loader), G_3D_loss=G_3D_loss_accum / len(loader), D_Sprite_loss=D_Sprite_loss_accum / len(loader), D_3D_loss=D_3D_loss_accum / len(loader))

    average_loss = total_loss / len(loader)
    return average_loss, G_Sprite_loss_accum / len(loader), G_3D_loss_accum / len(loader), D_Sprite_loss_accum / len(loader), D_3D_loss_accum / len(loader)


def main():
    training_session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()

    disc_Sprite = Discriminator(in_channels=3).to(config.DEVICE)
    disc_3D = Discriminator(in_channels=3).to(config.DEVICE)
    gen_3D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Sprite = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_Sprite.parameters()) + list(disc_3D.parameters()), 
        lr=config.LEARNING_RATE, 
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_3D.parameters()) + list(gen_Sprite.parameters()), 
        lr=config.LEARNING_RATE, 
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_SPRITE, gen_Sprite, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_3D, gen_3D, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_SPRITE, disc_Sprite, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_3D, disc_3D, opt_disc, config.LEARNING_RATE)

    dataset = PokemonDataset(root_3d_pokemon=config.TRAIN_DIR + "/3D_pokemon", root_sprite_pokemon=config.TRAIN_DIR + "/sprite_pokemon")
    val_dataset = PokemonDataset(root_3d_pokemon=config.VAL_DIR + "/3D_pokemon", root_sprite_pokemon=config.VAL_DIR + "/sprite_pokemon")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_loss, avg_G_Sprite_loss, avg_G_3D_loss, avg_D_Sprite_loss, avg_D_3D_loss = train_fn(
            disc_Sprite, disc_3D, gen_3D, gen_Sprite, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler
        )
        val_loss = validate_fn(val_loader, gen_Sprite, gen_3D, L1, mse)

        if val_loss < float('inf'):  # Update this condition as per your logic for saving best model
            save_checkpoint(gen_Sprite, opt_gen, filename=config.CHECKPOINT_GEN_SPRITE)
            save_checkpoint(gen_3D, opt_gen, filename=config.CHECKPOINT_GEN_3D)
            save_checkpoint(disc_Sprite, opt_disc, filename=config.CHECKPOINT_CRITIC_SPRITE)
            save_checkpoint(disc_3D, opt_disc, filename=config.CHECKPOINT_CRITIC_3D)

        input_image_paths, output_image_paths = generate_and_save_images(gen_Sprite, training_session_start, "model_images/Input_Images", "model_images/Output_Images")

        training_session_end = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        end_time = time.time()
        total_time = end_time - start_time

        print(f"Debug before log_metrics:")
        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, learning_rate: {config.LEARNING_RATE}, lambda_identity: {config.LAMBDA_IDENTITY}, lambda_cycle: {config.LAMBDA_CYCLE}")
        print(f"avg_G_Sprite_loss: {avg_G_Sprite_loss}, avg_G_3D_loss: {avg_G_3D_loss}, avg_D_Sprite_loss: {avg_D_Sprite_loss}, avg_D_3D_loss: {avg_D_3D_loss}")
        print(f"training_session_start: {training_session_start}, training_session_end: {training_session_end}, total_time: {total_time}")
        print(f"input_image_paths: {input_image_paths}, output_image_paths: {output_image_paths}")

        log_metrics(
            epoch, train_loss, val_loss, config.LEARNING_RATE, config.LAMBDA_IDENTITY, config.LAMBDA_CYCLE, 
            avg_G_Sprite_loss, avg_G_3D_loss, avg_D_Sprite_loss, avg_D_3D_loss, 
            training_session_start, start_time, end_time, total_time, 
            input_image_paths, output_image_paths
        )

if __name__ == "__main__":
    main()


    

