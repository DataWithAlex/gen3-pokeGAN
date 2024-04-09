import optuna
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from train import train_fn, validate_fn, Generator, Discriminator, PokemonDataset
import config

def create_study(lr_range=(1e-5, 1e-1), lambda_cycle_range=(0.0, 20.0)):
    def objective(trial):
        # Dynamic range for learning rate and lambda cycle
        lr = trial.suggest_loguniform('lr', *lr_range)
        lambda_cycle = trial.suggest_float('lambda_cycle', *lambda_cycle_range)

        # Apply the suggested lambda_cycle to config
        config.LAMBDA_CYCLE = lambda_cycle

        # Assuming DEVICE is defined in config
        device = config.DEVICE

        # Model setup
        disc_Sprite = Discriminator(in_channels=3).to(device)
        disc_3D = Discriminator(in_channels=3).to(device)
        gen_Sprite = Generator(img_channels=3, num_residuals=9).to(device)
        gen_3D = Generator(img_channels=3, num_residuals=9).to(device)

        # Optimizers
        opt_disc = optim.Adam(list(disc_Sprite.parameters()) + list(disc_3D.parameters()), lr=lr)
        opt_gen = optim.Adam(list(gen_Sprite.parameters()) + list(gen_3D.parameters()), lr=lr)

        # Loss functions
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        # Data Loaders
        dataset = PokemonDataset(root_3d_pokemon=config.TRAIN_DIR + "/3D_pokemon", root_sprite_pokemon=config.TRAIN_DIR + "/sprite_pokemon")
        val_dataset = PokemonDataset(root_3d_pokemon=config.VAL_DIR + "/3D_pokemon", root_sprite_pokemon=config.VAL_DIR + "/sprite_pokemon")
        train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

        # Training Loop
        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_fn(disc_Sprite, disc_3D, gen_3D, gen_Sprite, train_loader, opt_disc, opt_gen, l1_loss, mse_loss, torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler())

            val_loss = validate_fn(val_loader, gen_Sprite, gen_3D, l1_loss, mse_loss)

            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss

    study = optuna.create_study(direction='minimize')
    return study, objective

if __name__ == "__main__":
    lr_range = (1e-5, 1e-1)  # Example learning rate range
    lambda_cycle_range = (0.0, 20.0)  # Example lambda_cycle range

    study, objective = create_study(lr_range, lambda_cycle_range)
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
