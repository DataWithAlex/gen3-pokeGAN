import torch
import torch.nn as nn

# The 'Block' class is a custom PyTorch module, acting as a building block for our discriminator.
class Block(nn.Module):
    # The constructor method initializes the block with specified input channels, output channels, and stride.
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()  # Initialize the superclass (nn.Module) to inherit its properties and methods.
        
        # A sequential container that will hold a specific arrangement of layers:
        self.conv = nn.Sequential(
            # Convolutional layer with specific characteristics:
            # - in_channels: Number of channels in the input image.
            # - out_channels: Number of channels produced by the convolution.
            # - Kernel size: The size of the filter.
            # - Stride: The step size of the filter as it moves across the image.
            # - Padding: Zero-padding added to both sides of the input.
            # - bias: Whether to learn an additive bias (True).
            # - padding_mode: Specifies the type of padding, 'reflect' mirrors the input around the edge.
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            
            # Normalizes the features in each feature map independently, stabilizing training.
            nn.InstanceNorm2d(out_channels),
            
            # Activation function allowing a small gradient when the unit is not active, preventing dying neurons.
            nn.LeakyReLU(0.2),
        )
    
    # Forward propagation through this block, taking an input tensor 'x' and applying the defined sequence of layers.
    def forward(self, x):
        return self.conv(x)

    
# The Discriminator class defines the network to distinguish between real and fake images.
class Discriminator(nn.Module):
    # Initialization method with default input channels (e.g., RGB images) and feature progression through the network.
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()  # Initialize nn.Module superclass to get its functionalities.
        
        # Initial layer: Convolution without normalization, directly followed by LeakyReLU.
        # This is often done for the first layer to learn from the raw pixel values.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        # Sequentially building the internal layers of the discriminator.
        layers = []  # Start with an empty list to hold all layers.
        in_channels = features[0]  # Set the input channels for the next layer.
        
        # Iteratively construct layers from the features list, adjusting the stride and channels.
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature  # Update input channels for the next iteration.

        # Final layer: Convolution to reduce to a single output channel, without normalization, for classification.
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)  # Unpack the layers list into a Sequential container.

    # Forward pass through the discriminator, applying initial layer, then the model, and finally a sigmoid.
    def forward(self, x):
        x = self.initial(x)  # Initial layer processing.
        return torch.sigmoid(self.model(x))  # Apply sigmoid to the output for a probability between 0 and 1.

    
# A test function to instantiate the discriminator and run a forward pass with random data.
# At the end of discriminator_model.py

def test():
    x = torch.rand((5,3,256,256))  # Create a random tensor mimicking a batch of images
    model = Discriminator(in_channels=3)  # Instantiate the discriminator model
    preds = model(x)  # Pass the tensor through the model
    print(f"Output shape: {preds.shape}")  # Print the shape of the output tensor


if __name__=="__main__":
    test()