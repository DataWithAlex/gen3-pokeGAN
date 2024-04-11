import torch
import torch.nn as nn

# Defines a convolutional block that can either downsample or upsample the input feature map.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # Chooses between a convolutional layer or a transposed convolutional layer based on `down`.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),  # Normalizes the output for each feature map.
            nn.ReLU(inplace=True) if use_act else nn.Identity(),  # Applies ReLU or bypasses it.
        )

    def forward(self, x):
        return self.conv(x)

# Implements a residual block that adds the input to the output, helping with gradient flow in deep networks.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Adds the input `x` to the output of the block, facilitating learning identity functions.
        return x + self.block(x)

# Generator model that includes an initial layer, downsampling, residual blocks, upsampling, and a final layer.
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # Initial layer with a larger kernel size to capture more context.
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        # Downsamples the feature map while increasing the number of features.
        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])
        # Applies several residual blocks that help the model learn complex transformations.
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])
        # Upsamples the feature map back to the original size while decreasing the number of features.
        self.up_blocks = nn.ModuleList([
            ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        # Final layer that maps the features back to the image space.
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)  # Process input through the initial layer.
        for layer in self.down_blocks:  # Apply downsampling.
            x = layer(x)
        x = self.res_blocks(x)  # Apply residual blocks.
        for layer in self.up_blocks:  # Apply upsampling.
            x = layer(x)
        return torch.tanh(self.last(x))  # Output final image with tanh to normalize it to [-1, 1].

def test_detailed():
    # Initialize the test data and model.
    img_channels = 3  # Number of channels in the input images (RGB).
    img_size = 256  # Size of the input images.
    x = torch.randn((2, img_channels, img_size, img_size))  # A batch of 2 random images.
    gen = Generator(img_channels)  # Instantiate the generator.
    print(f"Input shape: {x.shape}")
    print("The input tensor represents a batch of 2 images, each with 3 color channels (RGB) and a spatial resolution of 256x256 pixels. The generator will transform these images through a series of convolutional operations aimed at feature extraction, transformation, and reconstruction.\n")

    # Initial convolutional layer.
    x = gen.initial(x)
    print(f"After initial layer: {x.shape}")
    print("The initial layer employs a 7x7 kernel, striking a balance between capturing local details and broader contextual information. The use of reflection padding minimizes edge artifacts, ensuring the features at the image boundaries are well represented. This layer sets the foundation for subsequent feature extraction and manipulation.\n")

    # Down-sampling blocks: Feature extraction and spatial reduction.
    for i, layer in enumerate(gen.down_blocks):
        x = layer(x)
        print(f"After down block {i+1}: {x.shape}")
        print(f"Down Block {i+1} employs strided convolution to reduce the spatial dimensions by half, effectively doubling the receptive field of the network's neurons. This process enhances the model's ability to abstract and understand larger patterns in the image, essential for capturing the essence of the input domain. Doubling the feature channels increases the model's capacity to represent more complex features.\n")

    # Residual blocks: Deep feature transformation.
    x = gen.res_blocks(x)
    print(f"After residual blocks: {x.shape}")
    print("Residual blocks apply a series of transformations while allowing the original input to bypass these operations through a shortcut connection. This design enables the network to learn identity mappings easily and safeguards against the vanishing gradient problem in deep networks. It's a pivotal component for ensuring depth in the model without losing information quality.\n")

    # Up-sampling blocks: Spatial expansion and feature refinement.
    for i, layer in enumerate(gen.up_blocks):
        x = layer(x)
        print(f"After up block {i+1}: {x.shape}")
        print(f"Up Block {i+1} uses transposed convolutions to increase the spatial dimensions, effectively 'reversing' the down-sampling process. This step is crucial for reconstructing the high-resolution details of the target domain from the abstract feature representations. The reduction in feature channels during up-sampling helps in focusing the model's capacity on accurately reconstructing the image details.\n")

    # Final layer: Image reconstruction.
    x = torch.tanh(gen.last(x))
    print(f"Output shape: {x.shape}")
    print("The final layer, similar to the initial layer, uses a 7x7 kernel for a smooth transition of the features back into the image space, preserving the quality of the reconstructed image. The tanh activation function normalizes the output to the range [-1, 1], suitable for image data. This normalization is crucial for matching the expected distribution of pixel values in the output images.\n")


if __name__ == "__main__":
    test_detailed()
