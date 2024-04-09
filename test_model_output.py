import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from generator_model import Generator
import glob

# Define directories
input_dir = "model_images/Input_Images"
output_dir = "model_images/Output_Images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Load the generator model
generator = Generator(img_channels=3, num_residuals=9).to('cuda')
generator.load_state_dict(torch.load('gen_Sprite.pth.tar', map_location='cuda')['state_dict'])
generator.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the input of the model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize the image
])

# Process images
input_images = glob.glob(os.path.join(input_dir, "*.png"))[:3]  # Process only the first 3 PNG images

for input_image_path in input_images:
    input_image = Image.open(input_image_path).convert('RGB')
    preprocessed_image = transform(input_image).unsqueeze(0).to('cuda')  # Apply the transformation
    
    # Generate the image
    with torch.no_grad():
        generated_image = generator(preprocessed_image).squeeze(0)
        generated_image = (generated_image + 1) / 2  # Denormalize

    # Convert the tensor to an image
    output_image = transforms.ToPILImage()(generated_image.cpu())
    
    # Save the image
    output_image_path = os.path.join(output_dir, os.path.basename(input_image_path))
    output_image.save(output_image_path)
    
    print(f"Processed {input_image_path} -> {output_image_path}")
