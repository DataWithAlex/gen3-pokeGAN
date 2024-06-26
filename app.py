import streamlit as st
from PIL import Image
from torchvision.transforms import transforms
from generator_model import Generator
import torch
import gdown
import os

# Assuming your Streamlit script is running and banner.jpg is in the same folder
banner_path = 'banner.jpg'  # Adjust path if your image is in a different folder

st.set_page_config(page_title="gen3_pokeGAN", page_icon=":sparkles:", layout="wide")

# Display the banner image at the top of the page
banner_image = Image.open(banner_path)
st.image(banner_image, use_column_width='always')

# Download model weights from Google Drive
@st.cache_data
def download_model():
    file_id = '1W3n1cXjoby2wps_mvUP7y9e71nBytksz'
    url = f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}"
    output = 'gen_Sprite.pth.tar'
    
    gdown.download(url, output, quiet=False)
    
    if not os.path.exists(output):
        raise FileNotFoundError(f"Failed to download the model file: {output}")

    return output

model_path = download_model()

def generate_image(image, generator):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        generated_tensor = generator(image_tensor).squeeze(0)
    generated_tensor = (generated_tensor + 1) / 2  # Denormalize
    generated_image = transforms.ToPILImage()(generated_tensor)
    return generated_image

# Load the generator model
def load_model(model_path):
    model = Generator(img_channels=3, num_residuals=9)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")
    model.eval()
    return model

generator = load_model(model_path)

st.title('Welcome to Gen3 PokeGAN!')

st.markdown("""
Generation 3 of the Pokemon games (e.g., Ruby/Sapphire/Emerald) had a unique pixel art style for the pokemon. Here you can input the new pokemon models, and it will show you how it would look like in pokemon Ruby/Sapphire/Emerald!
""")

# Sidebar for image upload
st.sidebar.header("Upload your image")

# Default image preloaded
default_image_path = 'test_image.png'  # Adjust the path if necessary
default_image = Image.open(default_image_path).convert('RGB')
st.sidebar.image(default_image, caption='Default Image', use_column_width=True)

# Option to upload a new image
uploaded_file = st.sidebar.file_uploader("...or choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(input_image, caption='Uploaded Image', use_column_width=True)
    image_to_process = input_image
else:
    image_to_process = default_image

# Generate button
if st.sidebar.button('Generate!'):
    with st.spinner('Generating...'):
        output_image = generate_image(image_to_process, generator)
        st.image(output_image, caption='Generated Image', use_column_width='auto')

# Additional info in the sidebar
st.sidebar.info("This app uses a generative model to convert uploaded images into a Pokémon-like style. Simply upload your image and click on 'Generate!'.")
st.sidebar.markdown("[gen3_pokeGAN GitHub Repository](https://github.com/username/gen3_pokeGAN)")
