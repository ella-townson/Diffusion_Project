from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

class ImageGenerator:
    def __init__(self):
        # Initialize the model
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id)
        self.pipeline.to(self.device)

    def generate_image(self, prompt, output_dir="generated_images"):
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate the image
        print(f"Generating image for prompt: '{prompt}'")
        image = self.pipeline(prompt).images[0]
        
        # Save the image with a fixed filename
        filename = f"{output_dir}/latest_image.png"
        image.save(filename)
        print(f"Image saved as: {filename}")
        return filename 