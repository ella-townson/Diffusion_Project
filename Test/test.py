import torch
from diffusers import StableDiffusionPipeline

# Check for MPS (Mac M1/M2) or fall back to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

# Generate the image
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]

# Save the image
image.save("astronaut_rides_horse.png")
