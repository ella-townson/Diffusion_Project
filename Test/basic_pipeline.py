import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import numpy as np

#load the model and scheduler
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True)

#set the number of timesteps to run the denoising process for 
scheduler.set_timesteps(50)

# Creates a tensor of 50 timesteps for the denoising processs
scheduler.timesteps 

# Create some random noise with the same shape as the desired output
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size))

# Initialize the input with the noise
input = noise

# Iterates over timesteps, using the model to predict the noisy residual.  
# The scheduler refines the input at each step, progressively denoising the image.  
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample


# Convert the final denoised tensor to a PIL image and save it
image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("denoised_image.png")



