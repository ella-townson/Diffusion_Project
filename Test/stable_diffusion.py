from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm
import os
from huggingface_hub import login

# Add your token here (or better, use environment variable)
login(token="Insert")  # Replace with the token you copied
# Check if output directory exists, if not create it
# VAE: Converts images to and from latent space
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Tokenizer & Text Encoder: Convert text to embeddings
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4")

# UNet: The core model that performs the denoising
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")

# Scheduler: Controls the denoising process
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4")

# Choose the appropriate device
torch_device = "mps" if torch.backends.mps.is_available() else "cpu"

# Move the models to the GPU (speed up inference)
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# Define the prompt and the image dimensions
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(prompt)

# Tokenize the prompt and generate the text embeddings
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# Generate the text embeddings for the unconditional prompt
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Generate the initial latent noise
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
# Scale the initial latent noise
latents = latents * scheduler.init_noise_sigma

# Set the number of denoising steps
scheduler.set_timesteps(num_inference_steps)

# Iterate over the denoising steps
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

#  Scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# Convert the image to PIL format and save it
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])
pil_image.save("astronaut_rides_horse.png")
