# Diffusion_Project

This project is a simple implementation of a diffusion model using the Diffusers library.

# Test 

## Test Pipeline

The test pipeline uses StableDiffusionPipeline to:
    - Load "CompVis/stable-diffusion-v1-4" model
    - Auto-select between MPS/CPU
    - Generate 512x512 images from text prompts
    - Handle pipeline setup and execution
    - Use default parameters

## Basic Pipeline

The basic pipeline implements a straightforward diffusion process using the DDPM (Denoising Diffusion Probabilistic Model) approach:
    - Uses a pretrained UNet2DModel from "google/ddpm-cat-256"
    - Implements a 50-step denoising process using DDPMScheduler
    - Starts from pure Gaussian noise (shape: 1x3x256x256)
    - Iteratively denoises the image using the UNet to predict noise residuals
    - Outputs a 256x256 pixel image
    - Uses an unconditional generation process (no text input)

## Stable Diffusion

Stable Diffusion is a latent diffusion model that:
    - Uses a VAE (AutoencoderKL) to compress images into an efficient 64x64 latent space
    - Incorporates CLIP text encoder for converting text prompts into embeddings
    - Uses a conditional UNet2DModel that processes both latent images and text embeddings
    - Implements classifier-free guidance for better text adherence (guidance_scale=7.5)
    - Uses the UniPC scheduler for the denoising process
    - Supports 512x512 pixel output images
    - Includes both conditional and unconditional embeddings for guidance
    - Performs the full pipeline: text encoding → latent noise generation → guided denoising → VAE decoding



## References

- [Diffusion Models](https://huggingface.co/docs/diffusers/index)

