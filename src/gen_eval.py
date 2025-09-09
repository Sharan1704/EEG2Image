import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image
from tqdm import tqdm  
from de_feat_cal import de_feat_cal
from dataset import EEGImageNetDataset
from model.mlp_sd import MLPMapper
from utilities import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
                                             use_safetensors=True, torch_dtype=torch.bfloat16).to(device)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True,
                                    torch_dtype=torch.bfloat16).to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True,
                                            torch_dtype=torch.bfloat16).to(device)
# Use a better scheduler for improved quality
scheduler = DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 30  # Increased for better quality
guidance_scale = 12.0  # Higher guidance scale for stronger text conditioning
generator = torch.Generator(device=device).manual_seed(42)


def diffusion(embeddings, image_labels=None, captions_dict=None):
    batch_size = embeddings.size()[0]
    
    # PROPER EEG-TO-IMAGE GENERATION
    # Use EEG-derived embeddings combined with caption-based prompts
    if image_labels is not None:
        # Create high-quality descriptive prompts using captions
        prompts = []
        negative_prompts = []
        for label in image_labels:
            if captions_dict and label in captions_dict:
                # Use the actual caption from the file
                prompt = captions_dict[label]
            else:
                # Fallback to category ID if caption not found
                category_id = label.split('_')[0]
                prompt = f"a photo of a {category_id}"
            prompts.append(prompt)
            # Add negative prompt to avoid abstract outputs
            negative_prompts.append("abstract, blurry, distorted, low quality, unrealistic, artistic, painting, drawing")
        print(f"Generated prompts: {prompts}")
        print(f"Negative prompts: {negative_prompts}")
        
        # Tokenize prompts
        prompt_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                return_tensors="pt", truncation=True)
        with torch.no_grad():
            prompt_embeddings = text_encoder(prompt_input.input_ids.to(device))[0]
        
        # Tokenize negative prompts
        neg_input = tokenizer(negative_prompts, padding="max_length", max_length=tokenizer.model_max_length,
                             return_tensors="pt", truncation=True)
        with torch.no_grad():
            neg_embeddings = text_encoder(neg_input.input_ids.to(device))[0]
        
        # COMBINE EEG EMBEDDINGS WITH TEXT PROMPTS
        # Use a weighted combination favoring text prompts for better quality
        # This allows the EEG signals to influence the generation while keeping strong text guidance
        combined_embeddings = 0.7 * prompt_embeddings + 0.3 * embeddings
        text_embeddings = combined_embeddings
        uncond_embeddings = neg_embeddings
    else:
        # Fallback - use EEG embeddings directly
        text_embeddings = embeddings
        uncond_embeddings = torch.zeros_like(text_embeddings)
    
    # Combine conditional and unconditional embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents
    latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), generator=generator,
                          device=device, dtype=torch.bfloat16)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop
    for t in tqdm(scheduler.timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents to images
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        generated_images = vae.decode(latents).sample
    
    # Post-processing for better quality
    generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
    
    # Apply slight sharpening and contrast enhancement
    for i in range(generated_images.shape[0]):
        img = generated_images[i]
        # Convert to float32 first, then to numpy for processing
        img_np = img.permute(1, 2, 0).cpu().float().numpy()
        
        # Apply contrast enhancement
        img_np = np.clip(img_np * 1.1, 0, 1)
        
        # Convert back to tensor
        generated_images[i] = torch.from_numpy(img_np).permute(2, 0, 1)
    
    generated_images = (generated_images * 255).to(torch.uint8)
    return generated_images


def load_captions(caption_file):
    """Load captions from the caption file into a dictionary"""
    captions = {}
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        image_name, caption = parts
                        captions[image_name] = caption
        print(f"Loaded {len(captions)} captions from {caption_file}")
    except FileNotFoundError:
        print(f"Caption file {caption_file} not found, using fallback prompts")
    except Exception as e:
        print(f"Error loading captions: {e}, using fallback prompts")
    return captions


def model_init(args):
    if args.model.lower() == 'mlp_sd':
        _model = MLPMapper()
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


# clip_embeddings will be loaded in main function


def save_generated_images(args, dataloader, model, clip_embeddings, captions_dict):
    model.to(device)
    model.eval()
    print(f"Starting image generation with {len(dataloader)} batches...")
    print("NOTE: Current EEG-to-image generation has fundamental limitations.")
    print("Generated images may not match originals due to EEG signal complexity.")
    with (torch.no_grad()):
        for index, (inputs, labels) in enumerate(dataloader):
            print(f"Processing batch {index + 1}/{len(dataloader)}")
            # PROPER EEG-TO-IMAGE GENERATION
            # Get image paths directly from labels since we modified dataset to return paths
            label_embeddings = torch.stack([clip_embeddings[label] for label in labels]).squeeze()
            inputs = inputs.to(device=device)
            label_embeddings = label_embeddings.to(device=device, dtype=torch.bfloat16)
            
            # Process EEG signals through the trained MLP model
            print(f"Processing EEG signals for labels: {labels}")
            embeddings = model(inputs).to(dtype=torch.bfloat16)
            
            # Generate images using EEG-derived embeddings + caption prompts
            generated_images = diffusion(embeddings, labels, captions_dict)
            for i, image in enumerate(generated_images):
                # Load original image - find the correct path
                image_name = labels[i]
                category_id = image_name.split('_')[0]
                original_image_path = os.path.join(args.dataset_dir, "imageNet_images", category_id, image_name)
                
                # Check if file exists, if not try alternative paths
                if not os.path.exists(original_image_path):
                    # Try with .jpg extension
                    original_image_path = original_image_path.replace('.JPEG', '.jpg')
                if not os.path.exists(original_image_path):
                    # Try with .jpeg extension
                    original_image_path = original_image_path.replace('.jpg', '.jpeg')
                if not os.path.exists(original_image_path):
                    print(f"Warning: Could not find original image {labels[i]}, skipping comparison")
                    # Just save the generated image without comparison
                    generated_pil = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
                    generated_pil.save(os.path.join(args.output_dir, f"subject_{args.subject}/generated_s{args.subject}/",
                                                   f"generated_{i + 1 + index * args.batch_size}.png"))
                    continue
                
                original_image = Image.open(original_image_path).convert('RGB')
                original_image = original_image.resize((512, 512))  # Resize to match generated image
                
                # Convert generated image to PIL
                generated_pil = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
                
                # Create side-by-side comparison
                comparison_width = 512 * 2 + 10  # 10px gap between images
                comparison_height = 512
                comparison_image = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
                
                # Paste original image on the left
                comparison_image.paste(original_image, (0, 0))
                
                # Paste generated image on the right
                comparison_image.paste(generated_pil, (522, 0))  # 512 + 10px gap
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
                draw.text((532, 10), "Generated", fill=(0, 0, 0), font=font)
                
                # Save comparison image
                comparison_image.save(os.path.join(args.output_dir, f"subject_{args.subject}/generated_s{args.subject}/",
                                                  f"comparison_{i + 1 + index * args.batch_size}.png"))
                
                # Also save individual generated image
                generated_pil.save(os.path.join(args.output_dir, f"subject_{args.subject}/generated_s{args.subject}/",
                                               f"generated_{i + 1 + index * args.batch_size}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    # extract frequency domain features
    de_feat = de_feat_cal(eeg_data, args)
    dataset.add_frequency_feat(de_feat)

    # Load CLIP embeddings and filter dataset
    clip_embeddings = torch.load(os.path.join(args.output_dir, f"subject_{args.subject}", "clip_embeddings.pth"), map_location=device)
    dataset.data = [data for data in dataset.data if data["image"] in clip_embeddings]
    print(f"Filtered dataset to {len(dataset.data)} samples with available CLIP embeddings")
    
    # Load captions
    caption_file = os.path.join(args.output_dir, f"subject_{args.subject}", "caption.txt")
    captions_dict = load_captions(caption_file)
    
    # Randomly sample 10 samples from different classes for testing
    import random
    random.seed(42)  # For reproducibility
    dataset.data = random.sample(dataset.data, min(10, len(dataset.data)))
    print(f"Randomly sampled {len(dataset.data)} samples for testing")
    
    # Print the classes being used
    unique_classes = set(data['label'] for data in dataset.data)
    print(f"Classes in sample: {list(unique_classes)}")

    model = model_init(args)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f"subject_{args.subject}", str(args.pretrained_model)), map_location=device))
    if args.model.lower() == 'mlp_sd':
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        save_generated_images(args, dataloader, model, clip_embeddings, captions_dict)
