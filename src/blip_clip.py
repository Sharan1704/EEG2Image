from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTokenizer, CLIPTextModel
from dataset import EEGImageNetDataset
from PIL import Image
import argparse
import torch
import os
from tqdm import tqdm

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = EEGImageNetDataset(args)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load models
    print("\nLoading BLIP model...")
    dic = {}
    blip_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(blip_id)
    model = BlipForConditionalGeneration.from_pretrained(blip_id, use_safetensors=True).to(device)
    print("\nLoading CLIP model...")
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
                                               use_safetensors=True).to(device)
    
    print(f"\nProcessing {len(dataset.images)} images for subject {args.subject}...")
    
    # Counters for statistics
    processed = 0
    skipped = 0
    
    for image_name in tqdm(dataset.images, desc="Processing images"):
        # First get the synset ID (folder name) from the image name
        image_synset = image_name.split("_")[0]
        image_path = os.path.join(args.dataset_dir, "imageNet_images", image_synset, image_name)
        
        try:
            raw_image = Image.open(image_path).convert("RGB")
            
            # Generate caption using BLIP
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            generation_config = {
                "max_length": 200,
                "num_beams": 20,
                "temperature": 0.5,
                "top_k": 0,
                "top_p": 0.9,
                "repetition_penalty": 2.0,
                "do_sample": True
            }
            out = model.generate(**inputs, **generation_config)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Save caption
            with open(os.path.join(args.output_dir, "caption.txt"), "a") as f:
                f.write(f"{image_name}\t{caption}\n")
            
            # Generate CLIP embeddings
            inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, 
                             truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_embeddings = text_encoder(inputs.input_ids.to(device))[0]
            dic[image_name] = text_embeddings
            
            processed += 1
            
        except FileNotFoundError:
            print(f"\nWarning: Could not find image {image_path}, skipping...")
            skipped += 1
            continue
        except Exception as e:
            print(f"\nError processing image {image_path}: {str(e)}")
            skipped += 1
            continue
            
        # Save embeddings periodically (every 100 images)
        if processed % 100 == 0:
            print(f"\nSaving checkpoint after {processed} images...")
            torch.save(dic, os.path.join(args.output_dir, "clip_embeddings_checkpoint.pth"))
    
    # Save final embeddings
    print(f"\nSaving final embeddings...")
    torch.save(dic, os.path.join(args.output_dir, "clip_embeddings.pth"))
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {processed}")
    print(f"Total images skipped: {skipped}")
