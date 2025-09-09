import torch
import numpy
import sys
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTokenizer, CLIPTextModel

# Add the required safe globals
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# First download the required models
print("Downloading required models...")
blip_id = "Salesforce/blip-image-captioning-base"
sd_id = "CompVis/stable-diffusion-v1-4"

# Download BLIP models
processor = BlipProcessor.from_pretrained(blip_id)
model = BlipForConditionalGeneration.from_pretrained(blip_id)

# Download CLIP models
tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder")

print("Models downloaded successfully!")

# Set the environment variables for GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Using GPU 7 as in the original script

# Run the main script for each subject
for subject in range(16):
    print(f"\nProcessing subject {subject}...")
    output_dir = f"output/subject_{subject}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python src/blip_clip.py -d data/ -g all -m clip -b 40 -o {output_dir} -s {subject}"
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"Error processing subject {subject}")
        # Continue with next subject instead of breaking
        continue

print("\nAll subjects processed!")
