import os
import numpy as np
import argparse
import random
import pandas as pd

from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    AutoencoderKL,
    AutoPipelineForText2Image,
    SanaPipeline,
    PixArtSigmaPipeline
)
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate privacy metrics.")
    parser.add_argument("--model_path", type=str, default=None, help="Path of the saved model pipeline.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the generative model.")

    parser.add_argument(
        "--real_csv",
        type=str,
        required=True,
        help="CSV file containing paths to real images.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--savedir", type=str, default="/pvc/PatientReIdentification", help="Directory where generations would be saved."
    )
    parser.add_argument("--extra_info", type=str, default=None)

    parser.add_argument(
        "--subset", type=int, default=2000, help="Create a smaller subset"
    )

    return parser.parse_args()

def seed_everything(seed=42):
    """Set all random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

####################################

"""
Pipeline loading functions for different T2I Models
"""
####################################
# RadEdit
def load_radedit_pipeline():

    print("!! Loading RadEdit Pipeline")
    # 1. UNet
    unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet")

    # 2. VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    # 3. Text encoder and tokenizer
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )

    # 4. Scheduler
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )

    return pipe

####################################
# Stable Diffusion 1.x/ 2.x

def load_sd_pipeline(model_path):

    pipeline_constants = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 1,
    }

    print("!! Loading Stable Diffusion Pipeline")
    print("!! Path: ", model_path)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        safety_checker=None,
    )
    return pipe

####################################
# SD3.5X LoRA

def load_sd35_lora_pipeline(model_path):

    print("!! Loading Stable Diffusion 3.5 Medium with LoRA Pipeline")
    base_model_id = "stabilityai/stable-diffusion-3.5-medium"
    lora_weights_filename = "sd3-5_medium_lora.safetensors"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    lora_weights_path = os.path.join(model_path, lora_weights_filename)

    print(f"Loading base pipeline: {base_model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id, torch_dtype=dtype
    )

    print(f"Loading LoRA weights from: {lora_weights_path}")
    pipe.load_lora_weights(
        model_path,  # Directory path
        weight_name=lora_weights_filename,  # Specific filename
        adapter_name="sd3_medium_finetune_MIMIC",  # Optional: Give your LoRA adapter a name
    )
    print("LoRA weights loaded successfully.")

    return pipe

####################################
# Lumina2.0

def load_lumina_pipeline(model_path):
    print("!! Loading Lumina 2.0 with LoRA Pipeline")
    base_model_id = "Alpha-VLLM/Lumina-Image-2.0"
    lora_weights_filename = "lumina2_lora.safetensors"
    dtype = torch.bfloat16

    lora_weights_path = os.path.join(model_path, lora_weights_filename)

    print(f"Loading base pipeline: {base_model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id, 
        torch_dtype=dtype)

    print("Base pipeline loaded.")

    print(f"Loading LoRA weights from: {lora_weights_path}")

    pipe.load_lora_weights(
        model_path,              # Directory path
        weight_name=lora_weights_filename, # Specific filename
        adapter_name="lumina2_medium_finetune_MIMIC" # Optional: Give your LoRA adapter a name
    )
    print("LoRA weights loaded successfully.")

    return pipe


####################################
# Sana
def load_sana_pipeline(model_path):
    pipe = SanaPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    return pipe


####################################
# Pixart Sigma
def load_pixart_pipeline(model_path):
    pipe = PixArtSigmaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
    )

    return pipe

"""
General Purpose Function to load the pipeline
"""

def load_pipeline(model_name, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RadEdit Model
    if model_name == "radedit":
        pipe = load_radedit_pipeline()
        pipe = pipe.to(device)

    # SD V1/ V2.x Models
    # elif ("V1" in model_path) or ("V2" in model_path):
    elif((model_name == "SD-V1-4") or (model_name == "SD-V1-5") or (model_name == "SD-V2") or (model_name == "SD-V2-1")):
        pipe = load_sd_pipeline(model_path)
        pipe = pipe.to(device)

    # SD 3.5 Medium with LoRA
    elif(model_name == "SD-V3-5"):
        pipe = load_sd35_lora_pipeline(model_path, device)
        pipe = pipe.to(device)

    # Sana Model
    elif(model_name == "sana"):
        pipe = load_sana_pipeline(model_path)
        pipe = pipe.to(device)

    # Pixart Sigma Model
    elif(model_name == "pixart_sigma"):
        pipe = load_pixart_pipeline(model_path)
        pipe = pipe.to(device)

    # Lumina2.0
    elif(model_name == "lumina"):
        pipe = load_lumina_pipeline(model_path, device)
        pipe = pipe.to(device)

    return pipe

def generate_synthetic_images(pipe, pipeline_constants, prompt, seed=42):

    # generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt,
        **pipeline_constants,
    )

    return image

class DummyDataset(Dataset):
    def __init__(self, df):
    
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        sample = {
                'id': self.df.iloc[idx]['id'],
                'prompt': self.df.iloc[idx]['annotated_prompt']
                }
        
        return sample

def main(args):

    seed_everything(42)
    assert args.model_name is not None
    assert args.real_csv is not None
    if(args.model_name != "radedit"):
        assert args.model_path is not None  # RadEdit is directly fetched from HF!

    if(args.extra_info is not None):
        # args.savedir = args.savedir + "_" + args.extra_info
        args.savedir = os.path.join(args.savedir, args.extra_info)
    
    os.makedirs(args.savedir, exist_ok=True)
    print("Saving images to {}".format(args.savedir))

    PIPELINE_CONSTANTS = {
        
        "radedit":{
            "num_inference_steps": 100,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V1-4":{
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V1-5":{
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V2":{
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V2-1":{
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V3-5": {
            "num_inference_steps": 40,
            "guidance_scale": 4.5,
            "num_images_per_prompt": 1,
            "max_sequence_length": 512,
        },
        "sana": {
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
        },
        "pixart_sigma": {
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
        },
        "lumina": {
            "num_inference_steps": 50,
            "guidance_scale": 4,
            "num_images_per_prompt": 1,
            "cfg_trunc_ratio": 0.25,
            "cfg_normalization": True,
        }
    }

    print("Loading data...")
    print(f"Loading CSV: {args.real_csv}")
    df = pd.read_csv(args.real_csv)
    print("Data loaded successfully!")

    if(args.subset):
        print(f"!!! Creating a subset of {args.subset} samples")
        df = df.sample(n=args.subset, random_state=42).reset_index(drop=True)

    ## Loading pipeline
    pipe = load_pipeline(args.model_name, args.model_path)
    ## Loading pipeline constants
    print(f"Constants set for the {args.model_name} pipeline: ")
    print(PIPELINE_CONSTANTS[args.model_name])

    print("{} Prompts found".format(len(df)))

    dataset = DummyDataset(df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    SYNTHETIC_PATHS = []

    print("Generating Images...")
    for batch in tqdm(dataloader):
        ALL_ID = batch['id']     
        PROMPTS = batch["prompt"]

        outputs = generate_synthetic_images(
                pipe=pipe, 
                pipeline_constants=PIPELINE_CONSTANTS[args.model_name], 
                prompt=PROMPTS
            )
        outputs = outputs.images

        for id, image in zip(ALL_ID, outputs):
            filename = "SyntheticImg_{}.png".format(id)
            savepath = os.path.join(args.savedir, filename)
            image.save(savepath)
            SYNTHETIC_PATHS.append(filename)

    # Append the filenames to the original CSV
    df['synthetic_filename'] = SYNTHETIC_PATHS
    filename = "generations_with_metadata.csv"
    df.to_csv(os.path.join(args.savedir, filename), index=False)
    print("Saved to: ", os.path.join(args.savedir, filename))

if __name__ == "__main__":
    args = parse_args()
    main(args)