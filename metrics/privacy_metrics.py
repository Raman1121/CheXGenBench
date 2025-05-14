import os
import numpy as np
import argparse
import random
import pandas as pd

from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    AutoencoderKL,
    AutoPipelineForText2Image,
)
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

SUPPORTED_MODELS = [
    "SD-V1-4",
    "SD-V1-5",
    "SD-V2",
    "SD-V2-1",
    "SD-V3-5",
    "radedit",
    "sana",
    "pixart_sigma",
    "lumina",
    "flux",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate privacy metrics.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path of the saved model pipeline."
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Name of the generative model."
    )
    parser.add_argument("--num_shards", type=int, default=0, help="Number of shards.")
    parser.add_argument("--shard", type=int, help="Shard index")
    parser.add_argument(
        "--subset", type=int, default=2000, help="Create a smaller subset"
    )
    parser.add_argument(
        "--reid_ckpt",
        type=str,
        default=None,
        help="Checkpoint path for the trained re-id model",
    )

    ## Data Args
    parser.add_argument(
        "--real_csv",
        type=str,
        required=True,
        help="CSV file containing paths to real images.",
    )
    parser.add_argument(
        "--real_img_dir", type=str, help="Directory containing real images."
    )
    parser.add_argument(
        "--prompt_col",
        type=str,
        default="annotated_prompt",
        help="Column denoting prompts in the CSV.",
    )
    parser.add_argument(
        "--gen_savedir",
        type=str,
        default="/pvc/PatientReIdentification",
        help="Directory where generations would be saved.",
    )
    parser.add_argument(
        "--results_savedir",
        type=str,
        default="/pvc/PatientReIdentification",
        help="Directory where results would be saved.",
    )
    parser.add_argument(
        "--extra_info",
        type=str,
        default=None,
        help="Extra info to save with the results.",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="To locally save images generated across different seeds.",
    )
    return parser.parse_args()


####################################################################################################

"""
We need to calculate the re-id score between:
1. The ground truth image corresponding to the prompt
2. Multiple generated images with the same prompt
"""


####################################################################################################
"""
Siamese Network trained to identify if two images belong to the same patient or not
"""


class SiameseNetwork(nn.Module):
    def __init__(self, network="ResNet-50", in_channels=3, n_features=128):
        super(SiameseNetwork, self).__init__()
        self.network = network
        self.in_channels = in_channels
        self.n_features = n_features

        if self.network == "ResNet-50":
            # Model: Use ResNet-50 architecture
            self.model = models.resnet50(pretrained=True)
            # Adjust the input layer: either 1 or 3 input channels
            if self.in_channels == 1:
                self.model.conv1 = nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            elif self.in_channels == 3:
                pass
            else:
                raise Exception(
                    "Invalid argument: "
                    + self.in_channels
                    + "\nChoose either in_channels=1 or in_channels=3"
                )
            # Adjust the ResNet classification layer to produce feature vectors of a specific size
            self.model.fc = nn.Linear(
                in_features=2048, out_features=self.n_features, bias=True
            )

        else:
            raise Exception(
                "Invalid argument: "
                + self.network
                + "\nChoose ResNet-50! Other architectures are not yet implemented in this framework."
            )

        self.fc_end = nn.Linear(self.n_features, 1)

    def forward_once(self, x):

        # Forward function for one branch to get the n_features-dim feature vector before merging
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

    def forward(self, input1, input2):

        # Forward
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)

        return output


####################################################################################################


def get_reidentification_score(model, img1, img2, transforms):
    with torch.no_grad():
        img1 = transforms(img1).unsqueeze(0).to("cuda")
        img2 = transforms(img2).unsqueeze(0).to("cuda")

        try:
            outputs = model(img1, img2)
        except:
            import pdb

            pdb.set_trace()
        score = torch.sigmoid(outputs)

    return score.item()


def seed_everything(seed=42):
    """Set all random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


# RadDino Model: https://huggingface.co/microsoft/rad-dino
def load_image_encoder():
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo)

    processor = AutoImageProcessor.from_pretrained(repo)

    return model, processor


def encode_image(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    cls_embeddings = outputs.pooler_output
    return cls_embeddings


########### Stablie Diffusion Pipelines ############


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


def load_sd35_lora_pipeline(model_path, device):

    print("!! Loading Stable Diffusion 3.5 Medium with LoRA Pipeline")
    base_model_id = "stabilityai/stable-diffusion-3.5-medium"
    lora_weights_filename = "sd3-5_medium_lora.safetensors"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    lora_weights_path = os.path.join(model_path, lora_weights_filename)

    print(f"Loading base pipeline: {base_model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id, torch_dtype=torch.float16
    ).to(device)

    print(f"Loading LoRA weights from: {lora_weights_path}")
    pipe.load_lora_weights(
        model_path,  # Directory path
        weight_name=lora_weights_filename,  # Specific filename
        adapter_name="sd3_medium_finetune_MIMIC",  # Optional: Give your LoRA adapter a name
    )
    print("LoRA weights loaded successfully.")

    return pipe


def load_sana_pipeline(model_path):
    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    return pipe


def load_pixart_pipeline(model_path):
    from diffusers import PixArtSigmaPipeline

    weight_dtype = torch.float16

    pipe = PixArtSigmaPipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )

    return pipe


def load_lumina_pipeline(model_path, device):
    print("!! Loading Lumina 2.0 with LoRA Pipeline")
    base_model_id = "Alpha-VLLM/Lumina-Image-2.0"
    lora_weights_filename = "lumina2_lora.safetensors"
    dtype = torch.bfloat16

    lora_weights_path = os.path.join(model_path, lora_weights_filename)

    print(f"Loading base pipeline: {base_model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id, torch_dtype=dtype
    ).to(device)

    print("Base pipeline loaded.")

    print(f"Loading LoRA weights from: {lora_weights_path}")

    pipe.load_lora_weights(
        model_path,  # Directory path
        weight_name=lora_weights_filename,  # Specific filename
        adapter_name="lumina2_medium_finetune_MIMIC",  # Optional: Give your LoRA adapter a name
    )
    print("LoRA weights loaded successfully.")

    return pipe


def load_pipeline(model_name, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RadEdit Model
    if model_name == "radedit":
        pipe = load_radedit_pipeline()
        pipe = pipe.to(device)

    # SD V1/ V2.x Models
    # elif ("V1" in model_path) or ("V2" in model_path):
    elif (
        (model_name == "SD-V1-4")
        or (model_name == "SD-V1-5")
        or (model_name == "SD-V2")
        or (model_name == "SD-V2-1")
    ):
        pipe = load_sd_pipeline(model_path)
        pipe = pipe.to(device)

    # SD 3.5 Medium with LoRA
    elif model_name == "SD-V3-5":
        pipe = load_sd35_lora_pipeline(model_path, device)

    # Sana 0.6B (512)
    elif model_name == "sana":
        pipe = load_sana_pipeline(model_path)
        pipe = pipe.to(device)

    # Pixart Sigma
    elif model_name == "pixart_sigma":
        pipe = load_pixart_pipeline(model_path)
        pipe = pipe.to(device)

    elif model_name == "lumina":
        pipe = load_lumina_pipeline(model_path, device)
        pipe = pipe.to(device)

    return pipe


def generate_synthetic_images(pipe, pipeline_constants, prompt, seed=42):

    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt,
        generator=generator,
        height=512,
        width=512,
        **pipeline_constants,
    ).images[0]

    return image


# Functions to calculate distance
def pixelwise_distance(image1: Image.Image, image2: Image.Image) -> float:

    arr1 = np.array(image1)
    arr2 = np.array(image2)
    if arr1.shape != arr2.shape:
        import pdb

        pdb.set_trace()
        raise ValueError("Images must have the same shape.")

    distance = np.linalg.norm(arr1 - arr2)
    distance = distance / np.sqrt(arr1.size)
    return distance


def latent_vector_distance(vector1: torch.Tensor, vector2: torch.Tensor) -> float:

    if vector1.shape != vector2.shape:
        raise ValueError("Latent vectors must have the same shape.")

    distance = torch.norm(vector1 - vector2)
    distance = distance / (vector1.numel() ** 0.5)

    return distance.item()


def main(args):

    seed_everything(42)
    assert args.model_name is not None
    if args.model_name != "radedit":
        assert args.model_path is not None  # RadEdit is directly fetched from HF!

    if args.num_shards > 0:
        assert args.shard is not None

    PIPELINE_CONSTANTS = {
        "radedit": {
            "num_inference_steps": 100,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V1-4": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V1-5": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V2": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": 1,
        },
        "SD-V2-1": {
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
    }

    model = SiameseNetwork(network="ResNet-50", in_channels=3, n_features=128).to(
        "cuda"
    )

    # Loading ckpt
    print("Loading Re-ID Model")
    CKPT_PATH = args.reid_ckpt
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()
    print("Done!")

    print("Loading data...")
    df_train = pd.read_csv(args.real_csv)
    df_train["path"] = df_train["path"].apply(
        lambda x: os.path.join(args.real_img_dir, x)
    )
    print("Data loaded successfully!")

    df_combined = df_train

    # Selecting only unique prompts
    _df = df_combined.drop_duplicates(subset=[args.prompt_col]).reset_index(drop=True)

    # Removing prompts with nans
    _df = _df.dropna(subset=[args.prompt_col]).reset_index(drop=True)

    if args.subset:
        _df = _df.sample(n=args.subset, random_state=42).reset_index(drop=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    if args.num_shards > 0:
        print("Dividing into {} shards".format(args.num_shards))
        all_shards = np.array_split(_df, args.num_shards)
        _df = all_shards[args.shard].reset_index(drop=True)

    print("Length of dataframe: ", len(_df))

    NUM_GENERATIONS = 10
    SEEDS = random.choices(range(1, 1000), k=NUM_GENERATIONS)
    ALL_REAL_IMAGES_PATHS = []
    ALL_PROMPTS = []
    ERROR_PROMPTS = []

    ALL_MEAN_REID_SCORES = []
    ALL_MAX_REID_SCORES = []

    ALL_MEAN_PIXEL_DISTANCES = []
    ALL_MIN_PIXEL_DISTANCES = []

    ALL_MEAN_LATENT_DISTANCES = []
    ALL_MIN_LATENT_DISTANCES = []

    # LOAD SD Pipeline (RadEdit)
    print(f"Loading {args.model_name} Pipeline")
    pipe = load_pipeline(args.model_name, args.model_path)
    print("Done!")

    print(f"Constants set for the {args.model_name} pipeline: ")
    print(PIPELINE_CONSTANTS[args.model_name])

    print("Loading Encoding Model")
    encoding_model, processor = load_image_encoder()
    print("Done!")

    GEN_SAVE_DIR = os.path.join(args.gen_savedir, args.model_name)
    if args.extra_info:
        GEN_SAVE_DIR = os.path.join(
            args.gen_savedir, args.model_name + "_" + args.extra_info
        )
    os.makedirs(GEN_SAVE_DIR, exist_ok=True)
    print(
        "Generations across multiple prompts and seeds would be saved at: ",
        GEN_SAVE_DIR,
    )

    for i in tqdm(range(len(_df))):
        _PATH = _df["path"][i]
        _PROMPT = _df[args.prompt_col][i]
        generated_images = []
        reid_scores = []
        pixel_distances = []
        latent_distances = []

        filename = _PATH.split("/")[-1].strip(".jpg")

        # Get real image from the training set corresponding to the prompt
        # There can be several real images corresponding to a prompt
        # Select 1

        # try:
        real_img_path = _PATH
        real_image = Image.open(real_img_path).resize((512, 512)).convert("RGB")

        print("Prompt: ", _PROMPT)
        # Generate images using this prompt
        for _seed in SEEDS:
            print("Generating with seed: ", _seed)

            gen_image = generate_synthetic_images(
                pipe, PIPELINE_CONSTANTS[args.model_name], _PROMPT, _seed
            )
            gen_image = gen_image.convert("RGB")
            generated_images.append(gen_image)

        # Calculate Re-id score between the real and the generated images
        print("Calculating Re-ID Scores!")
        for i, gen_img in enumerate(generated_images):

            # Save the generated image
            if args.save_generations:
                gen_img.save(
                    os.path.join(GEN_SAVE_DIR, filename + "_gen_{}.jpg".format(i))
                )

            # Re-Identification Score
            reid_score = round(
                get_reidentification_score(model, real_image, gen_img, transform), 4
            )
            reid_scores.append(reid_score)

            # Pixel-Wise Distance
            pixel_dist = round(pixelwise_distance(real_image, gen_image), 4)
            pixel_distances.append(pixel_dist)

            # Latent Distance
            real_img_encoded = encode_image(encoding_model, processor, real_image)
            syn_img_encoded = encode_image(encoding_model, processor, gen_img)
            latent_dist = round(
                latent_vector_distance(real_img_encoded, syn_img_encoded), 4
            )
            latent_distances.append(latent_dist)

        # Re-Identification Score
        _max_reid_score = max(reid_scores)
        _mean_reid_score = sum(reid_scores) / len(reid_scores)
        ALL_MEAN_REID_SCORES.append(_mean_reid_score)
        ALL_MAX_REID_SCORES.append(_max_reid_score)

        # Distance in Pixel-Space
        _min_pixel_dist = min(pixel_distances)
        _mean_pixel_dist = sum(pixel_distances) / len(pixel_distances)
        ALL_MEAN_PIXEL_DISTANCES.append(_mean_pixel_dist)
        ALL_MIN_PIXEL_DISTANCES.append(_min_pixel_dist)

        # Distance in Latent Space
        _min_latent_dist = min(latent_distances)
        _mean_latent_dist = sum(latent_distances) / len(latent_distances)
        ALL_MEAN_LATENT_DISTANCES.append(_mean_latent_dist)
        ALL_MIN_LATENT_DISTANCES.append(_min_latent_dist)

        ALL_REAL_IMAGES_PATHS.append(real_img_path)
        ALL_PROMPTS.append(_PROMPT)
        print("\n")

    # Create a dataframe of results of all scores

    results_df = pd.DataFrame()
    results_df["Real Path"] = ALL_REAL_IMAGES_PATHS
    results_df["Prompt"] = ALL_PROMPTS
    results_df["Mean_reid_scores"] = ALL_MEAN_REID_SCORES
    results_df["Max_reid_scores"] = ALL_MAX_REID_SCORES
    results_df["Mean_pixel_distance"] = ALL_MEAN_PIXEL_DISTANCES
    results_df["Min_pixel_distance"] = ALL_MIN_PIXEL_DISTANCES
    results_df["Mean_latent_distance"] = ALL_MEAN_LATENT_DISTANCES
    results_df["Min_latent_distance"] = ALL_MIN_LATENT_DISTANCES

    print("###### RESULTS ######")
    print(results_df)
    os.makedirs(args.results_savedir, exist_ok=True)

    results_name = f"privacy_metrics_{args.model_name}.csv"

    # If shards used
    if args.num_shards == 0:
        results_name = f"privacy_metrics_{args.model_name}_shard_{args.shard}.csv"

    # If given extra info
    if args.extra_info:
        results_name = f"privacy_metrics_{args.model_name}_{args.extra_info}.csv"

    errors_name = (
        f"error.csv" if args.num_shards == 0 else f"error_shard_{args.shard}.csv"
    )
    results_df.to_csv(os.path.join(args.results_savedir, results_name), index=False)
    print("Results saved at: ", os.path.join(args.results_savedir, results_name))

    ## Saving errored files
    if len(ERROR_PROMPTS) > 0:
        errors_df = pd.DataFrame()
        errors_df["Error_Prompts"] = ERROR_PROMPTS
        errors_df.to_csv(os.path.join(args.results_savedir, errors_name), index=False)
        print("Errors saved at: ", os.path.join(args.results_savedir, errors_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
