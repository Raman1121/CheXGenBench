import os
import ast
import torch
import pandas as pd
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
import argparse
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from termcolor import colored

from prdc import compute_prdc


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# RadDino Model: https://huggingface.co/microsoft/rad-dino
def load_RadDino_encoder():
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo)

    processor = AutoImageProcessor.from_pretrained(repo)

    return model, processor


# PRDC Metric from: https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf
def compute_prdc_metric(real_features, synthetic_features):
    nearest_k = 5
    metrics = compute_prdc(
        real_features=real_features,
        fake_features=synthetic_features,
        nearest_k=nearest_k,
    )

    return metrics


class RadDinoFeatureExtractor(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def encode_image(self, model, processor, image):
        inputs = processor(images=image, return_tensors="pt")

        with torch.inference_mode():
            inputs = inputs.to(model.device)
            outputs = model(**inputs)
        cls_embeddings = outputs.pooler_output
        return cls_embeddings

    def forward(self, images):
        return self.encode_image(self.model, self.processor, images)


class ImageDataset(Dataset):
    def __init__(self, img_paths=None, transform=None):

        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        assert img_paths is not None
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank = Image.new("RGB", (299, 299), (0, 0, 0))
            return self.transform(blank)


def get_labels_dict_from_string(x):
    return ast.literal_eval(x)


MIMIC_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def main(args):

    if args.num_shards == -1:
        args.num_shards = None
    if args.shard == -1:
        args.shard = None

    if args.experiment_type == "conditional":
        assert args.pathology is not None
        assert args.pathology in MIMIC_PATHOLOGIES

    if args.debug:
        print(
            colored("Debug mode is ON. Make sure this behavior is intended.", "yellow")
        )

    # Set random seed for reproducibility
    seed_everything(42)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Paths to your CSV files
    synthetic_csv = args.synthetic_csv
    real_csv = args.real_csv

    ######### Real Data #########

    # Load real images
    real_df = pd.read_csv(real_csv)
    # Creating paths to the images
    real_df[args.real_img_col] = real_df[args.real_img_col].apply(
        lambda x: os.path.join(args.real_img_dir, x)
    )
    # Drop rows with duplicate prompts
    real_df = real_df.drop_duplicates(subset=[args.real_caption_col]).reset_index(
        drop=True
    )

    ######### Synthetic Data #########

    # Load synthetic images
    synthetic_df = pd.read_csv(synthetic_csv)
    if args.debug:
        n_samples = args.debug_samples
        real_df = real_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        synthetic_df = synthetic_df.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )
    # Creating paths to the images
    synthetic_df[args.synthetic_img_col] = synthetic_df[args.synthetic_img_col].apply(
        lambda x: os.path.join(args.synthetic_img_dir, x)
    )

    # Implement the logic for running analysis on conditional prompts i.e. Calculating metrics only for a specific pathology
    if args.experiment_type == "conditional":

        print(
            colored(
                f"Calculating metrics for the samples containing the pathology: {args.pathology}",
                "yellow",
            )
        )
        real_df["chexpert_labels"] = real_df["chexpert_labels"].apply(
            get_labels_dict_from_string
        )
        # Create a separate column for pathology labels
        for col in MIMIC_PATHOLOGIES:
            real_df[col] = real_df["chexpert_labels"].apply(lambda x: x[col])

        # Fill NaN values with 0
        real_df.fillna(0, inplace=True)

        # Create a subset of the real dataset with the specified pathology
        real_df = real_df[real_df[args.pathology] == 1].reset_index(drop=True)

        # Include only those images from the synthetic dataset that have the same prompts as the real dataset containing the pathology
        real_prompts = real_df[args.real_caption_col].to_list()
        synthetic_df = synthetic_df[
            synthetic_df[args.synthetic_prompt_col].isin(real_prompts)
        ].reset_index(drop=True)

    real_image_paths = real_df[args.real_img_col].tolist()

    synthetic_image_paths = synthetic_df[
        args.synthetic_img_col
    ].tolist()  # The image path col in the CSV is 'img_savename'

    if args.num_shards is not None:
        print(colored(f"Dividing the dataset into {args.num_shards} shards.", "yellow"))
        print(colored(f"Shard Index: {args.shard}", "yellow"))
        ALL_REAL_PATHS = np.array_split(real_image_paths, args.num_shards)
        ALL_SYNTHETIC_PATHS = np.array_split(synthetic_image_paths, args.num_shards)
        real_image_paths = ALL_REAL_PATHS[args.shard]
        synthetic_image_paths = ALL_SYNTHETIC_PATHS[args.shard]

    # Define transform for loading images
    # Note: torchmetrics FID expects images in range [0, 1] without normalization
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # Scales to [0, 1]
        ]
    )

    # Create datasets
    real_dataset = ImageDataset(img_paths=real_image_paths, transform=transform)
    synthetic_dataset = ImageDataset(
        img_paths=synthetic_image_paths, transform=transform
    )

    # Create dataloaders
    real_dataloader = DataLoader(
        real_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    synthetic_dataloader = DataLoader(
        synthetic_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    print(
        colored(
            f"Processing {len(real_dataset)} real images and {len(synthetic_dataset)} synthetic images...",
            "yellow",
        )
    )

    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    # Use RadDino for feature extraction
    model, processor = load_RadDino_encoder()
    rad_dino_fe = RadDinoFeatureExtractor(model, processor)
    fid_raddino = FrechetInceptionDistance(feature=rad_dino_fe).to(device)

    NUM_KID_SUBSETS = 100  # Default
    KID_SUBSET_SIZE = min(len(synthetic_dataset), len(real_dataset))

    kid = KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, feature=2048).to(device)
    kid_raddino = KernelInceptionDistance(
        subset_size=KID_SUBSET_SIZE, feature=rad_dino_fe
    ).to(device)

    inception_score = InceptionScore(feature=2048).to(device)

    # Process real images
    print(colored("Processing real images...", "yellow"))
    ALL_REAL_FEATURES = []
    for batch in tqdm(real_dataloader):

        batch = batch.to(device)
        # Scale images from [0, 1] to [0, 255] as expected by torchmetrics
        batch = (batch * 255).to(torch.uint8)

        # Update FID and KID with real images
        fid.update(batch, real=True)
        # Update FID with RadDino Features
        fid_raddino.update(batch, real=True)

        kid.update(batch, real=True)
        # Update KID with RadDino Features
        kid_raddino.update(batch, real=True)

        # Collect all synthetic features for PRDC
        with torch.inference_mode():
            real_features = rad_dino_fe(batch)
            ALL_REAL_FEATURES.append(real_features.cpu())

    # Process synthetic images
    print(colored("Processing synthetic images...", "yellow"))
    ALL_SYNTHETIC_FEATURES = []
    for batch in tqdm(synthetic_dataloader):

        batch = batch.to(device)
        batch = (batch * 255).to(torch.uint8)

        # Update FID and KID with synthetic images
        fid.update(batch, real=False)
        # Update FID with RadDino features
        fid_raddino.update(batch, real=False)

        kid.update(batch, real=False)
        # Update KID with RadDino Features
        kid_raddino.update(batch, real=False)

        # Update inception score
        inception_score.update(batch)

        # Collect all synthetic features for PRDC
        with torch.inference_mode():
            synthetic_features = rad_dino_fe(batch)
            ALL_SYNTHETIC_FEATURES.append(synthetic_features.cpu())

    # Calculate metrics
    print(colored("Calculating metrics...", "yellow"))
    fid_value = fid.compute()
    fid_raddino_value = fid_raddino.compute()
    kid_mean, kid_std = kid.compute()
    kid_raddino_mean, kid_raddino_std = kid_raddino.compute()
    is_mean, is_std = inception_score.compute()

    # Concatenate all features for PRDC
    ALL_REAL_FEATURES = torch.cat(ALL_REAL_FEATURES, dim=0)
    ALL_SYNTHETIC_FEATURES = torch.cat(ALL_SYNTHETIC_FEATURES, dim=0)
    prdc_metrics = compute_prdc_metric(
        real_features=ALL_REAL_FEATURES,
        synthetic_features=ALL_SYNTHETIC_FEATURES,
    )

    # Save results
    results = {
        "FID": round(fid_value.item(), 3),
        "FID (RadDino)": round(fid_raddino_value.item(), 3),
        "Inception Score": round(is_mean.item(), 3),
        "KID": round(kid_mean.item(), 3),
        "KID (RadDino)": round(kid_raddino_mean.item(), 3),
        "Precision": round(prdc_metrics["precision"].item(), 3),
        "Recall": round(prdc_metrics["recall"].item(), 3),
        "Density": round(prdc_metrics["density"].item(), 3),
        "Coverage": round(prdc_metrics["coverage"].item(), 3),
        "Extra Info": args.extra_info,
        "Caption Type": args.synthetic_prompt_col
    }

    if args.experiment_type == "conditional":
        results["Pathology"] = args.pathology

    print("RESULTS ... ")
    for k, v in results.items():
        print(colored(f"{k}: {v}", "green"))

    # Save to CSV
    results_df = pd.DataFrame([results])

    def prepare_savename(args):
        if args.experiment_type == "conditional":
            savename = "conditional_image_generation_metrics.csv"
        else:
            if args.num_shards is not None:
                savename = f"image_generation_metrics_shard_{args.shard}.csv"
            else:
                savename = "image_generation_metrics.csv"

        if args.debug:
            savename = "debug_" + savename

        return savename

    savename = prepare_savename(args)

    savepath = (
        os.path.join(args.results_savedir, "saved_shards", savename)
        if args.num_shards is not None
        else os.path.join(args.results_savedir, savename)
    )

    if os.path.exists(savepath):
        print("Appending to existing results file.")
        existing_df = pd.read_csv(savepath)
        results_row = list(results.values())

        # existing_df.loc[len(results_df)] = results_row
        # Append the new results to the existing DataFrame
        new_row = pd.DataFrame([results_row], columns=results_df.columns)
        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
        existing_df.to_csv(savepath, index=False)
    else:
        print("Creating new results file.")
        results_df = pd.DataFrame([results])
        results_df.to_csv(savepath, index=False)

    print(colored(f"Results saved to {savepath}", "green"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate FID, KID, and Inception Score for synthetic images."
    )
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        required=True,
        help="CSV file containing paths to synthetic images.",
    )
    parser.add_argument(
        "--synthetic_img_col",
        type=str,
        default="img_savename",
        help="Col name in synthetic CSV for image paths.",
    )
    parser.add_argument(
        "--synthetic_prompt_col",
        type=str,
        default="annotated_prompt",
        help="Col name in synthetic CSV for prompts.",
    )
    parser.add_argument(
        "--synthetic_img_dir", type=str, help="Directory containing synthetic images."
    )

    parser.add_argument(
        "--real_csv",
        type=str,
        required=True,
        help="CSV file containing paths to real images.",
    )
    parser.add_argument(
        "--real_img_col",
        type=str,
        default="path",
        help="Col name in real CSV for image paths.",
    )
    parser.add_argument(
        "--real_caption_col",
        type=str,
        default="annotated_prompt",
        help="Col name in real CSV for image paths.",
    )
    parser.add_argument(
        "--real_img_dir", type=str, help="Directory containing real images."
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing images."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--extra_info",
        type=str,
        default="Some AI Model",
        help="Extra info to link the results with the specific model.",
    )

    parser.add_argument(
        "--results_savedir",
        type=str,
        default="Results",
        help="Directory to save the results.",
    )

    # Experiment Arguments
    parser.add_argument(
        "--experiment_type",
        type=str,
        default=None,
        help="Type of experiment to run (regular, conditional)",
    )
    parser.add_argument(
        "--pathology",
        type=str,
        default="regular",
        help="Type of experiment to run (regular, conditional)",
    )

    parser.add_argument(
        "--num_shards", type=int, default=-1, help="Number of shards to divide into."
    )
    parser.add_argument("--shard", type=int, default=None, help="Shard Index.")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    parser.add_argument("--debug_samples", type=int, default=100, help="Debug Samples.")

    args = parser.parse_args()
    main(args)
