"""
We calculate the image-text aignment score using the
health_multimodal toolbox (https://github.com/microsoft/hi-ml)
"""

from health_multimodal.text import get_bert_inference
from health_multimodal.image import get_image_inference
from health_multimodal.vlp import ImageTextInferenceEngine
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import ast
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from termcolor import colored
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Image-Text Alignment Score b/w Real Prompt and Synthetic Images"
    )
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        default=None,
        help="CSV File containing the path to synthetic images and corresponding prompts.",
    )
    parser.add_argument(
        "--synthetic_img_dir",
        type=Path,
        default=None,
        help="Directory containing the synthetic images.",
    )

    parser.add_argument(
        "--synthetic_prompts_col",
        type=str,
        default="annotated_prompt",
        help="Column name in the CSV file containing the prompts.",
    )
    parser.add_argument(
        "--synthetic_img_col",
        type=str,
        default="synthetic_filename",
        help="Column name in the CSV file containing the image paths.",
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
        "--results_savedir",
        type=str,
        default="Results",
        help="Directory to save the results.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference."
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
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    parser.add_argument("--debug_samples", type=int, default=100, help="Debug Samples.")

    return parser.parse_args()


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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_labels_dict_from_string(x):
    return ast.literal_eval(x)


def main(args):

    if args.experiment_type == "conditional":
        assert args.pathology is not None
        assert args.pathology in MIMIC_PATHOLOGIES

    if args.debug:
        print(
            colored("Debug mode is ON. Make sure this behavior is intended.", "yellow")
        )

    # Set random seed for reproducibility
    seed_everything(42)

    text_inference = get_bert_inference()
    image_inference = get_image_inference()
    image_text_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

    prompts_df = pd.read_csv(args.synthetic_csv)

    if args.debug:
        prompts_df = prompts_df.sample(
            n=args.debug_samples, random_state=42
        ).reset_index(drop=True)

    if args.experiment_type == "conditional":
        print(
            colored(
                f"Calculating metrics for the samples containing the pathology: {args.pathology}",
                "yellow",
            )
        )
        real_csv = args.real_csv
        prompts_df = prompts_df

        # Load real images
        real_df = pd.read_csv(real_csv)

        # Drop rows with duplicate prompts
        real_df = real_df.drop_duplicates(subset=[args.real_caption_col]).reset_index(
            drop=True
        )

        # Create a separate column for pathology labels
        real_df["chexpert_labels"] = real_df["chexpert_labels"].apply(
            get_labels_dict_from_string
        )

        for col in MIMIC_PATHOLOGIES:
            real_df[col] = real_df["chexpert_labels"].apply(lambda x: x[col])

        # Fill NaN values with 0
        real_df.fillna(0, inplace=True)

        # Create a subset of the real dataset with the specified pathology
        real_df = real_df[real_df[args.pathology] == 1].reset_index(drop=True)

        # Include only those images from the synthetic dataset that have the same prompts as the real dataset containing the pathology
        real_prompts = real_df[args.real_caption_col].to_list()
        prompts_df = prompts_df[prompts_df[args.synthetic_prompts_col].isin(real_prompts)].reset_index(
            drop=True
        )

    # Prepare paths for the synthetic images
    prompts_df[args.synthetic_img_col] = prompts_df[args.synthetic_img_col].apply(
        lambda x: args.synthetic_img_dir.joinpath(x)
    )
    synthetic_img_paths = prompts_df[args.synthetic_img_col].tolist()
    prompts = prompts_df[args.synthetic_prompts_col].tolist()

    print(colored(f"Length of the dataset: {len(prompts_df)}", "yellow"))

    all_scores = []

    for i in tqdm(range(len(prompts))):
        img_path = synthetic_img_paths[i]
        text = prompts[i]
        _score = round(
            image_text_inference.get_similarity_score_from_raw_data(img_path, text), 3
        )
        all_scores.append(_score)
    mean_alignment_scores = round(np.mean(all_scores, axis=0), 3)

    print(colored("RESULTS...", "green"))
    print(colored(f"Mean Img-Text Alignment Score: {mean_alignment_scores}", "green"))

    savename = (
        "conditional_img_text_alignment.csv"
        if args.experiment_type == "conditional"
        else "img_text_alignment.csv"
    )
    if args.debug:
        savename = "debug_" + savename

    savepath = os.path.join(args.results_savedir, savename)

    # Try to read if the dataframe already exists
    if os.path.exists(savepath):

        print(
            colored(f"Appending to existing results file found at {savepath}", "yellow")
        )
        results_df = pd.read_csv(savepath)

        # Append a new row with the new results
        new_row = {
            "Alignment_score": mean_alignment_scores,
            "Extra Info": args.extra_info,
        }

        if args.experiment_type == "conditional":
            new_row["Pathology"] = args.pathology

        results_df.loc[len(results_df)] = new_row
        results_df.to_csv(savepath, index=False)
        print(colored(f"Image-Text Alignment Scores saved to: {savepath}", "green"))
    else:
        results = {
            "Alignment_score": mean_alignment_scores,
            "Extra Info": args.extra_info,
        }
        if args.experiment_type == "conditional":
            results["Pathology"] = args.pathology

        print("Creating new results file.")
        results_df = pd.DataFrame([results])
        results_df.to_csv(savepath, index=False)


if __name__ == "__main__":
    args = parse_args()

    # Create the results directory if it doesn't exist
    os.makedirs(args.results_savedir, exist_ok=True)

    main(args)
