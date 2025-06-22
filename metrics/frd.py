# Source: https://github.com/RichardObi/frd-score/blob/main/frd_v1/compute_frd.py

"""
Compute and interpret fréchet radiomics distances between two datasets.
"""
import os
import shutil
import tempfile
from argparse import ArgumentParser
from time import time
import ast
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
#import matplotlib.pyplot as plt

from radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics, compute_and_save_imagefolder_radiomics_parallel, interpret_radiomic_differences
from utils import frechet_distance


MIMIC_PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 
                     'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 
                     'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

def copy_image(src_path, dest_folder):
    """Worker function: copy one image if it exists, else log."""
    if os.path.exists(src_path):
        try:
            shutil.copy(src_path, dest_folder)
        except Exception as e:
            print(f"Failed to copy {src_path}: {e}")
    else:
        print(f"Image not found: {src_path}")

def main(
        args,
        force_compute_fresh = False,
        interpret = False,
        parallelize = True
):
    
    if(args.experiment_type == 'conditional'):
        assert args.pathology is not None
        assert args.pathology in MIMIC_PATHOLOGIES

    seed_everything(42)

    # Creating a temp folder for real images (since they can be located in different folders)
    image_folder1 = tempfile.mkdtemp(prefix='real_images_')
    image_folder2 = tempfile.mkdtemp(prefix='synthetic_images_')

    # Copy all files from the real image CSV to the temp folder
    real_df = pd.read_csv(args.real_csv)
    synthetic_df = pd.read_csv(args.synthetic_csv)

    # Creating paths to the images
    real_df[args.real_img_col] = real_df[args.real_img_col].apply(
        lambda x: os.path.join(args.real_img_dir, x)
    )
    synthetic_df[args.synthetic_img_col] = synthetic_df[args.synthetic_img_col].apply(
        lambda x: os.path.join(args.synthetic_img_dir, x)
    )
    # Drop rows with duplicate prompts
    real_df = real_df.drop_duplicates(subset=[args.real_caption_col]).reset_index(
        drop=True
    )

    # Implement the logic for running analysis on conditional prompts i.e. Calculating metrics only for a specific pathology
    if(args.experiment_type == 'conditional'):
        print(f"Calculating metrics for the samples containing the pathology: {args.pathology}")
        real_df['chexpert_labels'] = real_df['chexpert_labels'].apply(get_labels_dict_from_string)

        # Create a separate column for pathology labels
        for col in MIMIC_PATHOLOGIES:
            real_df[col] = real_df['chexpert_labels'].apply(lambda x: x[col])
        
        # Fill NaN values with 0
        real_df.fillna(0, inplace=True)

        # Create a subset of the real dataset with the specified pathology
        real_df = real_df[real_df[args.pathology] == 1].reset_index(drop=True)
        
        # Include only those images from the synthetic dataset that have the same prompts as the real dataset containing the pathology
        real_prompts = real_df[args.real_caption_col].to_list()
        synthetic_df = synthetic_df[synthetic_df[args.synthetic_prompt_col].isin(real_prompts)].reset_index(drop=True)


    # Copy images to the temp folder (Real images)
    print("Copying real images to temporary folder: {}".format(image_folder1))

    worker = partial(copy_image, dest_folder=image_folder1)
    img_paths = real_df[args.real_img_col].tolist()
    with Pool(processes=8) as pool:
        pool.map(worker, img_paths)

    print("Copied {} real images to temporary folder.".format(len(real_df)))

    # Copy all files from the synthetic image CSV to the temp folder
    print("Copying synthetic images to temporary folder: {}".format(image_folder2))

    worker = partial(copy_image, dest_folder=image_folder2)
    img_paths = synthetic_df[args.synthetic_img_col].tolist()
    with Pool(processes=8) as pool:
        pool.map(worker, img_paths)
    print("Copied {} synthetic images to temporary folder.".format(len(synthetic_df)))

    radiomics_fname = 'radiomics.csv'
    radiomics_path1 = os.path.join(image_folder1, radiomics_fname)
    radiomics_path2 = os.path.join(image_folder2, radiomics_fname)

    # if needed, compute radiomics for the images
    if force_compute_fresh or not os.path.exists(radiomics_path1):
        print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder1, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder1, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 1.")
    else:
        print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

    if force_compute_fresh or not os.path.exists(radiomics_path2):
        print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder2, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder2, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 2.")
    else:
        print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2))

    # load radiomics
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)

    feats1, feats2 = convert_radiomic_dfs_to_vectors(
                        radiomics_df1, 
                        radiomics_df2,
                        match_sample_count=True, # needed for distance measures
                    ) 
    # Frechet distance
    fd = frechet_distance(feats1, feats2)
    frd = np.log(fd)

    print("FRD = {}".format(frd))

    # Save the results
    savename = "conditional_image_generation_metrics.csv" if args.experiment_type == 'conditional' else "image_generation_metrics.csv"
    if args.debug:
        savename = "debug_" + savename
    savepath = os.path.join(args.results_savedir, savename)

    # Try to read if the dataframe already exists
    if os.path.exists(savepath):
        print(f"Appending to existing results file found at {savepath}")
        results_df = pd.read_csv(savepath)
        results_df.loc[results_df.index[-1], 'FRD'] = frd

        if(args.experiment_type == 'conditional'):
            results_df.loc[results_df.index[-1], 'Pathology'] = args.pathology
        
        results_df.to_csv(savepath, index=False)
        print(f"FRD Score saved to: {savepath}")
    else:
        results = {
            "FRD": frd,
            "Extra Info": args.extra_info,
        }
        if(args.experiment_type == 'conditional'):
            results["Pathology"] = args.pathology

        print("Creating new results file.")
        results_df = pd.DataFrame([results])
        results_df.to_csv(savepath, index=False)
    
    if interpret:
        run_tsne = True
        interpret_radiomic_differences(radiomics_path1, radiomics_path2, run_tsne=run_tsne)

    return frd

    # Delete the temporary folder
    os.system(f"rm -rf {image_folder1}")
    os.system(f"rm -rf {image_folder2}")

if __name__ == "__main__":
    tstart = time()
    parser = ArgumentParser()

    # parser.add_argument('--image_folder1', type=str, required=True)
    # parser.add_argument('--image_folder2', type=str, required=True)
    
    # Paths for synthetic images
    parser.add_argument(
        "--synthetic_img_dir", type=str, help="Directory containing synthetic images."
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

    # Paths for real images
    parser.add_argument(
        "--real_img_dir", type=str, help="Directory containing real images."
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

    parser.add_argument('--force_compute_fresh', action='store_true', help='re-compute all radiomics fresh')
    parser.add_argument('--interpret', action='store_true', help='interpret the features underlying Fréchet Radiomics Distance')

    parser.add_argument(
        "--experiment_type", type=str, default=None, help="Type of experiment to run (regular, conditional)"
    )
    parser.add_argument(
        "--pathology", type=str, default='regular', help="Type of experiment to run (regular, conditional)"
    )
    
    args = parser.parse_args()

    main(
        # args.image_folder1,
        # args.image_folder2,
        args,
        force_compute_fresh=args.force_compute_fresh,
        interpret=args.interpret
        )

    tend = time()
    print("compute time (sec): {}".format(tend - tstart))