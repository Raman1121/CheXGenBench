# Source: https://github.com/RichardObi/frd-score/blob/main/frd_v1/src/radiomics_utils.py

from radiomics import featureextractor

import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from multiprocess import Pool
from random import sample
import logging
import warnings
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import sys
# sys.path.append('..')
from utils import frechet_distance, plot_tsne

logger = logging.getLogger('radiomics.imageoperations')
logger.setLevel(logging.ERROR)

def compute_slice_radiomics(img_slice, mask_slice, params_file='configs/2D_extraction.yaml'):
    device = 'cpu' # FIXME: GPU implementation is too slow, possibly bugged

    # check if params_file exists, else change path to parent of parent dir of this script
    if not os.path.exists(params_file):
        params_file = os.path.join("frd_v1", params_file)
    if not os.path.exists(params_file):
        params_file = os.path.join(os.path.dirname(__file__), params_file)
    if not os.path.exists(params_file):
        params_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), params_file)
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Radiomics parameters file {params_file} does not exist. ")

    # assume pixel size is in 1x1 mm. I am gonna assume 3rd dimension which is the depth is also 1 mm so that it is isotropic
    data_spacing = [1,1,1]

    sitk_img = sitk.GetImageFromArray(img_slice)
    sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
    sitk_img = sitk.JoinSeries(sitk_img)

    sitk_mask = sitk.GetImageFromArray(mask_slice)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT

    # prepare the settings and load
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    #extract 
    features = extractor.execute(sitk_img, sitk_mask)

    return features

def convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                    radiomics_df2,
                                    match_sample_count=False,
                                    return_image_fnames=False,
                                    return_feature_names=False,
                                    normalize=True,
                                    exclude_features=None
                                    ):
    """
    Convert radiomics dataframes to numpy arrays and normalize them wrt the real radiomics data
    also possibly remove features that are NaN in either of the arrays, and use random removal to match the sample count
    """
    imgfnames1 = radiomics_df1['img_fname'].values
    imgfnames2 = radiomics_df2['img_fname'].values

    # exclude shape-based radiomics which are for object-level radiomics
    # iterate through column names
    for col in radiomics_df1.columns:
        check = "shape2D" in col
        if check:
            radiomics_df1 = radiomics_df1.drop(columns=col)
            radiomics_df2 = radiomics_df2.drop(columns=col)

    if exclude_features is not None:
        if exclude_features == "textural":
            print("EXCLUDING TEXTURAL RADIOMICS.")
            # iterate through column names
            for col in radiomics_df1.columns:
                colsplit = col.split('_')
                check = (colsplit[0].split('-')[0] in ["original", "wavelet"]) and (colsplit[1].startswith("gl"))
                if check:
                    radiomics_df1 = radiomics_df1.drop(columns=col)
                    radiomics_df2 = radiomics_df2.drop(columns=col)
        
        elif exclude_features == "wavelet":
            print("EXCLUDING WAVELET RADIOMICS.")
            # iterate through column names
            for col in radiomics_df1.columns:
                check = col.startswith("wavelet")
                if check:
                    radiomics_df1 = radiomics_df1.drop(columns=col)
                    radiomics_df2 = radiomics_df2.drop(columns=col)

        elif exclude_features == "firstorder":
            print("EXCLUDING FIRST ORDER RADIOMICS.")
            # iterate through column names
            for col in radiomics_df1.columns:
                check = "firstorder" in col
                if check:
                    radiomics_df1 = radiomics_df1.drop(columns=col)
                    radiomics_df2 = radiomics_df2.drop(columns=col)

        else:
            raise NotImplementedError(f"Invalid exclude_features argument: {exclude_features}. Select one out of 'firstorder', 'wavelet', 'textural'.")
        
    # remove NaN and string radiomics
    # Identify columns with string data type
    radiomics_df1 = radiomics_df1.drop(columns=radiomics_df1.select_dtypes(include=['object']).columns)
    radiomics_df2 = radiomics_df2.drop(columns=radiomics_df2.select_dtypes(include=['object']).columns)

    radiomics_df1 = radiomics_df1.dropna()
    radiomics_df2 = radiomics_df2.dropna()

    if return_feature_names:
        assert radiomics_df1.columns.equals(radiomics_df2.columns)
        feature_names = radiomics_df1.columns

    # convert radiomics to arrays
    feats1 = radiomics_df1.to_numpy().astype(np.float32)
    feats2 = radiomics_df2.to_numpy().astype(np.float32)

    # match first dimension by random removal
    if match_sample_count:
        if feats1.shape[0] > feats2.shape[0]:
            mask = np.random.choice(feats1.shape[0], feats2.shape[0], replace=False)
            feats1 = feats1[mask]
            imgfnames1 = imgfnames1[mask]
        elif feats2.shape[0] > feats1.shape[0]:
            mask = np.random.choice(feats2.shape[0], feats1.shape[0], replace=False)
            feats2 = feats2[mask]
            imgfnames2 = imgfnames2[mask]

    # normalize features in these arrays wrt first feature dist
    if normalize:
        mean = np.mean(feats1, axis=0)
        std = np.std(feats1, axis=0)

        #print("normalization stats (mean, std): {} {}".format(mean, std))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats1 = (feats1 - mean) / std
            feats2 = (feats2 - mean) / std

    # remove features from both arrays that are NaN or inf in either
    nan_features_mask = np.isnan(feats1).any(axis=0) | np.isnan(feats2).any(axis=0) | np.isinf(feats1).any(axis=0) | np.isinf(feats2).any(axis=0)
    feats1 = feats1[:, ~nan_features_mask]
    feats2 = feats2[:, ~nan_features_mask]
    if return_feature_names:
        feature_names = feature_names[~nan_features_mask]
        assert len(feature_names) == feats1.shape[1] and len(feature_names) == feats2.shape[1]

    ret = [feats1, feats2]
    if return_image_fnames:
        assert len(imgfnames1) == feats1.shape[0] and len(imgfnames2) == feats2.shape[0]
        ret += [imgfnames1, imgfnames2]
    
    if return_feature_names:
        ret += [feature_names]
    
    return ret

def compute_and_save_imagefolder_radiomics(
        img_dir,
        radiomics_fname='radiomics.csv',
        subset=None
):
    out_dir = img_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    radiomics_csv_fname = os.path.join(out_dir, radiomics_fname)

    img_filenames = []
    radiomics = []

    img_fnames = os.listdir(img_dir)
    if subset is not None:
        img_fnames = sample(img_fnames, subset)

    # pyradiomics usage with numpy following https://github.com/AIM-Harvard/pyradiomics/issues/449
    # assume 2D images here
    for img_idx, img_fname in tqdm(enumerate(img_fnames), total=len(img_fnames)):
        #if img_idx > 10:
        #    break

        if not img_fname.split('.')[-1] in ['png', 'jpeg', 'jpg']:
            continue
        img_slice = np.asarray(Image.open(os.path.join(img_dir, img_fname)).convert('L')).copy()
        mask_slice = np.ones_like(img_slice)
        # to prevent bug in pyradiomics https://github.com/AIM-Harvard/pyradiomics/issues/765#issuecomment-1116713745
        mask_slice[0][0] = 0

        features = {}
        try:
            features = compute_slice_radiomics(img_slice, mask_slice)
        except RuntimeError:
            continue

        radiomics.append(features)
        img_filenames.append(img_fname)
    
    # bring all data together and save
    radiomics_df = pd.DataFrame(radiomics)
    radiomics_df.insert(loc=0, column='img_fname', value=img_filenames)
    # save df as a csv
    radiomics_df.to_csv(radiomics_csv_fname, index=False)

    print("saved radiomics to {}".format(radiomics_csv_fname))

    return radiomics_df

def compute_and_save_imagefolder_radiomics_parallel(
        img_dir,
        radiomics_fname='radiomics.csv',
        subset=None,
        num_workers=16
):
    out_dir = img_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    radiomics_csv_fname = os.path.join(out_dir, radiomics_fname)
    
    img_fnames = os.listdir(img_dir)
    if subset is not None:
        img_fnames = sample(img_fnames, subset)

    def get_radiomics_feature(split, img_list):
        radiomics = []
        img_filenames = []
        split_num = len(img_list) // num_workers
        img_list = img_list[split_num * split : split_num * (split + 1)]

        # pyradiomics usage with numpy following https://github.com/AIM-Harvard/pyradiomics/issues/449
        # assume 2D images here
        for img_idx, img_fname in enumerate(tqdm(img_list, total=len(img_list))):
            #if img_idx > 5:
            #    break
            if not img_fname.split('.')[-1] in ['png', 'jpeg', 'jpg']:
                continue
            img_slice = np.asarray(Image.open(os.path.join(img_dir, img_fname)).convert('L')).copy()
            mask_slice = np.ones_like(img_slice)
            # to prevent bug in pyradiomics https://github.com/AIM-Harvard/pyradiomics/issues/765#issuecomment-1116713745
            mask_slice[0][0] = 0

            features = {}
            try:
                features = compute_slice_radiomics(img_slice, mask_slice)
            except RuntimeError:
                continue
            radiomics.append(features)
            img_filenames.append(img_fname)

        return radiomics, img_filenames

    # Multi-Process
    pool = Pool()
    result_list = []
    imgs = os.listdir(img_dir)
    for i in range(num_workers):
        result = pool.apply_async(get_radiomics_feature, [i, img_fnames])
        result_list.append(result)
    
    radiomics = []
    img_filenames = []
    num_skipped = 0
    for r in result_list:
        try:
            radiomics_sub, filenames_sub = r.get(timeout=100000)
            radiomics += radiomics_sub
            img_filenames += filenames_sub
        except ValueError:
            num_skipped += 1
    
    # bring all data together and save
    radiomics_df = pd.DataFrame(radiomics)
    radiomics_df.insert(loc=0, column='img_fname', value=img_filenames)
    # save df as a csv
    radiomics_df.to_csv(radiomics_csv_fname, index=False)

    print("saved radiomics to {}".format(radiomics_csv_fname))
    if num_skipped != 0:
        print("had to skip {} images due to errors.".format(num_skipped))

    return radiomics_df

def compute_normalized_frd(feats1, feats2, val_frac=0.1):

    # randomly split feats1 into train (reference set) and val (Establish distance dist), via val_frac
    val_idx = np.random.choice(feats1.shape[0], int(val_frac*feats1.shape[0]), replace=False)
    train_idx = np.array([i for i in range(feats1.shape[0]) if i not in val_idx])
    in_activations_val = feats1[val_idx]
    in_activations = feats1[train_idx]

    id_mean = in_activations.mean(dim=0)

    ID_scores_val = np.stack([np.linalg.norm(id_mean - out, axis=0) for out in in_activations_val])
    ID_scores_val = ID_scores_val.detach().numpy()

    scores = np.stack([np.linalg.norm(id_mean - out, axis=0) for out in feats2])
    scores = scores.detach().numpy()

    all_scores = np.concatenate([ID_scores_val, scores])
    all_labels = np.concatenate([np.zeros(len(ID_scores_val)), np.ones(len(scores))])
    auc_dev = 2. * np.abs(roc_auc_score(all_labels, all_scores) - 0.5)

    return auc_dev

def interpret_radiomic_differences(
        radiomics_path1,
        radiomics_path2,
        run_tsne = True,
        viz_folder = 'outputs/interpretability_visualizations'
):
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    # load radiomics and convert to numpy arrays
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)
    feats1, feats2, imgfnames1, imgfnames2, feature_names = convert_radiomic_dfs_to_vectors(radiomics_df1, radiomics_df2, return_image_fnames=True, return_feature_names=True)
    # note: feats are normalized wrt the first radiomics df

    # visualize radiomic feature representations using t-SNE 
    if run_tsne:
        all_feats = {'radiomic': (feats1, feats2)}
        for feature_name, (f1, f2) in all_feats.items():
            plot_tsne(f1, f2, feature_name, viz_folder=viz_folder)


    # in case some images are jpegs and others are pngs,
    imgfnames1 = np.array([os.path.splitext(fname)[0] for fname in imgfnames1])
    imgfnames2 = np.array([os.path.splitext(fname)[0] for fname in imgfnames2])
    radiomics_df1['img_fname'] = [os.path.splitext(fname)[0] for fname in radiomics_df1['img_fname']]

    # if images are paired (i.e. between source domain and translated domain):
    # only use features that are present in both dataframes
    final_feats1 = []
    final_feats2 = []
    final_imgfnames = []
    for img_fname in radiomics_df1['img_fname'].values:
        # print(img_fname in imgfnames1, img_fname in imgfnames2)
        if img_fname in imgfnames1 and img_fname in imgfnames2:
            final_feats1.append(feats1[imgfnames1 == img_fname])
            final_feats2.append(feats2[imgfnames2 == img_fname])
            final_imgfnames.append(img_fname)

    if len(final_imgfnames) > 0:
        print("Dataset is paired, analyzing...".format(len(final_imgfnames)))

        feats1 = np.array(final_feats1).squeeze()
        feats2 = np.array(final_feats2).squeeze()
        imgfnames = np.array(final_imgfnames)

        # find images that differ most in radiomics between the two features
        # in squared l2 norm
        difference = feats1 - feats2

        # plot distribution of squared l2 radiomic differences across all images, 
        # averaged across all radiomic features
        avg_abs_differences = np.mean(difference**2, axis=1)
        sorted_indices = avg_abs_differences.argsort()[::-1]
        x = np.arange(len(avg_abs_differences))
        y = avg_abs_differences[sorted_indices]
        plt.figure(figsize=(4,2), dpi=300)
        plt.plot(x, y, '-', color='cornflowerblue')
        # plt.xscale('log')
        #plt.yscale('log')

        plt.title('Which images changed\nthe most/least between the datasets?')
        plt.xlabel('sorted image index')
        plt.ylabel('$||\Delta h||_2$ (img. level)')
        plt.grid()
        plt.savefig(os.path.join(viz_folder, "sorted_radiomic_image_differences.png"), bbox_inches="tight")
        plt.show()


        # plot the images that differ most and least
        num_images = 4
        fig, axs = plt.subplots(num_images//2, 2, figsize=(3, num_images*0.75), dpi=300)

        axs[0, 0].text(0, -50, "Input->Output images that changed\nthe most between datasets:", fontsize=8)
        # plot images with highest differences
        show_image_fnames = False
        for i in range(num_images//2):
            idx = sorted_indices[i]

            img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img1_fname):
                img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img2_fname):
                img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img1 = plt.imread(img1_fname)
            img2 = plt.imread(img2_fname)

            axs[i, 0].imshow(img1, cmap='gray')
            axs[i, 1].imshow(img2, cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            if show_image_fnames:
                axs[i, 0].set_title(imgfnames[idx])

        plt.savefig(os.path.join(viz_folder, "images_with_highest_radiomic_differences.png"), bbox_inches="tight")
        plt.close()
                
        fig, axs = plt.subplots(num_images//2, 2, figsize=(3, num_images*0.75), dpi=300)
        axs[0, 0].text(0, -50, "Input->Output images that changed\nthe least between datasets:", fontsize=8)
        for i in range(num_images//2):
            idx = sorted_indices[len(sorted_indices) - 1 - (num_images//2) + i]
            #print(len(sorted_indices) - 1 - (num_images//2) + i)
            img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img1_fname):
                img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img2_fname):
                img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"

            img1 = plt.imread(img1_fname)
            img2 = plt.imread(img2_fname)

            axs[i, 0].imshow(img1, cmap='gray')
            axs[i, 1].imshow(img2, cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            if show_image_fnames:
                axs[i, 0].set_title(imgfnames[idx])

        plt.savefig(os.path.join(viz_folder, "images_with_lowest_radiomic_differences.png"), bbox_inches="tight")

    # do the same type of plot but for each radiomic feature, averaged across all images
    # (see which radiomic features differ most between the two datasets in terms of squared l2 norm)

    #avg_abs_differences = np.mean(difference**2, axis=0)
    avg_abs_differences = (np.mean(feats2, axis=0) - np.mean(feats1, axis=0))**2.
    sorted_indices = avg_abs_differences.argsort()[::-1]
    x = np.arange(len(avg_abs_differences))
    y = avg_abs_differences[sorted_indices]
    plt.figure(figsize=(4,2), dpi=300)
    plt.plot(x, y, '-', color='indianred')
    # plt.xscale('log')
    plt.yscale('log')

    plt.title('How many radiomic features\nchanged noticeably between the datasets?')
    plt.xlabel('sorted feature index $j$')
    plt.ylabel('$|\Delta h|^j$ (dist. level)')
    plt.grid()
    # plot vertical line capturing 90% of the total difference
    total_diff = np.sum(y)
    ninety_percent_diff = 0.9 * total_diff
    cumsum_diff = np.cumsum(y)
    ninety_percent_idx = np.argmax(cumsum_diff > ninety_percent_diff)
    print("90% of the total difference is captured by the first {} features.".format(ninety_percent_idx+1))
    plt.axvline(x=ninety_percent_idx, color='k', linestyle='--')

    plt.savefig(os.path.join(viz_folder, "sorted_radiomic_feature_differences.png"), bbox_inches="tight")

    # get feature names in order of sorted indices
    sorted_feature_names = feature_names[sorted_indices]
    k = 10
    # print top k changed features
    print("Top {} changed radiomic features, with feature change values:".format(k))
    for i in range(k):
        print('{}: {:.1f}'.format(sorted_feature_names[i], np.log(y[i])))

    return