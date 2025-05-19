import os
import ast
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import random
from termcolor import colored
import warnings
from time import time
warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_elapsed_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Time Taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

class MIMICCXRDataset(Dataset):
    def __init__(self, image_path_list, labels_list, transform=None):
        self.image_path_list = image_path_list
        self.labels_list = labels_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_path_list[idx]
            
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels (all 14 of them)
        # labels = self.dataframe.iloc[idx][self.label_cols].values.astype(np.float32)
        label_vals = np.array(list(self.labels_list[idx].values()))
        labels = label_vals.astype(np.float32)
        labels = np.nan_to_num(labels)

        label_tensor = torch.tensor(labels, dtype=torch.float32)

        """
        Replace all instaces of -1 (denoting uncertainity for a condition) with 0
        """
        label_tensor = torch.where(label_tensor == -1, torch.tensor(0), label_tensor)        
        
        return image, label_tensor


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_classes=14, pretrained=True):
        super(MultiLabelClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features in the last layer
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            # Handle other model architectures
            in_features = self.model.get_classifier().in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

        if idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            all_outputs.append(outputs.sigmoid().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    return val_loss, all_outputs, all_labels


def calculate_metrics(outputs, labels, threshold=0.5):
    # Convert probabilities to binary predictions
    pred_labels = (outputs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': round(accuracy_score(labels.flatten(), pred_labels.flatten()), 3),
        'precision_micro': round(precision_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'recall_micro': round(recall_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'f1_micro': round(f1_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'precision_macro': round(precision_score(labels, pred_labels, average='macro', zero_division=0), 3),
        'recall_macro': round(recall_score(labels, pred_labels, average='macro', zero_division=0), 3),
        'f1_macro': round(f1_score(labels, pred_labels, average='macro', zero_division=0), 3),
    }
    
    # Calculate AUC for each class and average
    try:
        aucs = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) > 1:  # Only calculate AUC if there are both positive and negative samples
                aucs.append(roc_auc_score(labels[:, i], outputs[:, i]))
        
        if aucs:
            metrics['auc_macro'] = np.mean(aucs)
            metrics['auc_per_class'] = {i: auc for i, auc in enumerate(aucs)}
        else:
            metrics['auc_macro'] = float('nan')
    except:
        metrics['auc_macro'] = float('nan')
    
    return metrics

def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

def variable_size_collate_fn(batch):
    """
    Collate function for handling variable-sized images.
    Pads images to the maximum size in the batch.
    """
    # Filter out None values
    valid_samples = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]
    
    if len(valid_samples) == 0:
        return None, None
    
    images = []
    labels = []
    
    # Find max dimensions in this batch
    max_h = max([img.shape[1] for img, _ in valid_samples])
    max_w = max([img.shape[2] for img, _ in valid_samples])
    
    # Pad images to max dimensions
    for image, label in valid_samples:
        # Current image dimensions
        c, h, w = image.shape
        
        # Create new padded tensor
        padded_img = torch.zeros((c, max_h, max_w), dtype=image.dtype)
        padded_img[:, :h, :w] = image
        
        images.append(padded_img)
        
        # Handle any issues with labels
        if torch.isnan(label).any() or (label == -1).any():
            label = torch.nan_to_num(label, nan=0.0)
            label = torch.where(label == -1, torch.tensor(0.0), label)
            
        labels.append(label)
    
    # Stack the images and labels
    batched_images = torch.stack(images, dim=0)
    batched_labels = torch.stack(labels, dim=0)
    
    return batched_images, batched_labels

def create_data_from_train_setting(args, df, train_setting):

    """
    Returns a list of image paths and corresponding labels
    """

    ALL_TRAINING_SETTINGS = ['all_original', 'all_synthetic', 'mixed', 'augmented']
    assert train_setting in ALL_TRAINING_SETTINGS

    ####### All original #######
    if(train_setting == 'all_original'):
        image_col = 'path'
        df[image_col] = df[image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
        image_path_list = df[image_col].to_list()
        labels_list = df[args.labels_col].to_list()
        return image_path_list, labels_list

    ####### All synthetic #######
    elif(train_setting == 'all_synthetic'):
        image_col = 'synthetic_filename'
        df[image_col] = df[image_col].apply(lambda x: os.path.join(args.synthetic_image_dir, x))
        image_path_list = df[image_col].to_list()
        labels_list = df[args.labels_col].to_list()
        return image_path_list, labels_list

    elif(train_setting == 'mixed'):
        real_image_col = 'path'
        synthetic_image_col = 'synthetic_filename'

        df[real_image_col] = df[real_image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
        df[synthetic_image_col] = df[synthetic_image_col].apply(lambda x: os.path.join(args.synthetic_image_dir, x))

        # Replace 50% of the real data with synthetic data
        random.seed(42)
        all_ids = df['id'].to_list()
        ids_to_remove = random.sample(all_ids, k=int(0.5*len(df)))

        # Select real images not having these ids
        real_image_path_list = df[~df['id'].isin(ids_to_remove)][real_image_col].to_list()
        real_labels_list = df[~df['id'].isin(ids_to_remove)][args.labels_col].to_list()

        # Select synthetic images having the 'removed' ids
        synthetic_image_path_list = df[df['id'].isin(ids_to_remove)][synthetic_image_col].to_list()
        synthetic_labels_list = df[df['id'].isin(ids_to_remove)][args.labels_col].to_list()

        image_path_list = real_image_path_list + synthetic_image_path_list
        labels_list = real_labels_list + synthetic_labels_list

        return image_path_list, labels_list

    ####### Real Data fully augmented with synthetic data #######
    elif(train_setting == 'augmented'):
        real_image_col = 'path'
        synthetic_image_col = 'synthetic_filename'

        df[real_image_col] = df[real_image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
        df[synthetic_image_col] = df[synthetic_image_col].apply(lambda x: os.path.join(args.synthetic_image_dir, x))

        # Image Paths
        real_image_path_list = df[real_image_col].to_list()
        synthetic_image_path_list = df[synthetic_image_col].to_list()

        # Labels
        real_labels_list = df[args.labels_col].to_list()
        synthetic_labels_list = df[args.labels_col].to_list()

        image_path_list = real_image_path_list + synthetic_image_path_list
        labels_list = real_labels_list + synthetic_labels_list

        return image_path_list, labels_list
    
    else:
        raise ValueError(f"Invalid training setting. Got {args.training_setting} but args.training_setting should be in {ALL_TRAINING_SETTINGS}")

def main(args):
    seed_everything(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load MIMIC-CXR dataset from dataframe
    print("Loading MIMIC-CXR data...")
    
    # Load the dataframe containing paths and 14 labels
    df = pd.read_csv(args.train_csv)
    df[args.labels_col] = df[args.labels_col].apply(lambda x: get_labels_dict_from_string(x))
    # df[args.image_col] = df[args.image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
    # TODO:
    
    # NOTE: Preprocessing step: The labels are dictionaries but enclosed within strings in the CSV.
    try:
        df['chexpert_labels'] = df['chexpert_labels'].apply(get_labels_dict_from_string)
    except:
        pass

    if(args.debug):
        print("DEBUGGING!!!")
        print(f"Creating a subset of {args.debug_samples} samples")
        args.epochs = 3
        df = df.sample(n=args.debug_samples, random_state=42).reset_index(drop=True)
    
    """
    Prepare training data according to the training setting
    """
    image_paths_list, labels_list = create_data_from_train_setting(args, df, args.training_setting)

    label_cols = list(df['chexpert_labels'].iloc[0].keys())

    # import pdb; pdb.set_trace()
    
    
    # Split the data
    # train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=None)
    image_paths_list_train, image_paths_list_train_val, labels_list_train, labels_list_train_val = train_test_split(
                    image_paths_list,
                    labels_list,
                    test_size=0.05,     
                    random_state=42 
                )
    
    print(f"Training set size: {len(image_paths_list_train)}")
    print(f"Validation set size: {len(image_paths_list_train_val)}")
    print(f"Learning Rate: {args.learning_rate}")
    
    # Create datasets
    train_dataset = MIMICCXRDataset(image_paths_list_train, labels_list_train, transform=train_transform)
    val_dataset = MIMICCXRDataset(image_paths_list_train_val, labels_list_train_val, transform=val_transform)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=variable_size_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=variable_size_collate_fn)
    
    # Create model
    num_classes = 14  # Fixed for MIMIC-CXR dataset
    model = MultiLabelClassifier(args.model_name, num_classes, pretrained=True)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    # print(colored(f"{model}", "green"))
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    train_losses = []
    val_losses = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(f"{script_dir}/checkpoints/", exist_ok=True)
    args.output_dir = os.path.join(script_dir, "training_results", args.t2i_model)
    os.makedirs(args.output_dir, exist_ok=True)
    # os.makedirs(f"{script_dir}/training_results/{args.t2i_model}", exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        start_time = time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        # if(epoch + 1) % args.va_epochs == 0:
        val_loss, val_outputs, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
    
        # Calculate metrics
        metrics = calculate_metrics(val_outputs, val_labels, threshold=args.threshold)
        
        elapsed_time = time() - start_time
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        format_elapsed_time(elapsed_time)
        print(f"Metrics: ")
        for metric, value in metrics.items():
            if metric != 'auc_per_class':  # Don't print per-class AUC in the log
                print(f"  {metric}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': {k: v for k, v in metrics.items() if k != 'auc_per_class'},
            }, f"{script_dir}/checkpoints/{args.model_name}_{args.extra_info}_{args.t2i_model}.pth")
            print("Saved best model")
        
        # Update scheduler
        scheduler.step(val_loss)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    filename = f"{args.model_name}_loss_curves.png"
    plt.savefig(os.path.join(args.output_dir, filename))
    # plt.savefig(f"{script_dir}/training_results/{args.model_name}_loss_curves.png")
    
    # Print final best metrics
    print("\nBest Validation Performance:")
    for metric, value in best_metrics.items():
        if metric != 'auc_per_class':  # Don't print per-class AUC in the final summary
            print(f"  {metric}: {value:.4f}")
            
    # Print per-class AUC scores
    if 'auc_per_class' in best_metrics:
        print("\nPer-class AUC scores:")
        for class_idx, auc in best_metrics['auc_per_class'].items():
            class_name = label_cols[class_idx]
            print(f"  {class_name}: {auc:.4f}")

    # Wriwte the best metrics to a CSV file
    results_df = pd.DataFrame([best_metrics])
    filename = f"{args.model_name}_{args.extra_info}.csv"
    results_df.to_csv(os.path.join(args.output_dir, filename), index=False)
    # results_df.to_csv(f"{script_dir}/training_results/{args.model_name}_{args.extra_info}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label Classification on MIMIC-CXR")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name from timm")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--va_epochs", type=int, default=10, help="Number of epochs for validation")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--run", type=str, default="training", choices=['training', 'inference'], help="Run either training or inference.")

    parser.add_argument("--train_csv", type=str, required=True, help="Path to CSV with image paths and labels")
    parser.add_argument("--image_col", type=str, default="path", help="Column name in CSV that contains image paths")
    parser.add_argument("--labels_col", type=str, default="chexpert_labels", help="Column name in CSV that contains the labels")
    parser.add_argument("--real_image_dir", type=str, default=None, help="Base Directory containing images")
    parser.add_argument("--synthetic_image_dir", type=str, default=None, help="Base Directory containing synthetic images")

    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for data loading")
    parser.add_argument("--stratify", action="store_true", help="Whether to stratify the train/val split")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode with a small subset of data")
    parser.add_argument("--debug_samples", type=int, default=500, help="Number of samples to use in debug mode")

    parser.add_argument("--extra_info", type=str, default=None, help="Extra info about an experiment")

    parser.add_argument("--training_setting", type=str, default=None, help="Training setting") 
    parser.add_argument("--t2i_model", type=str, default=None, help="Evaluation using the synthetic data from which T2I model.")
    
    args = parser.parse_args()
    main(args)