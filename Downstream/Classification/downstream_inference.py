import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast
from termcolor import colored
import random
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

def fix_state_dict(state_dict):
    new_state_dict = {}

    for k,v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v

    return new_state_dict

class MIMICCXRDataset(Dataset):
    def __init__(self, dataframe, image_col='path', label_cols='chexpert_labels', transform=None):
        
        self.dataframe = dataframe
        self.image_col = image_col
        self.transform = transform
        self.label_cols = label_cols

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.dataframe.iloc[idx][self.image_col]
            
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels (all 14 of them)
        label_vals = np.array(list(self.dataframe.iloc[idx][self.label_cols].values()))
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
        else:
            # Handle other model architectures
            in_features = self.model.get_classifier().in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def safe_collate_fn(batch):
    """
    Robust collate function that handles inconsistent label shapes.
    """
    images = []
    labels = []
    
    for sample in batch:
        image, label = sample
        
        # Validate tensors
        if image is None or label is None:
            continue
            
        if label.shape != torch.Size([14]):
            continue
            
        images.append(image)
        labels.append(label)
    
    if len(images) == 0:
        return None, None
    
    batched_images = torch.stack(images, dim=0)
    batched_labels = torch.stack(labels, dim=0)
    
    return batched_images, batched_labels


def calculate_metrics(outputs, labels, threshold=0.5, label_names=None):
    """
    Calculate metrics for multi-label classification.
    
    Args:
        outputs: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Threshold for binary classification
        label_names: List of label names for the 14 classes
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    pred_labels = (outputs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels.flatten(), pred_labels.flatten()),
        'precision_micro': precision_score(labels, pred_labels, average='micro', zero_division=0),
        'recall_micro': recall_score(labels, pred_labels, average='micro', zero_division=0),
        'f1_micro': f1_score(labels, pred_labels, average='micro', zero_division=0),
        'precision_macro': precision_score(labels, pred_labels, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, pred_labels, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, pred_labels, average='macro', zero_division=0),
    }
    
    # Calculate per-class metrics
    per_class_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for i in range(labels.shape[1]):
        per_class_metrics['precision'].append(
            precision_score(labels[:, i], pred_labels[:, i], zero_division=0))
        per_class_metrics['recall'].append(
            recall_score(labels[:, i], pred_labels[:, i], zero_division=0))
        per_class_metrics['f1'].append(
            f1_score(labels[:, i], pred_labels[:, i], zero_division=0))
        
        # Calculate AUC only if both classes are present
        if len(np.unique(labels[:, i])) > 1:
            per_class_metrics['auc'].append(roc_auc_score(labels[:, i], outputs[:, i]))
        else:
            per_class_metrics['auc'].append(float('nan'))
    
    # Add average AUC
    valid_aucs = [auc for auc in per_class_metrics['auc'] if not np.isnan(auc)]
    metrics['auc_macro'] = np.mean(valid_aucs) if valid_aucs else float('nan')
    
    # Add per-class metrics to overall metrics dictionary
    if label_names:
        for metric_name, values in per_class_metrics.items():
            for i, label_name in enumerate(label_names):
                metrics[f'{metric_name}_{label_name}'] = values[i]
    
    return metrics, per_class_metrics


def plot_confusion_matrices(labels, predictions, label_names, save_path):
    """
    Plot confusion matrices for each class.
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions (binary)
        label_names: List of label names
        save_path: Path to save the plot
    """
    num_classes = labels.shape[1]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(num_classes):
        if i < len(axes):
            cm = confusion_matrix(labels[:, i], predictions[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {label_names[i]}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(labels, outputs, label_names, save_path):
    """
    Plot ROC curves for each class.
    
    Args:
        labels: Ground truth labels
        outputs: Model outputs (probabilities)
        label_names: List of label names
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(12, 10))
    
    for i in range(labels.shape[1]):
        # Skip if there's only one class present
        if len(np.unique(labels[:, i])) <= 1:
            continue
        
        fpr, tpr, _ = roc_curve(labels[:, i], outputs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{label_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def run_inference(model, dataloader, device, threshold=0.5, label_names=None):
    """
    Run inference on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        threshold: Threshold for binary classification
        label_names: List of label names
        
    Returns:
        all_outputs: Model outputs (probabilities)
        all_labels: Ground truth labels
        metrics: Dictionary of metrics
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # Handle empty batches (from the collate function)
            if images is None or labels is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            outputs = outputs.sigmoid().cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    if not all_outputs:
        raise ValueError("No valid outputs were produced during inference.")
    
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    print(f"Inference completed on {len(all_labels)} samples")
    
    # Calculate metrics
    metrics, per_class_metrics = calculate_metrics(all_outputs, all_labels, threshold, label_names)
    
    return all_outputs, all_labels, metrics, per_class_metrics


def main(args):

    seed_everything(seed=42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)

    test_df[args.image_col] = test_df[args.image_col].apply(lambda x: os.path.join(args.real_image_dir, x))

    test_df['chexpert_labels'] = test_df['chexpert_labels'].apply(get_labels_dict_from_string)
    print(colored(f"Loaded test data from {args.test_csv}...", "green"))
    print("\n")

    label_cols = list(test_df['chexpert_labels'].iloc[0].keys())
    print(colored(f"Labels: {label_cols}", "cyan"))
    print("\n")

    if(args.debug):
        test_df = test_df.sample(n=args.debug_samples, random_state=42).reset_index(drop=True)

    print(colored(f"Num Test Samples: {len(test_df)}", "cyan"))
    print("\n")
    
    # Create dataset
    test_dataset = MIMICCXRDataset(test_df, image_col=args.image_col, transform=test_transform)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=safe_collate_fn
    )
    
    # Create model
    print(colored(f"Creating model with architecture {args.model_name}...", "cyan"))
    print("\n")
    model = MultiLabelClassifier(args.model_name, num_classes=14, pretrained=False)
    
    # Load checkpoint
    print(colored(f"Loading checkpoint from {args.checkpoint}...", "cyan"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # Might have to replace "module." from every key
            fixed_state_dict = fix_state_dict(dict(checkpoint['model_state_dict']))
            model.load_state_dict(fixed_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(colored("Model loaded successfully", "green"))
    print("\n")
    
    # Create output directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(current_dir, args.output_dir, args.extra_info, args.model_name) if args.extra_info else os.path.join(current_dir, args.output_dir, args.model_name, args.t2i_model)
    os.makedirs(args.output_dir, exist_ok=True)

    print(colored(f"Created Results Directory at {args.output_dir}", "green"))
    print("\n")
    
    # Run inference
    outputs, labels, metrics, per_class_metrics = run_inference(
        model, test_loader, device, args.threshold, label_cols
    )
    
    # Save metrics to output directory
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, f'metrics.csv_{args.t2i_model}'), index=False)
    
    # Create per-class metrics dataframe
    per_class_df = pd.DataFrame({
        'class': label_cols,
        'precision': per_class_metrics['precision'],
        'recall': per_class_metrics['recall'],
        'f1': per_class_metrics['f1'],
        'auc': per_class_metrics['auc']
    })
    per_class_df.to_csv(os.path.join(args.output_dir, f'per_class_metrics_{args.t2i_model}.csv'), index=False)
    
    # Print metrics
    print(colored("\nTest Metrics:", "cyan"))
    for metric, value in metrics.items():
        if not metric.startswith(('precision_', 'recall_', 'f1_', 'auc_')) or metric in ['precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']:
            print(f"  {metric}: {value:.4f}")
    
    print(colored("\nPer-class Performance:", "cyan"))
    for i, label in enumerate(label_cols):
        print(colored(f"  {label}:", "green"))
        print(f"    Precision: {per_class_metrics['precision'][i]:.4f}")
        print(f"    Recall: {per_class_metrics['recall'][i]:.4f}")
        print(f"    F1: {per_class_metrics['f1'][i]:.4f}")
        print(f"    AUC: {per_class_metrics['auc'][i]:.4f}")
    
    # Create binary predictions
    predictions = (outputs >= args.threshold).astype(int)
    
    # Plot confusion matrices
    plot_confusion_matrices(
        labels, 
        predictions, 
        label_cols, 
        os.path.join(args.output_dir, f'confusion_matrices_{args.t2i_model}.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        labels, 
        outputs, 
        label_cols, 
        os.path.join(args.output_dir, f'roc_curves_{args.t2i_model}.png')
    )
    
    # Save predictions if requested
    if args.save_predictions:
        # Create a dataframe with image paths and predictions
        pred_df = test_df[[args.image_col]].copy()
        
        # Add prediction probabilities
        for i, col in enumerate(label_cols):
            pred_df[f'{col}_prob'] = outputs[:, i]
            pred_df[f'{col}_pred'] = predictions[:, i]
        
        # Save to CSV
        pred_df.to_csv(os.path.join(args.output_dir, f'predictions_{args.t2i_model}.csv'), index=False)
        print(colored(f"Predictions saved to {os.path.join(args.output_dir, f'predictions_{args.t2i_model}.csv')}", "green"))
    
    print(colored(f"All results saved to {args.output_dir}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for MIMIC-CXR Multi-label Classification")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model architecture name")

    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--real_image_dir", type=str, default=None, help="Base Directory containing real images")
    parser.add_argument("--synthetic_image_dir", type=str, default=None, help="Base Directory containing synthetic images")
    parser.add_argument("--image_col", type=str, default="path", help="Column name in CSV that contains image paths")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to CSV")

    parser.add_argument("--cpu", action="store_true", help="Use CPU even if CUDA is available")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with a small subset of data")
    parser.add_argument("--debug_samples", type=int, default=500, help="Number of samples to use in debug mode")

    parser.add_argument("--extra_info", type=str, default=None, help="Extra info about an experiment")
    parser.add_argument("--t2i_model", type=str, default=None, help="Evaluation using the synthetic data from which T2I model.")
    
    args = parser.parse_args()
    main(args)