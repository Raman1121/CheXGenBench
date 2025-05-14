---
license: apache-2.0
language:
- en
tags:
- medical
size_categories:
- 10K<n<100K
---

# SynthCheX-75K

SynthCheX-75K is a synthetic dataset developed from Sana (0.6B) [1], a highly-capable text-to-image model, fine-tuned on chest radiographs. The dataset contains 75,649 high-quality samples after a strict filteration process. This dataset is released with the paper: **CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs**.

### Filtration Process for SynthCheX-75K

Generative models can lead to both high and low-fidelity generations on different subsets of the dataset. In order to keep the sample quality high in SynthCheX-75K, a stringent filtration process was adopted using HealthGPT [2], a highly-capable medical VLM with advanced understanding, reasoning and generation.

The VLM was provided with the following meta-prompt to classify the quality of each generated sample. 

**META PROMPT:** __"You are an expert radiologist and medical annotator. Your task is to assess the quality of an image given its description and classify the image as either 'High Quality', 'Medium Quality', or 'Low Quality'. Keep your responses limited to only these three options. If the image is not relevant to the description, respond with 'Not Relevant'."__

After quality label assignment, images with "*Low Quality*" and "*Not Relevant*" labels were removed from the dataset leading to 75,649 samples of high-quality radiographs with pathological annotations.

### Dataset Usage

```python
from datasets import load_dataset
dataset = load_dataset("raman07/SynthCheX-75K", trust_remote_code=True)
print(dataset['train'][0])

{'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512>,
 'filename': 'SyntheticImg_02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png',
 'prompt': 'There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.',
 'labels_dict': {
    'Atelectasis': 0,
    'Cardiomegaly': 0,
    'Consolidation': 0,
    'Edema': 0,
    'Enlarged Cardiomediastinum': 0,
    'Fracture': 0,
    'Lung Lesion': 0,
    'Lung Opacity': 0,
    'No Finding': 1,
    'Pleural Effusion': 0,
    'Pleural Other': 0,
    'Pneumonia': 0,
    'Pneumothorax': 0,
    'Support Devices': 0
    }
}
```