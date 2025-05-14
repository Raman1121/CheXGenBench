# CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs

![](assets/images/chexgenbench-overview.png)

<!-- ## Contents -->

<!-- - [Overview](#overview)
    - [Quantitative Analysis](#quantitative-analysis)
        - [Generative Quality Metrics](#generative-quality-metrics)
        - [Diversity and Mode Coverage Metrics](#diversity-and-mode-coverage-metrics)
    - [Qualitative Analysis](#qualitative-analysis)
- [Usage](#usage)
    - [Generating Synthetic Data](#generating-synthetic-data)
    - [Calculating Global Metrics](#calculating-global-metrics)
    - [Calculating Conditional Metrics](#calculating-conditional-metrics)
    - [Calculating Privacy Metrics](#privacy-metrics) -->

# Environment Setup
- Python>=3.10.0
- Pytorch>=2.0.1+cu12.1
```
git clone https://github.com/Raman1121/CheXGenBench.git
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

## Generating Synthetic Data

### Training Text-to-Image Models
We trained 11 different models for this work and the checkpoints are released [here](https://huggingface.co/collections/raman07/chexgenbench-models-6823ec3c57b8ecbcc296e3d2). Training of T2I models de-coupled from this repository i.e. You can train your favourite T2I model using any framework of choice [Diffusers](https://github.com/huggingface/diffusers), [ai-toolkit](https://github.com/ostris/ai-toolkit), etc. 

- **Downloading Training Images:** Download the MIMIC-CXR Dataset after accepting the license from [here](https://physionet.org/content/mimic-cxr/2.0.0/).
- **Using LLaVA-Rad Annotations:** We used LLaVA-Rad Annotations because of enhanced caption quality. They are presented in the `MIMIC_Splits/` folder.
    - Training CSV: `MIMIC_Splits/LLAVARAD_ANNOTATIONS_TRAIN.csv`
    - Test CSV: `MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv`

- **Data Organization:** Once training is finished, use the `MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv` file to generate images for evaluation. Ensure that during generation, you save both the **original prompt** and the generated **synthetic image**. Organize this data into a CSV file with the following columns:
    - `'prompt'`: Contains the text prompt used for generation.
    - `'img_savename'`: Contains the filename (or path) of the saved synthetic image.
- **File Placement:** After generating all the synthetic images and creating the CSV file:
    - Place the generated CSV file containing the prompts and image filenames in the `assets/CSV` directory.
    - Place all the generated synthetic image files in the `assets/synthetic_images` directory.

# Usage

This section provides instructions on how to use the benchmark to evaluate your Text-to-Image model's synthetic data.

## Quantitative Analysis: Generation Fidelity

![](assets/images/sana-performance.png)

The quantitative analysis assesses the synthetic data at two distinct levels to provide a granular understanding of its quality:

**Overall Analysis:** This level calculates metrics across the entire test dataset, consisting of all pathologies present in the MIMIC dataset. It provides a general indication of the synthetic data's overall quality.

```bash
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics.sh
```

**Important Note:** Calculating metrics like FID and KID can be computationally intensive and may lead to "Out of Memory" (OOM) errors, especially with large datasets (If using V100 GPUs or lower). If you encounter this issue, you can use the memory-saving version of the script:-

```bash
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics_memory_saving.sh
```

The results would be stored in `Results/image_generation_metrics.csv`

**Conditional Analysis:** This level calculates each metric separately for each individual pathology present in the dataset. This allows for a detailed assessment of how well the T2I model generates synthetic data for specific medical conditions.

```bash
cd Benchmarking-Synthetic-Data
./scripts/image_quality_metrics_conditional.sh
```
The results would be stored in `Results/conditional_image_generation_metrics.csv`

<div style="border: 1px solid #ccc; padding: 10px; background-color: #e7f3fe;">
  <strong>Tip:</strong> Enhance your results by providing additional information about the model or specific checkpoint used for generating the synthetic data. You can typically do this by setting the <code>EXTRA_INFO</code>argument when running the scripts (refer to the example scripts for specific usage).
</div>

## Quantitative Analysis: Privacy Metrics

![](assets/images/Privacy-Metrics.png)

Run the following script to calculate privacy and patient re-identification metrics.
```bash
cd Benchmarking-Synthetic-Data
./scripts/privacy_metrics.sh
```

## Quantitative Analysis: Downstream Utility

### Image Classification

For image classification, we used 20,000 samples from the MIMIC Dataset for training. To evaluate, you first need to generate synthetic samples using the same 20,000 prompts with your T2I Model. The prompts are provided in `MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv`.

### Radiology Report Generation

To fine-tune LLaVA-Rad, the first step is creating a *new* environment following the steps mentioned in the official [LLaVA-Rad repository](https://github.com/microsoft/LLaVA-Rad).