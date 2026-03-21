# Robustness Ablation of Deep CV Models under Clean and Corrupted Validation  


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/GPU-CUDA-green?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Models-VGG%20%7C%20ResNet%20%7C%20ConvNeXt%20%7C%20ViT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Datasets-CIFAR10%20%7C%20FashionMNIST%20%7C%20ImageNet100-orange?style=for-the-badge" />
</p>

---

## 📌 Overview

This repository contains the complete implementation and experimental pipeline for **Deep Learning Assignment – 02**, where the main objective is to study the **robustness of computer vision models** under **clean** and **corrupted validation/test settings**.

The assignment investigates how checkpoint selection changes when:
- validation is performed on **clean data**, and
- validation is performed on **corrupted data**.

The study is carried out across:
- **4 backbone architectures**
- **3 datasets**
- **multiple corruption-selection policies**
- **visual analysis**
- **feature perturbation analysis**

---

## 🎯 Assignment Objective

The central goal of this assignment is:

> To train multiple deep image classification models on standard datasets, compare model selection under clean and corrupted validation conditions, evaluate robustness on corrupted test inputs, and analyze the effect of perturbations through feature-space and feature-map visualizations.

---

## 🧠 Models Used

The following architectures were used:

| Model | Family | Role in Study |
|------|--------|---------------|
| **VGG16-BN** | Classical CNN | Baseline deep convolution model |
| **ResNet50** | Residual CNN | Strong CNN with skip connections |
| **ConvNeXt-Tiny** | Modern CNN | Modernized convolution backbone |
| **ViT-B/16** | Transformer | Vision Transformer baseline |

---

## 🗂️ Datasets Used

| Dataset | Classes | Notes |
|--------|---------|------|
| **CIFAR-10** | 10 | RGB natural image dataset |
| **Fashion-MNIST** | 10 | Grayscale fashion article dataset, converted to RGB |
| **ImageNet-100** | 100 | Local folder version with `train/` and `val/` splits |

---

## ⚙️ Experimental Setup

### Data split
A fixed **20% validation split** was created from the training set and reused across all settings.

### Input pipeline
All data were standardized to:
- **3 channels**
- **224 × 224** input size

### Training strategy
Each model-dataset pair was trained once, and multiple selector checkpoints were derived from the same training trajectory.

### Validation selector settings
The implemented fast configuration uses the following selector settings:
- `clean`
- `corr_k1_s2`
- `corr_k3_s2`
- `corr_k5_s2`

Where:
- `corr` = corrupted validation
- `k1`, `k3`, `k5` = number of corruption types averaged
- `s2` = severity level 2

### Corruptions used
The accelerated experiment uses 5 corruption types:

1. `gaussian_noise`
2. `motion_blur`
3. `fog`
4. `brightness`
5. `jpeg_compression`

---

# 📖 Assignment Questions and How They Were Addressed

---

## Q1. Train VGG, ResNet, ConvNeXt, and ViT on CIFAR-10, Fashion-MNIST, and ImageNet-100

### What was done
All required models were trained on all three datasets.

This created a total of:

- **4 models × 3 datasets = 12 experiment pairs**

### Experiment pairs
- CIFAR-10 + VGG
- CIFAR-10 + ResNet
- CIFAR-10 + ConvNeXt
- CIFAR-10 + ViT
- Fashion-MNIST + VGG
- Fashion-MNIST + ResNet
- Fashion-MNIST + ConvNeXt
- Fashion-MNIST + ViT
- ImageNet-100 + VGG
- ImageNet-100 + ResNet
- ImageNet-100 + ConvNeXt
- ImageNet-100 + ViT

---

## Q2. Use 20% of the training set as validation and keep it fixed

###  What was done
A fixed **stratified 20% validation split** was created from the training set.

### Why it matters
This ensures:
- fair comparison across selector settings,
- reproducibility,
- stable evaluation protocol.

### Output artifacts
Saved split files:
- `splits/cifar10_seed42_val20.json`
- `splits/fashion_mnist_seed42_val20.json`
- `splits/imagenet100_seed42_val20.json`

---

## Q3. Compare clean validation vs corrupted validation

###  What was done
Model selection was compared under:
- **clean validation**
- **corrupted validation**

The clean selector:
- used validation accuracy on clean validation samples.

The corrupted selectors:
- used validation accuracy averaged over corrupted versions of the validation data.

### Implemented selector names
- `clean`
- `corr_k1_s2`
- `corr_k3_s2`
- `corr_k5_s2`

---

## Q4. Study how much corruption should be used in validation

###  What was done
Instead of using only one corrupted validation setting, multiple corruption-set sizes were studied:

- `K = 1`
- `K = 3`
- `K = 5`

### Meaning
- `corr_k1_s2`: average over first 1 corruption type
- `corr_k3_s2`: average over first 3 corruption types
- `corr_k5_s2`: average over all 5 selected corruption types


> **Note:** This repository uses a **5-corruption accelerated protocol**.

---

## Q5. Report clean and corrupted test accuracy

###  What was done
For every saved selector checkpoint:
- clean test accuracy was computed
- corrupted test accuracy was computed
- mean corrupted accuracy was summarized
- robustness comparison tables were generated

### Main result tables
- `full_assignment_all_selectors.csv`
- `full_assignment_clean_rows.csv`
- `full_assignment_best_corrupt_rows.csv`
- `clean_vs_best_corrupt_comparison.csv`
- `report_all_selector_robustness.csv`
- `report_clean_accuracy_comparison.csv`
- `report_robustness_comparison.csv`

---

## Q6. Visualize the impact of clean vs corrupted validation settings

###  What was done
A dedicated visualization pipeline was executed for **all selectors across all completed pairs**.

### Generated visual artifacts
For each selector, the following were produced:

- confusion matrix on clean test
- confusion matrix on corrupted test
- feature-space projection on clean test
- feature-space projection on corrupted test
- projection decision region (2D proxy)
- feature-map visualization on clean image
- feature-map visualization on corrupted image
- classification reports in CSV format

---

## Q7. Optional / intermediate feature perturbation study

###  What was done
A separate notebook was used to perturb intermediate feature representations and study their effect on robustness.

### Feature perturbation settings
Perturbations were applied at:
- early layers
- middle layers
- late layers

### Output files
- `feature_perturbation_master.csv`
- `feature_perturbation_cifar10_resnet.csv`
- `feature_perturbation_fashion_mnist_resnet.csv`
- `feature_perturbation_imagenet100_resnet.csv`
- `feature_perturbation_errors.csv`


---

# 🏗️ Pipeline Summary

## Step 1 — Dataset preparation
All datasets were prepared locally first:
- CIFAR-10
- Fashion-MNIST
- ImageNet-100

This avoided repeated downloads across training notebooks.

## Step 2 — One notebook per pair
Instead of one monolithic notebook, 12 pair-specific notebooks were used.

### Why
- easier debugging
- easier parallel execution
- better GPU job control
- cleaner checkpoint management

## Step 3 — Train once, select many
Each model-dataset pair was trained once, while multiple selector checkpoints were evaluated from the same training history.

### Why
This is much more efficient than retraining for every selector.

## Step 4 — Aggregate results
The pair outputs were combined into final report tables using the aggregation notebook.

## Step 5 — Generate visuals
A separate notebook generated visuals for all selectors across all pairs.

## Step 6 — Feature perturbation
An additional notebook handled the intermediate feature perturbation study.

---

# 📊 Main Result Files

## Final aggregated CSVs
- `tables/full_assignment_all_selectors.csv`
- `tables/full_assignment_clean_rows.csv`
- `tables/full_assignment_best_corrupt_rows.csv`
- `tables/clean_vs_best_corrupt_comparison.csv`
- `tables/report_all_selector_robustness.csv`
- `tables/report_clean_accuracy_comparison.csv`
- `tables/report_robustness_comparison.csv`

## Visual-generation logs
- `tables/all_selector_visual_generation_log.csv`
- `tables/all_selector_visual_generation_errors.csv`

## Feature perturbation outputs
- `tables/feature_perturbation_master.csv`
- `tables/feature_perturbation_errors.csv`

---

# 📁 Repository Structure

```text
.
├── assignment2_gpu_split_module.py
├── 00_prepare_all_datasets_once.ipynb
├── 01_cifar10_vgg_gpu_local_fast5.ipynb
├── 02_cifar10_resnet_gpu_local_fast5.ipynb
├── ...
├── 12_imagenet100_vit_gpu_local_fast5.ipynb
├── 13_generate_all_selector_visuals_fast5.ipynb
├── 14_aggregate_all_results_fast5_UPDATED.ipynb
├── 15_feature_perturbation_ablation_fast5_UPDATED.ipynb
├── tables/
├── plots/
├── splits/
└── report/
