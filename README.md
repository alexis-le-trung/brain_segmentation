# Brain MRI Segmentation with Cascaded U-Net

This project implements a cascaded U-Net architecture for automatic segmentation of brain MRI images. The goal is to accurately delineate brain structures from volumetric MRI scans.

## Project Overview

- **Data Handling:** The pipeline loads and preprocesses MRI data for training and evaluation.
- **Preprocessing:** Includes normalization, resizing, and augmentation to improve model robustness.
- **Model Architecture:** Utilizes a cascade of U-Net models, where each stage refines the segmentation output of the previous one.
- **Metrics:** Evaluates segmentation performance using metrics such as Dice coefficient, Jaccard index, and pixel-wise accuracy.
- **Utilities:** Helper functions for data management, visualization, and reproducibility.

## Folder Structure

- `src/`
  - `data.py` — Data loading and management.
  - `preprocessing.py` — Preprocessing and augmentation routines.
  - `models.py` — U-Net and cascade model definitions.
  - `metrics.py` — Implementation of evaluation metrics.
  - `main.py` — Training and evaluation script.
  - `utils.py` — Miscellaneous utility functions.

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   Place MRI images and labels in the appropriate folder.

3. **Train or evaluate models:**
   ```sh
   python src/main.py
   ```

## Results

| Metric            | CSV               |  GM               | WM                |
|-------------------|-------------------|-------------------|-------------------|
| Dice coefficient  |       0.8892      |       0.8228      |       0.7609      |
| ASD               |       0.2320      |       0.1984      |       0.3143      |
| MHD               |       0.3471      |       0.2436      |       0.4199      |

