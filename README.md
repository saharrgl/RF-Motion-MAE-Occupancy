# Multimodal MAE-Based Occupancy Detection Using WiFi CSI and Motion Features

This repository contains the implementation of a **Masked Autoencoder (MAE)–based multimodal framework** for **smart building occupancy detection** using **WiFi Channel State Information (CSI)** and **derived motion features**.

The project investigates how **self-supervised pretraining** with MAE improves robustness against **noise, missing data, and environmental variability**, compared to traditional RF-only and motion-only baselines.

---

## Project Overview

Smart homes require reliable occupancy detection to:
- reduce energy consumption,
- enable intelligent automation,
- improve spectrum and WiFi resource management.

However, existing approaches suffer from:
- noisy RF measurements,
- sensitivity to stationary occupants,
- missing or corrupted CSI frames,
- heavy dependence on labeled data.

To address these challenges, this project proposes a **two-stage framework**:
1. **Self-supervised MAE pretraining** on RF + motion signals  
2. **Supervised fine-tuning** for binary occupancy classification (Empty vs Occupied)

---

## Dataset

The dataset used in this project is publicly available and can be downloaded from Figshare:

🔗 **Dataset Download Link:**  
https://figshare.com/ndownloader/articles/24939765/versions/1

### Dataset Description
- **Signal Type:** WiFi Channel State Information (CSI)
- **Environment:** Bedroom (3.8m × 2.4m) and Living Room (3.2m × 4.4m)
- **Hardware:**  
  - Intel 5300 NIC  
  - 3 external antennas  
  - 802.11n CSI tool
- **Configuration:**  
  - 3×1 antenna setup  
  - 30 subcarriers  
  - 20 MHz bandwidth  
  - 5 GHz band  
  - 1000 Hz sampling rate
- **Participants:**  
  - 15 volunteers (9 female, 6 male, ages 23–26)
- **Activities:**  
  - Lying down, reading, writing, walking, cleaning, arm training, squats, running, jumping
- **Labels:**  
  - Occupancy (Empty / Occupied)  
  - Activity category (optional)

After downloading, extract the dataset and follow the preprocessing steps in the notebook.

---

## Data Preprocessing

All preprocessing is implemented inside the provided **Jupyter Notebook (.ipynb)** and includes:

1. **Sliding Window Segmentation**
   - Window length: 500 time steps
   - Each window treated as one sample

2. **Motion Feature Extraction**
   From CSI amplitude:
   - Variance of amplitude
   - Short-Time Energy (STE)
   - Temporal gradient of mean amplitude
   - Temporal gradient of STE

3. **Multimodal Fusion**
   - RF-only: `[500 × 90]`
   - Motion-only: `[500 × 4]`
   - RF + Motion (proposed): `[500 × 94]`

4. **Train / Validation Split**
   - 80% training
   - 20% validation

---

##  Model Architecture

### 1 Masked Autoencoder (MAE)
- **Patch size:** 10 time steps
- **Encoder:** Transformer Encoder
  - 4 layers
  - 4 attention heads
  - Embedding dimension: 128
- **Decoder:** Transformer Decoder
  - 2 layers
- **Mask ratio:** 40%

The MAE learns temporal structure by reconstructing **masked signal patches**, without using labels.

---

### 2 Fine-Tuning Classifier
- Encoder output → temporal pooling
- Fully connected classification head
- Dropout for regularization
- Binary output: Empty vs Occupied

---

##  Baselines

The following baselines are implemented and compared:

- **RF-only + Dropout**
- **Motion-only + Dropout**
- **RF + Motion + Dropout**
- **RF + Motion + MAE (Proposed)**

---

##  Results

### Key Observations
- MAE pretraining consistently improves:
  - accuracy,
  - robustness at low coverage,
  - resistance to noisy CSI frames.
- Motion-only performs well for dynamic activities but fails for stationary occupants.
- RF-only is sensitive to environmental noise.
- **RF + Motion + MAE** achieves the best trade-off between accuracy and coverage.

### Selective Classification
Confidence-based thresholding is used to evaluate:
- Accuracy vs Coverage trade-offs
- Reliability under uncertainty

 **Important Note:**  
All reported plots and numerical values are based on the **latest completed evaluation runs**.  
Due to stochastic training components (random initialization, masking, dropout), results may slightly vary across executions.

---

##  Notebook Implementation

All components are implemented in a **single Jupyter Notebook (.ipynb)**, which includes:
- Dataset loading and preprocessing
- Motion feature computation
- Baseline model training
- MAE pretraining
- Fine-tuning and evaluation
- Visualization and result plots

The notebook is **fully compatible with Google Colab**.

---

## How to Run (Google Colab)

1. Upload the `.ipynb` file to Google Colab
2. Download and extract the dataset
3. Update dataset paths in the notebook
4. Run cells sequentially

No additional configuration is required.

---

##  References

- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022  
- Tong et al., *Masked Autoencoders as Spatiotemporal Learners*, NeurIPS 2022  
- Social-MAE: *Social Masked Autoencoder for Multi-Person Motion Representation Learning*

---

##  Acknowledgments

This project was conducted as part of a graduate-level course/research project.  
Special thanks to the instructor and prior work that inspired the use of MAE for robust representation learning.

---

##  Contact

**Author:** Sahar Rezagholi  
**Affiliation:** Rutgers University  
