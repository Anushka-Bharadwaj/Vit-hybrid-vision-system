🔬 ViT-Hybrid Vision System

This repository presents a **modified Vision Transformer (ViT)** architecture enriched with:

- 🧠 Custom transformer encoders
- 🌐 Multi-scale patch embeddings
- ⚙️ Relative Position Embeddings (RoPE)
- 📦 CNN stem before ViT
- 🤖 BLIP-based captioning + feature fusion
- 🐜 Hybrid PSO-ACO-based feature selection
- 🎯 Final SVM classification using hybrid ViT + BLIP features

## 📌 Key Components

- **Vision Transformer** (multi-head attention, conv-based MLPs)
- **BLIP Integration** for image caption generation + feature extraction
- **Hybrid PSO-ACO Algorithm** for robust feature selection
- **SVM** classifier with selected hybrid features
- Rich metrics (Top-1/3, precision, recall, AUC), loss curves, confusion matrices

##🗃️ Directory Structure

vit-hybrid-vision-system/
├── Vit\_Modified.py
├── Hybrid\_PSO\_ACO\_Feature\_Visualization.py
├── Images/ or SoyMCData/
├── results/
├── README.md
├── requirements.txt
└── .gitignore

## 🛠️ Setup
pip install torch torchvision transformers matplotlib seaborn scikit-learn ptflops tqdm

## 🚀 Run Training + Evaluation
python Vit_Modified.py

## 🎯 Results
* Captions generated with BLIP
* ViT feature vectors extracted and fused
* Feature selection with PSO + ACO
* Final prediction with SVM
* Outputs: metrics JSON, plots, classification reports, confusion 
