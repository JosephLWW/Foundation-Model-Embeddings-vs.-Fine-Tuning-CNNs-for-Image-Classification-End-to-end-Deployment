# Foundation Model Embeddings vs. Fine Tuning CNNs for Image Classification: End-to-end Deployment

## Overview

This project presents a comprehensive comparison of two distinct approaches to fashion image classification: **DINOv2 foundation model embeddings** versus a custom fine-tuned CNN (ResNet34, named **DressNet**). The best-performing model is deployed as an interactive Hugging Face application, demonstrating a complete research-to-production pipeline for computer vision tasks.

## Project Objectives

- **Compare feature extraction methods**: foundation model embeddings vs. traditional fine-tuning
- **Fine-tune a ResNet34 architecture** (DressNet) specifically for fashion classification
- **Utilize DINOv2** as a feature extractor for high-level embeddings
- **Deploy the superior model** as an interactive Hugging Face Spaces application
- **Document a reproducible** end-to-end pipeline from experimentation to deployment

## Features

- **DINOv2 Foundation Model**: Leverages pre-trained ViT features from Meta's DINOv2 for zero-shot feature extraction
- **Custom DressNet Model**: Fine-tuned ResNet34 architecture optimized for fashion image classification
- **Comparative Analysis**: Comprehensive evaluation of both approaches across multiple metrics
- **Interactive Deployment**: Fully functional Hugging Face application with real-time predictions
- **Complete Pipeline**: From data preparation and model training to deployment and user interaction

## Models

### DINOv2 (Foundation Model Approach)

| Aspect | Details |
|--------|---------|
| **Model** | DINOv2 ViT-base |
| **Approach** | Frozen feature extractor |
| **Methodology** | Images processed through DINOv2 to obtain high-level embeddings, followed by a simple classifier head |
| **Advantages** | No domain-specific training required, leverages powerful pre-trained representations |

### DressNet (Fine-Tuned CNN Approach)

| Aspect | Details |
|--------|---------|
| **Model** | ResNet34 architecture with custom modifications |
| **Training** | Fine-tuned on fashion-specific dataset with transfer learning |
| **Name Origin** | "DressNet" reflects the fashion domain focus |
| **Advantages** | Optimized for specific task, potentially better performance on domain-specific data |

## Dataset

The project utilizes a fashion image classification dataset featuring over 16,000 images of dresses sourced from the largest fashion shopping website in Germany. The dataset has been carefully curated to ensure balanced representation across classes and high-quality annotations. The identifiers corresponding to the training and test examples are provided in the files train_2025.csv and test_2025.csv. We’re also providing the labels indicating the category each dress belongs

### Key Dataset Characteristics:

- Multiple dress categories
- High-resolution images suitable for deep learning
- Balanced class distribution
- Professionally annotated labels

## Workflow

1. **Data Preparation**: Image preprocessing, augmentation, and train/val/test splitting
2. **Feature Extraction (DINOv2)**: Generate embeddings using the pre-trained DINOv2 model
3. **Model Training (DressNet)**: Fine-tune ResNet34 on the fashion dataset
4. **Comparative Evaluation**: Assess both models on accuracy, precision, recall, and inference speed
5. **Model Selection**: Choose the best-performing model based on evaluation metrics
6. **Application Development**: Build an interactive Gradio interface for the selected model
7. **Hugging Face Deployment**: Deploy the application on Hugging Face Spaces
8. **Documentation**: Create comprehensive documentation and usage guidelines

## Installation & Setup
To run this project, install the necessary dependencies and ensure you have a GPU-enabled environment for efficient model training. requirements.txt file is provided.

## Deployment on Hugging Face

The best-performing model (DressNet) has been deployed as an interactive web application on Hugging Face Spaces. The application allows users to:

- Upload fashion images
- Receive instant classification predictions
- View confidence scores for each category
- Batch process multiple images

See: https://huggingface.co/spaces/JosephLWW/Practical-Deep-Learning-With-Visual-Data


## Repository Structure

```
Foundation-Model-Embeddings-vs.-Fine-Tuning-CNNs-for-Image-Classification-End-to-end-Deployment/

├── Data/
│   ├── Images/                 # Images
│   ├── train_2025.csv          # Training Set Identifiers
│   └── test_2025.csv           # Test Set Identifiers
├── Training/
│   ├── Data Preprocessing and Preparation.ipynb
│   ├── Classification with Preextracted DinoV2 Features + ASHA Hyperparameter Tuning.ipynb
│   ├── Fine-tuning an ImageNet Pre-trained CNN.ipynb
│   ├── Model Evaluation and Comparison.ipynb
│   ├── load_transform.py       # Some heper functions
│   └── saved_models/           # Trained model weights
├── Deployment/
│   ├── src/                    
│   │   ├── helpers.py          # Libraries
│   ├── bundles/                # Trained model weights
│   ├── app.py                  # Run app to process an image
│   ├── To run app.txt          # Link to Hugging Face app
│   ├── requirements.txt        # Project dependencies
│   └── README.md               # Readme for Huggingface
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── LICENSE                     # License
```

## Future Improvements

- Experiment with other foundation models (CLIP, SAM)
- Implement ensemble approaches combining both methods
- Expand dataset with more diverse fashion categories
- Add multi-label classification for outfit composition
- Incorporate textual descriptions for multimodal analysis
- Optimize deployment for edge devices (ONNX, TensorRT)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{fashion_classification_2024,
  title = {Foundation Model Embeddings vs. Fine Tuning CNNs for Image Classification: End-to-end Deployment},
  author = {Wan-Wang, Joseph},
  year = {2025},
  url = {https://github.com/JosephLWW/Foundation-Model-Embeddings-vs.-Fine-Tuning-CNNs-for-Image-Classification-End-to-end-Deployment/tree/main}
}
```
## Contact

For questions or inquiries about this project, please reach out via GitHub Issues or contact the project maintainers.

---

**Last Updated**: December 2025
