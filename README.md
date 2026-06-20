# Deep Learning-Based Decision Support Framework for Multi-Crop Plant Disease Detection and Management in Smart Farming


**Project Type:** Thesis / Research Project
**Research Area:** Deep Learning, Smart Farming, Plant Disease Detection, Decision Support System

---

## Project Overview

This repository contains the implementation of a deep learning-based plant disease detection and decision-support framework for smart farming. The project focuses on multi-crop plant disease classification using transfer learning models and provides explainable prediction outputs, preliminary severity/risk indication, and rule-based disease management recommendations.

Plant diseases are a major threat to agricultural productivity, especially in crop-dependent countries such as Bangladesh. Manual disease diagnosis is often time-consuming, expert-dependent, and difficult to scale for smallholder farmers. This project develops an image-based smart farming support system that can classify crop diseases from leaf images and generate useful decision-support outputs for farmers, researchers, and agricultural stakeholders.

The implemented system compares three transfer learning models:

* MobileNetV2
* ResNet50
* EfficientNetV2-S

The best-performing model is integrated into an inference pipeline and a Streamlit-based web application for disease prediction, confidence score display, top-5 prediction analysis, preliminary risk indication, and rule-based management recommendations.

---

## Research Objectives

The main objectives of this project are:

1. To prepare and preprocess a Bangladesh-focused multi-crop plant disease image dataset.
2. To implement and compare transfer learning models for plant disease classification.
3. To evaluate model performance using accuracy, precision, recall, F1-score, confusion matrix, and class-wise analysis.
4. To identify the best-performing model for disease detection.
5. To apply Grad-CAM explainability to visualize important image regions used by the model.
6. To develop a rule-based decision-support module for preliminary severity/risk indication and disease management recommendations.

---

## Dataset

The study uses a Bangladesh-focused multi-crop plant disease image dataset from Hugging Face:

```text
Saon110/bd-crop-vegetable-plant-disease-dataset
```

Dataset summary:

| Split      | Number of Images |
| ---------- | ---------------: |
| Training   |           86,467 |
| Validation |           24,698 |
| Test       |           12,423 |
| Total      |          123,588 |

The dataset contains 94 crop disease and healthy classes.

---

## Models Used

Three transfer learning models were implemented and compared:

| Model            | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| MobileNetV2      | Lightweight baseline model suitable for faster inference |
| ResNet50         | Strong convolutional neural network baseline             |
| EfficientNetV2-S | Final selected model due to highest observed performance |

---

## Model Performance

The main model comparison results are shown below:

| Model            | Test Accuracy | Precision | Recall | F1-score |
| ---------------- | ------------: | --------: | -----: | -------: |
| MobileNetV2      |        0.9453 |    0.9466 | 0.9453 |   0.9444 |
| ResNet50         |        0.9555 |    0.9562 | 0.9555 |   0.9550 |
| EfficientNetV2-S |        0.9575 |    0.9606 | 0.9575 |   0.9571 |

EfficientNetV2-S achieved the highest observed accuracy and weighted F1-score and was selected as the final model for the decision-support system.

---

## Model Efficiency Analysis

| Model            | Parameters | Model Size |  Inference Time |    FPS |
| ---------------- | ---------: | ---------: | --------------: | -----: |
| MobileNetV2      |     2.344M |    9.18 MB |  5.634 ms/image | 177.49 |
| ResNet50         |    23.701M |   90.72 MB |  6.521 ms/image | 153.35 |
| EfficientNetV2-S |    20.298M |   78.30 MB | 22.106 ms/image |  45.24 |

MobileNetV2 was the fastest and lightest model, while EfficientNetV2-S provided the best classification performance.

---

## Statistical Significance Testing

McNemar’s test was used to compare paired model predictions on the same test set.

| Comparison                      |   p-value | Significant at 0.05 |
| ------------------------------- | --------: | ------------------- |
| EfficientNetV2-S vs ResNet50    |    0.3235 | No                  |
| EfficientNetV2-S vs MobileNetV2 | 1.866e-09 | Yes                 |
| ResNet50 vs MobileNetV2         | 4.784e-07 | Yes                 |

EfficientNetV2-S achieved the highest observed performance, although its improvement over ResNet50 was not statistically significant.

---

## Key Features

The project includes:

* Multi-crop plant disease classification
* Transfer learning-based model comparison
* EfficientNetV2-S final inference pipeline
* Top-5 prediction output
* Confidence-based preliminary severity/risk indication
* Rule-based disease management recommendation
* Farmer alert message generation
* Grad-CAM explainability
* Streamlit web application
* Dataset distribution analysis
* Error analysis and confused class-pair analysis
* Model efficiency analysis
* Dataset quality and duplicate-image checking
* Statistical significance testing using McNemar’s test

---

## Project Structure

```text
Multi-crop-plant-disease-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── run_local.md
│
├── configs/
│   ├── efficientnetv2s.yaml
│   ├── mobilenetv2.yaml
│   └── resnet50.yaml
│
├── Data/
│   ├── class_names.json
│   ├── label2id.json
│   ├── id2label.json
│   └── sample/
│
├── models/
│   ├── efficientnetv2s_best.pth
│   ├── mobilenetv2_best.pth
│   └── resnet50_best.pth
│
├── Notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_mobilenetv2_pilot.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_gradcam_efficientnetv2s.ipynb
│   └── 05_clean_pipeline_demo.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   ├── gradcam/
│   └── prediction_reports/
│
└── src/
    ├── data_loader.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    ├── transforms.py
    ├── utils.py
    ├── decision_support.py
    ├── inference_demo.py
    ├── batch_inference_demo.py
    ├── dataset_distribution_analysis.py
    ├── model_efficiency_analysis.py
    ├── error_analysis.py
    ├── top5_prediction_analysis.py
    ├── data_leakage_check.py
    ├── generate_all_model_predictions.py
    └── statistical_significance_test.py
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Run the web application using:

```bash
streamlit run app.py
```

The application allows users to upload a plant leaf image and receive:

* Predicted crop and disease class
* Confidence score
* Top-5 predictions
* Preliminary severity/risk indication
* Management recommendations
* Alert message
* Downloadable prediction report

---

## Running Inference from Command Line

Single image prediction:

```bash
python -m src.inference_demo --image Data/sample/test_leaf.jpg
```

Batch prediction:

```bash
python -m src.batch_inference_demo
```

---

## Running Analysis Scripts

Dataset distribution analysis:

```bash
python -m src.dataset_distribution_analysis
```

Model efficiency analysis:

```bash
python -m src.model_efficiency_analysis
```

Error analysis:

```bash
python -m src.error_analysis
```

Top-5 prediction analysis:

```bash
python -m src.top5_prediction_analysis
```

Dataset duplicate-quality checking:

```bash
python -m src.data_leakage_check
```

Statistical significance testing:

```bash
python -m src.statistical_significance_test
```

---

## Explainability

Grad-CAM was used to visualize the image regions that influenced model predictions. This helps interpret whether the model focuses on disease-relevant leaf regions or irrelevant background artifacts.

---

## Decision-Support Module

The decision-support module provides:

* Preliminary risk indication based on predicted class and confidence score
* Rule-based crop disease management suggestions
* Alert message generation for farmer-facing output
* Downloadable prediction report

The severity/risk indication is confidence-based and should not be interpreted as true biological disease severity estimation.

---

## Limitations

This project has some limitations:

1. The system is image-based and does not currently use soil moisture, temperature, humidity, or other IoT sensor data.
2. The severity/risk indication is preliminary and confidence-based, not based on expert-labelled disease severity scores.
3. The system has not yet been validated through real-time field deployment.
4. Some noisy, watermarked, or out-of-distribution images may reduce prediction reliability.
5. A dataset quality check was conducted using duplicate-image analysis. Some possible cross-split duplicate groups were observed in the original dataset split; therefore, future work should consider a cleaner split reconstruction and re-evaluation for publication-level validation.

---

## Future Work

Future improvements may include:

* Cleaner dataset reconstruction and model re-evaluation
* Real field validation with farmer-captured images
* Integration of soil moisture, temperature, humidity, and environmental sensor data
* True disease severity estimation using expert-labelled severity datasets
* Mobile application deployment
* GPS-based disease mapping
* Multilingual farmer advisory system
* Lightweight edge deployment for smart farming devices

---

## Thesis Scope Statement

This repository supports the thesis titled:

**A Deep Learning-Based Decision Support Framework for Multi-Crop Plant Disease Detection and Management in Smart Farming**

The implemented work focuses on image-based disease classification, explainability, preliminary risk indication, and rule-based management recommendation. IoT sensor integration and real-time field deployment are considered future extensions.

---

## Author

**Md Salman Farshi**
