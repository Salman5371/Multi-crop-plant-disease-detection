# Multi-Crop Plant Disease Detection

Deep learning-based precision multi-crop plant disease classification for smart farming in Bangladesh.

## Research Focus
This project develops and evaluates deep learning models for multi-crop plant disease classification using a large Bangladesh-focused dataset. The study emphasizes:
- multi-crop disease classification
- explainability using Grad-CAM
- lightweight deployment feasibility

## Dataset
Primary dataset:
- Saon110/bd-crop-vegetable-plant-disease-dataset

Source:
- Hugging Face dataset page

Notes:
- The dataset is gated, so access approval may be required.
- Do not upload the dataset to GitHub.
- Respect the dataset license and usage terms.

## Models
The main models planned for comparison are:
- MobileNetV2
- ResNet50
- EfficientNetV2-S

## Planned Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class analysis
- Inference time
- Model size

## Project Structure
- `data/` contains raw and processed dataset files
- `notebooks/` contains exploratory and pilot notebooks
- `src/` contains reusable Python scripts
- `configs/` contains model configuration files
- `outputs/` stores logs, plots, Grad-CAM results, and tables
- `models/` stores trained model weights locally
- `paper/` stores manuscript draft sections

## Workflow
1. Prepare environment and dataset access
2. Run pilot training with MobileNetV2
3. Train baseline models
4. Evaluate results
5. Generate Grad-CAM visualizations
6. Benchmark lightweight deployment
7. Write paper sections in parallel

## Important Rules
- Do not upload raw dataset files to GitHub
- Do not upload large trained model weights to GitHub
- Commit code, configs, and writing regularly
- Keep experiments reproducible

## First Milestone
Initial project structure pushed to GitHub successfully.

## Next Milestones
- Environment setup
- Pilot notebook creation
- MobileNetV2 pilot training
- Baseline comparison
- Paper drafting