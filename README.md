# LeafLensAI — Plant Disease Classification

LeafLensAI is a research prototype for classifying plant leaf images using a trained ResNet18 deep-learning model. It includes a Streamlit web interface for image uploads, confidence scores, and top predictions.

**Important:** This model is intended for experimentation and research, not as a substitute for professional agricultural diagnosis. Performance drops significantly on field-style images.

## Features

- ResNet18 image classifier trained on PlantVillage.
- 39 output categories, including healthy and diseased plant categories.
- Streamlit web interface for uploading a leaf image and viewing predictions.
- Top-3 predictions with confidence scores.
- Evaluation on PlantVillage and PlantDoc to investigate domain generalization.
- CPU inference support; compatible NVIDIA GPUs can be used when a suitable PyTorch CUDA build is installed.

## Project structure

```text
LeafLensAI/
├── app.py
├── requirements-inference.txt
├── requirements.txt
├── src/
│   ├── models.py
│   └── data.py
├── results/
│   └── resnet18_baseline/
│       └── best.pt
├── docs/
└── README.md
```

The trained checkpoint is distributed separately as a GitHub Release asset and is not included in ordinary Git commits.

## Installation

### Requirements

- Python 3.10 or 3.11 recommended; verify compatibility with the selected PyTorch build.
- Git
- Internet access to download the model checkpoint
- Approximately 1 GB or more of free disk space recommended for the environment and model cache

### 1. Clone the repository

```bash
git clone https://github.com/riyasachan94514/LeafLensAI.git
cd LeafLensAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt and run:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install PyTorch and torchvision

For a CPU-only laptop:

```powershell
python -m pip install torch torchvision
```

For an NVIDIA GPU, install the PyTorch build appropriate for the laptop's GPU and supported CUDA runtime by following the official selector:

https://pytorch.org/get-started/locally/

Then install the inference dependencies:

```powershell
python -m pip install -r requirements-inference.txt
```

## Download the trained model

Download [`best.pt` (42.78 MB)](https://github.com/dhruvmahajan001/LeafLens-AI/releases/download/v1.0.0/best.pt) from the v1.0.0 release.
Create the model directory if it does not exist:

```powershell
New-Item -ItemType Directory -Force "results\resnet18_baseline"
```

Place the downloaded checkpoint at:

```text
results/resnet18_baseline/best.pt
```

The app expects this exact path. Do not rename the checkpoint.

## Run the application

From the repository root, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, normally:

http://localhost:8501

Upload a leaf image to see the predicted category and top alternatives.

## Model and evaluation

The model was trained on the PlantVillage dataset with 39 output classes. Cross-dataset evaluation was conducted on shared categories from PlantVillage and PlantDoc.

| Evaluation | Images | Accuracy |
|---|---:|---:|
| PlantVillage test, 28 shared classes | 5,781 | 99.53% |
| PlantDoc open evaluation, full 39-class output | 2,569 | 13.78% |
| PlantDoc restricted evaluation, 28 shared classes | 2,569 | 22.89% |

The PlantVillage result measures performance on held-out images from a distribution similar to training. PlantDoc contains more varied, field-style photographs.

The large accuracy gap indicates that benchmark performance does not guarantee reliable predictions on real-world images. The model can be confidently wrong on unfamiliar images. Confidence scores should not be interpreted as calibrated probabilities of disease.

These figures describe the current recorded experiment; they are not a guarantee of real-world performance. Informal tests on internet images are qualitative observations unless every image has a verified ground-truth label and is evaluated using a documented protocol.

## Limitations

- The model predicts among its trained categories; it does not reliably detect every possible plant disease.
- It may misclassify unfamiliar crops, backgrounds, lighting, camera angles, or disease stages.
- A high confidence score does not guarantee correctness.
- PlantDoc performance is substantially lower than PlantVillage test performance.
- The model has not been established as suitable for clinical, commercial, or agricultural decision-making.
- The application is a research prototype, not a validated diagnostic service.

## Dataset sources

- PlantVillage: https://huggingface.co/datasets/Project-AgML/plant_village_classification
- PlantDoc: https://huggingface.co/datasets/Project-AgML/plant_doc_classification

Please review and comply with the dataset licenses and terms before redistributing any images. This repository does not require publishing the datasets.

## Research and reproducibility

The repository includes training and evaluation scripts under `src/`. The Streamlit inference app uses the existing trained checkpoint and does not retrain the model.

For reproducible research, record the checkpoint version, software versions, evaluation split, and dataset provenance when reporting results.

## License and attribution

Add the license applicable to your original code and verify the licensing and attribution requirements of any reused code, model weights, or assets before redistributing them. Do not assume third-party code or datasets are covered by the same license.

## Disclaimer

LeafLensAI is an experimental research project. Its predictions may be incorrect and should not be used alone to make plant-health or treatment decisions.
