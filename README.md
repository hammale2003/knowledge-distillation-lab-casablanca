# Document Understanding & Continual Learning Evaluation Platform

**Research Internship Project — Laboratory of Complex Systems, Ecole Centrale Casablanca**

---

## 📚 Project Context

This repository was developed as part of a research internship at the Laboratory of Complex Systems, Ecole Centrale Casablanca. The project focuses on the evaluation, comparison, and continual learning of document understanding models, with a special emphasis on knowledge distillation, catastrophic forgetting, and advanced benchmarking using modern deep learning architectures.

## 🚀 Features

- **Gradio Web Interfaces** for interactive evaluation and visualization:
  - `gradio_document_evaluation.py`: Comprehensive platform for loading, evaluating, and comparing state-of-the-art document classification models (DiT, ViT, LayoutLMv3, Donut, etc.) on the RVL-CDIP dataset and its variants. Includes batch evaluation, per-class analysis, and visual comparison tools.
  - `evalution_of_the_student.py`: Advanced interface for comparing student (distilled) and teacher models, simulating and measuring continual learning scenarios, and analyzing catastrophic forgetting (including EWC regularization). Provides detailed plots, metrics, and recommendations for model improvement.
- **Knowledge Distillation Pipeline** (`distillation.py`): Custom training loop for distilling large teacher models into compact student models, with checkpointing, learning curve tracking, and dynamic loss weighting.
- **Dataset Support**: Integrates with HuggingFace Datasets, especially `sitloboi2012/rvl_cdip_large_dataset` and `HAMMALE/rvl_cdip_OCR`.
- **Visualization**: Generates detailed plots for performance, efficiency, per-class accuracy, and forgetting analysis.
- **Extensive Documentation**: Each interface includes in-app documentation and guidance for users.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. **Install dependencies:**
   All required packages are listed in `requirements.txt`. Install them using pip:
   ```bash
   pip install -r requirements.txt
   ```

   **Main dependencies:**
   - torch
   - torchvision
   - transformers
   - datasets
   - numpy
   - Pillow
   - matplotlib
   - seaborn
   - pandas
   - gradio
   - sentencepiece
   - tqdm
   - accelerate
   - safetensors
   - huggingface-hub
   - tokenizers

3. **(Optional) GPU Support:**
   For best performance, use a machine with CUDA-enabled GPU and the appropriate PyTorch version.

## 💻 Usage

### 1. Document Model Evaluation Platform

Launch the Gradio interface for model evaluation and comparison:
```bash
python gradio_document_evaluation.py
```
- Load pre-trained models (DiT, LayoutLMv3, Donut, etc.)
- Predict on single images or evaluate on full datasets
- Batch evaluation and visual comparison of multiple models
- Per-class accuracy, inference time, and comprehensive reports

### 2. Student vs Teacher & Continual Learning Analysis

Launch the advanced interface for student/teacher comparison and continual learning:
```bash
python evalution_of_the_student.py
```
- Compare distilled student models with large teacher models
- Simulate and analyze continual learning (Naive, Rehearsal, LwF)
- Measure and visualize catastrophic forgetting (including EWC regularization)
- Generate detailed plots and recommendations

### 3. Knowledge Distillation Training

To train a student model via knowledge distillation:
```bash
python distillation.py
```
- Edit the script to set the desired checkpoint or start from scratch
- Training progress, checkpoints, and learning curves will be saved automatically

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Main packages include:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
gradio>=4.0.0
sentencepiece>=0.2.0
tqdm>=4.65.0
accelerate>=0.20.0
safetensors>=0.3.0
huggingface-hub>=0.15.0
tokenizers>=0.13.0
```

## 🏫 Credits & Acknowledgements

- **Laboratory of Complex Systems, Ecole Centrale Casablanca** — Research supervision and scientific guidance
- **HuggingFace** — Transformers, Datasets, and model zoo
- **Gradio** — For rapid prototyping of interactive ML interfaces
- **RVL-CDIP Dataset** — Standard benchmark for document classification

## 📄 License

This project is for academic and research purposes. Please cite appropriately if used in publications.

---

For questions or collaboration, please contact the Laboratory of Complex Systems, Ecole Centrale Casablanca. 
