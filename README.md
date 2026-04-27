# EnsembleDeepSets: Immunological Deep Ensemble with Attention for Age prediction framework

**EnsembleDeepSets** is a robust machine learning framework designed to predict donor age from single-cell RNA sequencing (scRNA-seq) data. It integrates the interpretability of **DeepSets** architectures with the robust performance of **AutoGluon** tabular ensembles.

## 🌟 Key Features

*   **Multi-Architecture Support**: Includes DeepSets, Attention-based Pooling, and Sparse Transformers for single-cell data.
*   **Intelligent Caching**: Built-in caching system for data splits, embeddings, and model weights to accelerate iterative development.
*   **Interpretability**: Tools to calculate **Cell Contributions** and **Gene Contributions**.
*   **Comprehensive Benchmarking**: Automated calculation of Pearson/Spearman correlations, MAE, RMSE, compared with traditional ML methods.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Lzcstan/EnsembleDeepSets.git
cd EnsembleDeepSets
```

### 2. Environment Setup
The project relies on **PyTorch (CUDA 11.8)** and **AutoGluon**. Since the provided `requirements.txt` contains absolute conda paths, we recommend creating a fresh environment and installing core dependencies first:

```bash
# Create environment (Python 3.9+ recommended)
conda create -n deepsets python=3.10
conda activate deepsets

# 1. Install PyTorch (Specific to CUDA 11.8 as per your environment)
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. Install AutoGluon and Scientific Computing Stack
pip install autogluon==1.4.0 scanpy pandas numpy scikit-learn matplotlib seaborn tqdm openpyxl rich
```

*Note: A raw `requirements.txt` is provided in the repo if you wish to attempt an exact replication, but cleaning it for your specific OS path is recommended.*

## 📂 Data Preparation

The project expects a specific directory structure. By default, scripts look for data in `/personal/ImmAge` (configurable in `Config` classes).

**Recommended Structure:**
```text
./data/                      # Change DATA_DIR in scripts to point here
├── donor_metadata.xlsx      # Columns: donor_id, age, dataset, is_train
├── marker_genes.csv         # (Optional) List of genes to filter
└── donor_files/             # Directory containing per-donor .h5ad files
    ├── donor_ID1.h5ad
    ├── donor_ID2.h5ad
    └── ...
```

## 📦 Pretrained Checkpoints

We provide several pretrained checkpoints for reproducibility and downstream inference.  
These checkpoints can be used with `predict_with_deepsets.py` or loaded manually in PyTorch for evaluation and interpretation.

> Please make sure that the input gene set, preprocessing strategy, and model configuration are consistent with the checkpoint you use.
| Model | Training Set | Epochs | Download Link |
|:---|:---|:---|:---|
| main model [Ensemble-DeepSets] | AIDA, OneK1K | 60 | [Download](https://disk.pku.edu.cn/link/AAC39485B4DF9043FC8F270C2E2CFD16D4) |
| SC2018 | AIDA, OneK1K, SC2018 | 60 | [Download](https://disk.pku.edu.cn/link/AA4954B60BCCDE4457ACB22A61E9E21F22) |
| siAge | AIDA, OneK1K, siAge | 50 | [Download](https://disk.pku.edu.cn/link/AA7A491ABEA5374BAF83670D61857678B4) |
| siAge (extreme) [Ensemble-DeepSets (tuning)] | AIDA, OneK1K, siAge (age < 20 or age > 80) | 60 | [Download](https://disk.pku.edu.cn/link/AA81F2EF1231D34366B88CA5DBCB107DBF) |
| SC2018 + siAge | AIDA, OneK1K, SC2018, siAge | 50 | [Download](https://disk.pku.edu.cn/link/AAC340B295DEAB41B4A42A2DF8B46E03FD) |
| SC2018 + siAge (extreme) | AIDA, OneK1K, SC2018, siAge (age < 20 or age > 80) | 60 | [Download](https://disk.pku.edu.cn/link/AAD70EAA9859BC4EC3BFD283C481F40F59) |

## 🚀 Usage

### Mode A: Ensemble Training (AutoGluon + DeepSets)

```bash
python ag_integrat_gene_mae_sweep.py
```
*   **What it does**: 
    1.  Loads data with caching.
    2.  Trains standard AutoGluon tabular models (NNTorch, FASTAI, etc.) on pseudobulk.
    3.  Trains a custom `DeepSetsTabularModelAttn` on single-cell data.
    4.  Ensembles the results.
*   **Output**: Saved in `./results_all_gene_.../` including scatter plots and metrics.

### Mode B: Standalone Deep Learning Training
You can train the neural networks (DeepSets or Transformers) directly without AutoGluon using the utility script.

```bash
python dist_train_cache.py --model_type deepsets --epochs 100 --batch_size 8
```
*   **Options**: `--model_type` can be `deepsets`, `transformer`, or `sparse_transformer`.
*   **Useful flags**: `--skip_training` (only visualize data), `--evaluate_only` (load existing weights).

### Mode C: Inference & Interpretation
To run predictions on new data or analyze which cells drive aging signatures:

```bash
python predict_with_deepsets.py
```
*   **Configuration**: Edit the `PredictConfig` class in the file to point to your model path.
*   **Features**:
    *   Generates `predictions_detailed.csv`.
    *   Computes cell-level contribution scores (Attention/Gradient).

## 📄 Project Structure

| File | Description |
|:---|:---|
| **`dist_train_cache.py`** | **Core Engine**. Contains PyTorch model definitions (`DeepSetsAgePredictor`, `UnifiedSparseTransformer`), `SingleCellDataset`, training loops, caching logic, and extensive plotting functions. |
| **`ag_extra_models.py`** | **Bridge**. Adapts the PyTorch models from `dist_train_cache.py` to be compatible with AutoGluon's `AbstractModel` API. Implements the `DeepSetsTabularModelAttn`. |
| **`pseudo_redo.py`** | **Metrics**. Utility class `BenchmarkItem` for calculating Pearson/Spearman correlations, MAE, and generating regression plots. |
| **`ag_integrat_gene_mae_sweep.py`** | **Main Script**. The primary entry point for training the full ensemble. Handles data loading strategies and result visualization. |
| **`predict_with_deepsets.py`** | **Inference**. Standalone script for loading trained models and running prediction/interpretation on test sets without retraining. |
| **`deepsets_contribu_multi_tests.py`** | **Analysis**. Specialized script for validating model weights and debugging contribution scores (Integrated Gradients). |

## 🧠 Interpretability

EnsembleDeepSets provides two kinds of interpretability:

1.  **Cell-Level**: Using attention weights or gradient-based methods.
2.  **Gene-Level**: Using gradient-based methods.

## ⚙️ Configuration

Key parameters are located at the top of `ag_integrat_gene_mae_sweep.py` and `ag_extra_models.py`:

```python
MAX_CELLS_PER_DONOR = 1000  # Downsample large donors for memory efficiency
CACHE_DIR = "./donor_cache_ag_all_gene" # Location for cached pickle files
USE_RAW = False             # Whether to use raw counts or log1p data
EPOCH = 60                 # Training epochs for the DeepSets component
```

## 🤝 Contributing

Contributions are welcome! Please ensure you update `requirements.txt` if adding new dependencies and run `dist_train_cache.py --skip_training` to verify data loading integrity before submitting PRs.

## 📜 License

MIT License
