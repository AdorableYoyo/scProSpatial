
## 👩🏻‍🔬 What is scProSpatial?

- **A deep learning framework unifying single-cell transcriptomics with spatial proteomics**
- **Models complex biology from RNA to protein, cell to tissue**

## 🧠 What can scProSpatial do?

- **Accurately predicts comprehensive surface proteomes (~2500 proteins) from RNA**, surpassing typical spatial technology limits (e.g., CODEX ~40 proteins)
- **Integrates unpaired data**: Aligns scRNA-seq with spatial proteomics via contrastive learning (no shared features needed)
- **Robustly generalizes** to new datasets and conditions (Out-of-Distribution)

## 🌟 What is the broader impact?

- **Provides deeper insights into metastatic breast cancer** by characterizing tumor microenvironments
- **Enables cross-scale analysis** to identify molecular drivers

## Overview
![Model Overview](/fig/spatialpro_fig.001.jpeg)
## 🧐 2-Stage Training

### Stage 1: Self-Supervised Pre-training 
- Learns RNA & Protein features from diverse omics data (CITE-seq, scRNA-seq, CODEX)
- Uses masked token prediction & PPI network guidance
- Handles batch effects for seamless data integration across platforms

### Stage 2: Contrastive Alignment (Demo: see `train_clip.py`)
- Fine-tunes encoders with paired CITE-seq data
- Aligns RNA & Protein representations into a shared embedding space
- Enables cross-modal prediction (RNA → Protein)

## 💯 Evaluation

### Benchmarks (Paired Data)
RNA-to-Protein translation on CITE-seq datasets with multiple evaluation settings:
- Random split
- Few-shot learning
- Out-of-distribution (OOD) generalization

**Demo**: See `train_translation.py` for benchmark evaluation

### Zero-Shot Case Studies (Unpaired Data)
- Predicts spatial protein expression from scRNA-seq data
- Maps predictions to CODEX imaging data via learned embeddings
- No paired training data required

**Reproducibility**: See Jupyter notebooks for Figure 4 and Figure 5 demonstrating zero-shot capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scProSpatial.git
cd scProSpatial

# Install dependencies
pip install torch scanpy numpy pandas sklearn einops
pip install wandb  # Optional, for experiment tracking
```

## Demo Usage

We provide demo scripts with sample data to help you get started quickly. All demo scripts are configured to run with minimal setup.

### Data 
Can be downloaded via 
[Sample Data](https://cuny547-my.sharepoint.com/:f:/g/personal/ywu1_gradcenter_cuny_edu/Esuo2eOXRJpBsPKxCB3cu9MB0rb_XHXbBhc-bNA2qRHNfA?e=2Nei6p)

### 1. Stage 2 Training: Contrastive Alignment (Demo)
```bash
python train_clip.py
```
This demonstrates the contrastive learning stage using demo data. Default settings:
- Uses demo dataset at `/raid/home/yoyowu/scProSpatial/data/demo_clip_test_rna.h5ad`
- Runs for 1 epoch in test-only mode
- No training is performed by default (test_only=True)

### 2. Evaluation: RNA-to-Protein Translation (Demo)
```bash
python train_translation.py
```
This demonstrates the benchmark evaluation on paired data. Default settings:
- Uses demo dataset automatically when test_only=True
- Runs evaluation on pretrained model
- Shows performance metrics (CCC, Pearson, Spearman)

### 3. Run Inference (Demo)
```bash
python inference.py
```
This runs protein prediction on RNA data. Default settings:
- Uses demo breast tissue data
- Outputs predictions to test file
- Handles datasets larger than model capacity by chunking

### 4. Zero-Shot Evaluation
For zero-shot case studies on unpaired data, see the Jupyter notebooks:
- `reproducibility/Figure4.ipynb`: Spatial protein prediction from scRNA-seq
- `reproducibility/Figure5.ipynb`: Mapping to CODEX via learned embeddings

## Using Your Own Data

### Data Preparation

#### 1. Gene ID Mapping
Your data must include a `MYID` variable in the AnnData object that maps to our standardized gene IDs:

- **For RNA data**: Map using `feb27_2025_gene_ncbi_mapping.csv`
- **For protein data**: Map using `may14_2484_prot_mapping.csv`

Example of adding MYID to your AnnData:
```python
import pandas as pd
import scanpy as sc

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Load mapping file
mapping = pd.read_csv('feb27_2025_gene_ncbi_mapping.csv')

# Add MYID to your data
# Assuming your genes are in adata.var_names
adata.var['MYID'] = adata.var_names.map(mapping.set_index('NCBI')['MYID'])
```

#### 2. Training on Your Data

Modify the training scripts to point to your data:

```python
# In train_translation.py, modify the paths:
args.RNA_path = '/path/to/your/rna_data.h5ad'
args.Pro_path = '/path/to/your/protein_data.h5ad'

# Set test_only=False to enable training
args.test_only = False

# Adjust other parameters as needed
args.epochs = 100  # Set desired number of epochs
args.batch_size = 16
```

#### 3. Inference on New Data

For inference, create an artificial protein AnnData object with the same cell dimensions as your RNA data:

```python
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

# Load protein mapping
mapping = pd.read_csv('may14_2484_prot_mapping.csv')

# Load your RNA data
brst_rna = sc.read_h5ad("your_rna_data.h5ad")
k = brst_rna.shape[0]  # Number of cells

# Generate random data matrix with shape (cells × proteins)
X_random = np.random.rand(k, mapping.shape[0])
print(X_random.shape)  # Should be (k, number_of_proteins)

# Create artificial protein AnnData object
scP_artificial = ad.AnnData(
    X=X_random,
    var=pd.DataFrame(
        {
            'MYID': mapping['MYID'].values
        },
        index=mapping['Protein']  # Protein names as var_names
    )
)

# Match cell indices
scP_artificial.obs.index = brst_rna.obs.index

# Save for inference
scP_artificial.write('artificial_protein_adata.h5ad')
```

Then run inference:
```python
python inference.py \
    --RNA_path your_rna_data.h5ad \
    --Pro_path artificial_protein_adata.h5ad \
    --save_dir your_predictions.h5ad
```

## Model Architecture

scProSpatial uses a transformer-based architecture with:
- **RNA Encoder**: Performer-based encoder for RNA expression
- **Protein Encoder**: Performer-based encoder for protein expression  
- **Joint Projection**: Projects both modalities to shared embedding space
- **Translator**: MLP-based module for RNA-to-protein translation

### Pre-training Details

**Stage 1: Self-Supervised Pre-training**
- Uses masked language modeling (MLM) with PPI-guided network embeddings
- Trains on diverse omics datasets (CITE-seq, scRNA-seq, CODEX)
- Incorporates dataset-specific embeddings to handle batch effects
- Outputs: `pretrained_rna_encoder` and `pretrained_protein_encoder`
- **Note**: Pre-training code available upon request

**Stage 2: Contrastive Fine-tuning**
- Uses CLIP-like contrastive loss to align RNA and protein spaces
- Fine-tunes on paired CITE-seq data
- Creates unified embedding space for cross-modal prediction

## Key Parameters

- `enc_max_seq_len`: Maximum number of genes for RNA (default: 20000)
- `dec_max_seq_len`: Maximum number of proteins (default: 320)
- `dim`: Embedding dimension (default: 128)
- `batch_size`: Training batch size (default: 16)
- `lr`: Learning rate (default: 2e-4)

## Output Format

Predictions are saved as an AnnData object with:
- `adata.obsm['protein_predicted']`: Predicted protein expression matrix
- `adata.uns['protein_predicted_names']`: Protein names
- `adata.uns['protein_predicted_myid']`: Protein MYIDs

## Notes

- **Pretrained Models**: The model uses MLM pretraining with PPI-guided network embeddings. Pretrained weights are provided, and full pretraining code is available upon request.
- **Memory Requirements**: For large datasets, the inference script automatically chunks proteins to fit within model capacity.
- **Multiple Datasets**: The model supports training on multiple datasets simultaneously using dataset-specific embeddings.
- **Reproducibility**: See the Jupyter notebooks in the repository for complete reproducibility of all figures in the paper.

## Citation

If you use scProSpatial in your research, please cite:
```
[Citation will be added after publication]
```


## Contact

For questions or issues, please open an issue on GitHub or contact [ywu1@gradcenter.cuny.edu].