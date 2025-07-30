# CascadeNet

## Installation

Set up the CascadeNet environment with the required dependencies:

```bash
conda create -n cascadenet python==3.8
conda activate cascadenet
pip install -e .
```
## Environment verification
Here, we quickly verify that the environment is set up correctly by running a mini‐test on a few lines from the caida dataset. This will generate sample traces from `test_data/original.csv` and place the outputs under `test_result/`:

```bash
bash test/cascadenet_minitest.sh
```
You should see generated trace files in `test_result/` without errors.

## Installation for evaluation

Create and configure the evaluation environment with the required dependencies:

```bash
# Create evaluation environment
conda create -n cascadenet_eval python=3.8.11
conda activate cascadenet_eval

# Install packages
pip install -e .              # Install main package
pip install -e netml/         # Install modified netml package
```

## Environment Verification

Verify that the evaluation environment is properly configured by running a comparative test between real and synthetic data.

**Note:** Ensure you are in the project root directory before executing the test.

This verification uses:
- Real data: `test_data/original.csv` (reference dataset)
- Synthetic data: `test_data/generated.csv` (generated sample data)

```bash
# Ensure you're in the project root directory
bash test/evaluation_minitest.sh
```

This verification test will:
- Validate that all required Python packages are correctly installed
- Test the evaluation framework's core functionality
- Verify that evaluation scripts can execute without dependency errors
- Generate sample outputs under `test_result/evaluation/`

**Important:** The outputs in `test_result/evaluation/` are for verification purposes only and do not represent meaningful evaluation results, as they use minimal test data. The actual evaluation results will be generated in the full evaluation pipeline described below.

## Detailed Instructions

This section provides comprehensive instructions for reproducing all experimental results presented in the paper.

### Data
Download raw data (caida, ca, ton_iot, and dc) from the [project data folder on Google Drive](https://drive.google.com/drive/folders/1OASaruR4w-Ry5Ei1baaH8TF_8PIkt7Jp?usp=sharing) and place it under the project root directory. The baseline [result](https://drive.google.com/drive/folders/15-v4GaxFEvkgNUs-G7ONR06VI0c7w6AU?usp=sharing) CSVs are also available.
Extract each dataset folder and place them under the project root in the following structure:
```
CascadeNet/
├── data/
│   ├── caida/
│   └── ...
└── result/
    ├── e-wgan-gp/
    └── ...
```

### Overview of Experiments

The experimental evaluation consists of several components:

1. **Baseline Comparisons**: CascadeNet vs. existing methods
2. **Backbone Architecture Analysis**: Different neural network architectures
3. **Ablation Studies**: Impact of individual components
4. **Parameter Sensitivity**: Effect of different configurations (time series length)

All experiments generate results that correspond to specific figures and tables in the paper.

### Running Complete Experimental Suite

**Note:** 
- On a machine with an RTX 4090 GPU and an 8-core CPU, each experiment configuration typically takes 3–5 hours. Each provided `.sh` script runs several such configurations, so total runtime may vary.
- Ensure you are in the project root directory and have activated the `cascadenet` environment before running experiments.

```bash
conda activate cascadenet
```

### Data Dependencies

**Important:** There are precomputed baseline data (see [Data](#data) section for accessing precomputed baseline results). The CascadeNet experiments will generate new results that can be compared against these baselines.

### Complete Experimental Suite
All Paper Experiments (Non-overlapping):
```bash
bash config/experiments/all_figures.sh
```
This script runs all experiments presented in the paper without duplication, providing the most efficient way to reproduce all results.
#### Approximately Runtime: 7 days

### Individual Experiment Groups
For users who prefer to run specific subsets of experiments:

**Notes:**
- The following individual scripts can contain overlapping experiments, and are provided for convenience when investigating specific aspects of the research.
- Approximately Runtime if running sequentially: 10 days
- The anomaly detection experiments (figures 10, 24, 25, 26, 27, 28) are **included** in the primary baseline comparison script.

#### Baseline Comparison Experiments

These experiments compare CascadeNet against existing baseline methods:

**Primary Baseline Comparison (Figures 2, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 23, Table 2):**
```bash
bash config/experiments/figure_2_6_7_8_9_11_12_13_14_15_17_23_table_2.sh
```

**Anomaly Detection Comparison (Figures 10, 24, 25, 26, 27, 28):**
```bash
bash config/experiments/figure_10_24_25_26_27_28.sh
```

#### Architecture and Component Analysis

**Backbone Architecture Comparison (Figure 16):**
```bash
bash config/experiments/figure_16.sh
```
Evaluates different neural network backbone choices for CascadeNet:
- MLP-MLP architecture configuration
- MLP-RNN architecture configuration  
- RNN-RNN architecture configuration

*Results saved to: `result/cascadegan_test/`*

**Optimization Techniques Ablation (Figure 18):**
```bash
bash config/experiments/figure_18.sh
```
Analyzes the impact of key optimization techniques:
- CN-w/o-ZI: CascadeNet without zero-inflation method
- CN-w/o-Cond: CascadeNet without conditional input

**Conditional Input Strategy Ablation (Figure 19):**
```bash
bash config/experiments/figure_19.sh
```
Studies the effect of different conditional input features:
- CN-w/o-NPF: Without number of packets per flow
- CN-w/o-DUR: Without flow duration
- CN-w/o-NumA: Without number of active time steps
- CN-w/o-MaxPR: Without maximum packet rate
- CN-w/o-MeanPR: Without mean packet rate
- CN-w/o-StdPR: Without standard deviation of packet rate
- CN-w/o-MeanI: Without mean interval between active time steps

*Results saved to: `result/cascadegan_test/`*

**Time Series Length Analysis (Figures 20, 21, 22):**
```bash
bash config/experiments/figure_20_21_22.sh
```
Compares CascadeNet performance with different time series sequence lengths.

### Experimental Output

Experimental results are organized in two main directories:

**`result/cascadenet/`** - Contains results from most experiments:
- Primary baseline comparisons (figures 2, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 23, table 2)
- Anomaly detection experiments (figures 10, 24, 25, 26, 27, 28)
- Optimization techniques ablation (figure 18)
- Time series length analysis (figures 20, 21, 22)

**`result/cascadegan_test/`** - Contains results from specific ablation studies:
- Backbone architecture comparison experiments from figure 16:
  - MLP-MLP architecture results
  - MLP-RNN architecture results  
  - RNN-RNN architecture results
- Conditional input feature ablation experiments from figure 19:
  - Results for each feature removal variant (CN-w/o-DUR, CN-w/o-MaxPR, etc.)

# Evaluation

This section provides instructions for reproducing the experimental results and figures presented in the paper. 

## Reproducing Paper Results

This section reproduces all figures and tables presented in the paper. All generated results will be saved in `result/evaluation/`.

### Prerequisites
1. Ensure data for all baseline generators is available. Pre-generated outputs for baseline methods are already provided (see [Data](#data) section for details).
You only need to run CascadeNet to generate its outputs before running the evaluation.

2. Navigate to the evaluation directory:
   ```bash
   cd evaluation/
   ```

3. Activate the evaluation environment:
   ```bash
   conda activate cascadenet_eval
   ```

### Complete Evaluation Pipeline

To reproduce all results with a single command:

```bash
bash run.sh
```
#### Approximately Runtime: 1 day

### Individual Evaluation Components

##### Feature Comparison Analysis
*Reproduces Figures 2, 6, 7, 12, 13, 16, 17, 18, 20, 21, 22, 23*

```bash
cd stats/
bash plot_figures_paper.sh
```

This component generates statistical/temporal/time series comparisons between synthetic and real data across multiple feature dimensions.

**Important:** The `stats` module generates figures as specified in the script `/evaluation/stats/plot_figures_paper.sh`. This  script performs multi-stage analysis where the initial `diff_syn` experiments must be executed first to generate the `normalization.csv` file, which serves as the unified normalization standard for Earth Mover's Distance (EMD) calculations across all subsequent figures. The processing pipeline includes:

**Stage 1: Differential Synthesis Analysis (Figures 6, 7, 12, 13)**
- Temporal feature analysis: PDF and distribution comparisons
- Statistical feature analysis: PDF and distribution comparisons  

**Subsequent Stages:**
- Time series trace generation and packet rate analysis (Figures 2, 23)
- Ablation studies of optimization techniques (Figure 18)
- Time series length sensitivity analysis (Figures 20, 21, 22)
- Timestamp recovery evaluation (Figure 17)
- Architecture comparison analysis (Figure 16)
- Conditional input ablation studies (Figure 19)

##### Anomaly Detection Evaluation
*Reproduces Figures 10, 24, 25, 26, 27, 28*

```bash
bash anomaly_detection_cross/run.sh
```

Compares F1-score (↑) of anomaly detection models trained on synthetic vs. real traces to assess detection effectiveness.

##### Burst Analysis
*Reproduces Figures 8, 14*

```bash
cd burst_analysis/
bash run.sh
```

Measures how well synthetic traces preserve traffic burst patterns using Normalized EMD (↓) between real and synthetic burst metrics.

##### Throughput Prediction
*Reproduces Figures 9, 15*

```bash
cd throughput_prediction_steps/
bash run.sh
```

Measures the Normalized MAE (↓) between real throughput and predictions from models trained on synthetic traces.

##### Synthetic-Real Divergence Analysis
*Reproduces Table 2*

```bash
cd syn_real_div/
python run.py
```

Quantifies how much synthetic data diverges from training data, using record-level differences (↑ = more divergence).

##### Scalability Analysis
*Reproduces Figure 11*

```bash
cd scalability/
python run.py
```

Evaluates the computational scalability of synthetic trace generators by measuring training and generation time.

## Output Organization

All experimental results are systematically organized under `result/evaluation/` with the following structure:

- Stats: `result/evaluation/stats/`
- Anomaly detection results: `result/evaluation/anomaly_detection/`
- Burst analysis outputs: `result/evaluation/burst_analysis/`
- Throughput prediction results: `result/evaluation/throughput_prediction_steps/`
- Divergence analysis: `result/evaluation/syn_real_div/`
- Scalability metrics: `result/evaluation/scalability/`
