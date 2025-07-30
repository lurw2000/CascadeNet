## STAN Baseline README
This repository contains scripts and tools for analyzing and synthesizing network traffic data using the STAN framework.

---

### Installation
Create a new Conda environment:
```bash
conda create -n stan python=3.10 -y
conda activate stan
``` 

Install the required dependencies:
```bash
pip install stannetflow
pip install -r requirements.txt
``` 

### Usage
#### Main Script Execution
To run the STAN pipeline, use the run.sh script:

```bash
bash run.sh <dataset_name>
```
#### Available Datasets
You can choose from the following datasets when running the script: caida, ca, or dc
For example, to process the caida dataset, run:

```bash
bash run.sh caida
```