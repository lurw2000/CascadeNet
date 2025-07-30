## E-WGAN-GP Baseline README
---

### Installation
Create a new Conda environment:
```bash
conda create -n ewgan python=3.10 -y
conda activate ewgan
``` 

Install the required dependencies:
```bash
pip install -r requirements.txt
``` 

### Usage
#### Main Script Execution
To run the E-WGAN-GP pipeline, use the run.sh script:

```bash
bash run.sh <dataset_name>
```
#### Available Datasets
You can choose from the following datasets when running the script: caida, ca, or dc
For example, to process the caida dataset, run:

```bash
bash run.sh caida
```