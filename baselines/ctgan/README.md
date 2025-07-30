# CTGAN

## installation
Using `pip`
```
pip install ctgan
```

Using `conda`
```
conda install -c pytorch -c conda-forge ctgan
```

## Quick Start with run.sh

To generate all synthetic data by ctgan

```
bash run.sh
```

## training
The input dataset should be a csv file.
```
python training --input=<INPUT_DATASET> --output=<OUTPUT_FOLDER> --type=<pcap|netflow>
```

## evaluation
The input dataset should be a csv file.
```
python evaluating --input=<INPUT_DATASET> --output=<OUTPUT_FOLDER> --type=<pcap|netflow>
```
