from setuptools import setup, find_packages

setup(
    name="nta",
    author = 'Runwei, Xinyu',
    version = '0.1',
    packages = find_packages(),
    install_requires = [
        "torch==2.4.1",
        "tensorboard==2.14.0",
        "gensim==3.8",
        "scapy==2.4.5",
        "pandas==1.5.3",
        "scikit-learn==1.1.3",
        "matplotlib==3.7.5",
        "tqdm==4.67.1",
        "packaging==25.0",
        "pywavelets==1.4.1",
        "annoy==1.17.3",
        "umap-learn==0.5.7",
        "seaborn==0.13.2",
        "paramiko==3.5.1",
        "nolds==0.6.2",
        "pyyaml==6.0.2",
        "statsmodels==0.14.1",
    ],
)