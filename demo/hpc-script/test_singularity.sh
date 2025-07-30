cd ~/ML-testing-dev
pip install -e .
python -c 'import torch; print(torch.cuda.is_available())'