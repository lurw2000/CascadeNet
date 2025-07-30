## NetDiffusion Baseline README
NetDiffusion transforms .pcap files into images, fine-tunes a diffusion model (via LoRA), and generates synthetic network data images (with reverse conversion to .pcap). This repository is designed for Linux systems and leverages Stable Diffusion WebUI and ControlNet.

---

### 1. Setting Up SSH Connection
Establish an SSH connection with port forwarding:

```bash
ssh -L 7860:LocalHost:7860 username@server_address
```

### 2. Environment Setup
Before running the scripts, set up the required environments:

**NetDiffusion Environment:**
```bash
conda create -n netdiffusion python=3.10 -y
conda activate netdiffusion
pip install --extra-index-url https://pypi.nvidia.com tensorrt-libs==8.6.1
conda env update \
  --name netdiffusion \
  --file environments/netdiffusion.yml \
  --prune
```

**Generate Environment:**
```bash
conda env create -f environments/generate.yml
```

**Lora Environment:**
```bash
conda env create -f environments/lora.yml
```

### 3. Install Dependencies
**nprint Installation:**
Install nprint and its dependencies using:
```bash
bash dependencies/install_dependencies.sh
```
Then run a quick sanity check for verification:
```bash
source dependencies/sanity_check.sh
```

**Chrome and Chromedriver Installation:**
```bash
bash dependencies/install+test_chrome138.sh
```

### 4. Running the Pipeline
**Terminal 1:**
Connect via SSH as shown above.
Clone stable diffusion and run the WebUI:
```bash
cd fine_tune/sd-webui-fork/stable-diffusion-webui/
git clone git@github.com:AUTOMATIC1111/stable-diffusion-webui.git
cd ../../../
conda activate netdiffusion
bash launch_webui.sh
```

**Terminal 2:**
Run the pipeline under netdiffusion baseline directory:
```bash
bash non_webui.sh
```
**Note**: Follow the on-screen instructions and wait for all images to be generated before proceeding to the next steps.

