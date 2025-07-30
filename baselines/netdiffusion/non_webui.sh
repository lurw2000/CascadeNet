#!/usr/bin/env bash
set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
NETDIFFUSION_DIR="$SCRIPT_DIR"
RESULTS_DIR="$BASELINES_DIR/../result/netdiffusion"
PREPROCESSED_IMGS="$RESULTS_DIR/preprocessed_fine_tune_imgs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$PREPROCESSED_IMGS"

# Initialize Conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

#########################################
# Data Preprocessing
#########################################
echo "==> Process 1: Data Preprocessing"
echo "==> 1a. Splitting pcap by flow and converting to images"
conda activate netdiffusion
cd "$NETDIFFUSION_DIR/data_preprocessing"

echo "Running split_by_flow.py with input pcap"
python3 split_by_flow.py

# Convert PCAP flows to images
echo "Running pcap_to_img.py to generate images"
python3 pcap_to_img.py

# Automate webui interaction using Selenium
cd "$NETDIFFUSION_DIR/fine_tune"
input_dir="$RESULTS_DIR/preprocessed_fine_tune_imgs"
output_dir="$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/image/4290_network"
python3 automate_webui.py --input_dir "$input_dir" --output_dir "$output_dir"


echo "==> Please verify that all images are properly generated in the specified output directory:"
echo "==> Since there are 4290 flows in ton_iot dataset, please check whether flow_0.png to flow_4289.png are all present at:"
echo "    $NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/image/4290_network"
echo ""
echo "==> Once you have confirmed that the images are generated correctly, press ENTER to proceed with caption modification."
read -p "Press ENTER to continue..."

# Modify captions
echo "==> 1b. Running caption modification"
cd "$NETDIFFUSION_DIR/fine_tune"
python3 caption_changing.py "test/image/4290_network"
conda deactivate

#########################################
# Fine-Tuning
#########################################
echo "==> Process 2: Fine-Tuning"
echo "==> 2a. Beginning fine-tuning (LoRA) with accelerate"
cd "$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork"
conda activate lora

# Define variables
CONFIG_DIR="${HOME}/.cache/huggingface/accelerate"
CONFIG_FILE="${CONFIG_DIR}/default_config.yaml"

# Create the configuration directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_FILE" <<EOL
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOL

# Set appropriate permissions (optional)
chmod 600 "$CONFIG_FILE"
echo "Accelerate configuration has been created at $CONFIG_FILE"

accelerate launch --num_cpu_threads_per_process=2 \
  "./train_network.py" \
  --bucket_no_upscale \
  --bucket_reso_steps=64 \
  --cache_latents \
  --caption_extension=".txt" \
  --clip_skip=2 \
  --gradient_checkpointing \
  --learning_rate="0.0001" \
  --logging_dir="$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/log" \
  --lr_scheduler="constant" \
  --lr_scheduler_num_cycles="1" \
  --max_data_loader_n_workers="1" \
  --max_grad_norm="1" \
  --resolution="816,768" \
  --max_train_steps="2" \
  --mem_eff_attn \
  --mixed_precision="fp16" \
  --network_alpha="128" \
  --network_dim=128 \
  --network_module=networks.lora \
  --optimizer_type="AdamW8bit" \
  --output_dir="$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/model" \
  --output_name="test_task_model" \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --save_every_n_epochs="1" \
  --save_model_as="safetensors" \
  --save_precision="fp16" \
  --seed="1234" \
  --text_encoder_lr=5e-05 \
  --train_batch_size="1" \
  --train_data_dir="$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/image" \
  --unet_lr=0.0001 \
  --xformers



# COPY FINE-TUNED MODEL AND PREP WEBUI
echo "==> 2b. Copying fine-tuned LoRA model to stable-diffusion-webui"
conda activate netdiffusion
cd "$NETDIFFUSION_DIR/fine_tune/kohya_ss_fork"

# Copy the trained model
cp model_training/test/model/test_task_model.safetensors \
   "$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/stable-diffusion-webui/models/Lora/"

Download Stable Diffusion v1.5 if not present
cd "$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/models/Stable-diffusion/"
if [ ! -f "v1-5-pruned.safetensors" ]; then
  echo "==> Downloading Stable Diffusion v1.5"
  wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors
else
  echo "Stable Diffusion v1.5 already exists."
fi

Clone ControlNet repository if not present
cd "$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/extensions/"
if [ ! -d "sd-webui-controlnet" ]; then
  echo "==> Cloning sd-webui-controlnet repository"
  git clone https://github.com/Mikubill/sd-webui-controlnet.git
else
  echo "sd-webui-controlnet repository already exists."
fi

Download ControlNet model if not present
cd "$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/stable-diffusion-webui/stable-diffusion-webui/extensions/sd-webui-controlnet/models/"
if [ ! -f "control_v11p_sd15_canny.pth" ]; then
  echo "==> Downloading ControlNet canny model"
  wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
else
  echo "ControlNet canny model already exists."
fi

conda deactivate

conda activate generate
cd "$NETDIFFUSION_DIR/generate"
python install_extensionPanel.py
conda deactivate

#########################################
# Generation
#########################################
echo "==> Process 3: Beginning generation process"
conda activate generate
cd "$NETDIFFUSION_DIR/generate"
python iterate.py
conda deactivate


#########################################
# POST-GENERATION CORRECTION
#########################################
echo "==> Process 4: Post-Generation"
echo "==> Copying generated images and running correction scripts"
GENERATED_OUTPUT_DIR="$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/stable-diffusion-webui/stable-diffusion-webui/outputs/txt2img-images"
mkdir -p "$RESULTS_DIR/generated_imgs"

find "$GENERATED_OUTPUT_DIR" -type f -name "*.png" -exec cp -v {} "$RESULTS_DIR/generated_imgs" \;
if [ $(ls -1 "$RESULTS_DIR/generated_imgs"/*.png 2>/dev/null | wc -l) -eq 0 ]; then
    echo "No images to copy."
else
    echo "Images successfully copied to $RESULTS_DIR/generated_imgs."
fi

conda activate netdiffusion
cd "$NETDIFFUSION_DIR/post_generation"

echo "Running color_processor.py"
python3 color_processor.py

echo "Running img_to_nprint.py"
python3 img_to_nprint.py

echo "Running mass_reconstruction.py"
python3 mass_reconstruction.py

echo "Running final_process.py"
python3 final_process.py

conda deactivate


# DONE
echo "==> NetDiffusion pipeline completed!"
echo "Check your results under: $RESULTS_DIR"