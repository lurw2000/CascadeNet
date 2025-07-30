#!/usr/bin/env bash
set -e  # Exit on error

# Determine the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
NETDIFFUSION_DIR="$SCRIPT_DIR"
RESULTS_DIR="$BASELINES_DIR/../../result/netdiffusion"
PREPROCESSED_IMGS="$RESULTS_DIR/preprocessed_fine_tune_imgs"

# Create necessary directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$PREPROCESSED_IMGS"

# Set the default Gradio URL
GRADIO_URL="http://localhost:7860"

# Inform the user about the Gradio URL being used
echo "==> Starting the WebUI with GRADIO_URL: $GRADIO_URL"

# Provide instructions to the user
echo ""
echo "==> Automation Script Started"
echo "==> Since there are 4290 flows in the ton_iot dataset, please ensure the following:"
echo "    - If in the 'Data Preprocessing' process, verify that all images (from flow_0.png to flow_4289.png) are properly generated under:"
echo "        $NETDIFFUSION_DIR/fine_tune/kohya_ss_fork/model_training/test/image/4290_network"
echo "==> Once you have verified that all images are properly generated, press \"Ctrl + C\" to terminate the WebUI."
echo ""
echo "==> If in the 'Install ControlNet' process, you need to press \"Ctrl + C\" to terminate and relaunch the WebUI using bash launch_webui.sh"
echo ""
# Navigate to the stable-diffusion-webui directory and launch the WebUI
cd "$NETDIFFUSION_DIR/fine_tune/sd-webui-fork/stable-diffusion-webui/"
bash webui.sh
