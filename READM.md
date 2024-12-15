# Rockchip NPU Multimodal Conversion script

### This repository contains a full pipeline to convert Qwen2VL safetensors models to run on the RK3588 NPU

## To use

1. Clone this repo and cd to the directory
2. Make sure you have enough RAM. If you don't, use `fallocate` to expand RAM with a swapfile.
3. Build and run the Docker container
4. Select your model (2B or 7B)
5. Enjoy