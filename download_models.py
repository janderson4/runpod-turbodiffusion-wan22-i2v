import os
import subprocess

os.makedirs("checkpoints", exist_ok=True)
os.chdir("checkpoints")

def wget(url):
    subprocess.run(["wget", "-nc", url], check=True)

# Wan 2.2 VAE + text encoder
wget("https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth")
wget("https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth")

# TurboDiffusion Wan 2.2 I2V quantized models
wget("https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth")
wget("https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth")
