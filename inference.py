import os
import subprocess
import uuid

CHECKPOINTS = "/checkpoints"
OUTPUT_DIR = "/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_i2v(
    image_path,
    prompt,
    num_steps=4,
    num_frames=81,
    resolution="720p",
    seed=0,
):
    output_path = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp4"

    cmd = [
        "python",
        "turbodiffusion/inference/wan2.2_i2v_infer.py",
        "--model", "Wan2.2-A14B",
        "--high_noise_model_path", f"{CHECKPOINTS}/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "--low_noise_model_path", f"{CHECKPOINTS}/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        "--vae_path", f"{CHECKPOINTS}/Wan2.1_VAE.pth",
        "--text_encoder_path", f"{CHECKPOINTS}/models_t5_umt5-xxl-enc-bf16.pth",
        "--image_path", image_path,
        "--prompt", prompt,
        "--num_steps", str(num_steps),
        "--num_frames", str(num_frames),
        "--resolution", resolution,
        "--seed", str(seed),
        "--quant_linear",
        "--attention_type", "sagesla",
        "--sla_topk", "0.1",
        "--ode",
        "--save_path", output_path
    ]

    subprocess.run(cmd, check=True)
    return output_path
