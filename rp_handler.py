import runpod
import base64
import requests
import os
from inference import run_i2v

def save_image(input_data):
    if "image_url" in input_data:
        img = requests.get(input_data["image_url"]).content
    else:
        img = base64.b64decode(input_data["image_base64"])

    path = "/tmp/input.jpg"
    with open(path, "wb") as f:
        f.write(img)
    return path

def handler(event):
    inp = event["input"]

    image_path = save_image(inp)
    prompt = inp["prompt"]

    video_path = run_i2v(
        image_path=image_path,
        prompt=prompt,
        num_steps=inp.get("num_steps", 4),
        seed=inp.get("seed", 0),
    )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    return {
        "video_base64": video_b64
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
