import os
import requests
import time
import io
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.models import AutoencoderKL
from RealESRGAN import RealESRGAN
import gradio as gr
import subprocess
from tqdm import tqdm
import shutil
import uuid
import json
import threading

# Constants
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a CUDA-capable GPU.")

device = torch.device("cuda")
print(f"Using device: {device}")

# Replace the global abort_status with an Event
abort_event = threading.Event()

css = """
.gradio-container {
    max-width: 100% !important;
    padding: 20px !important;
}
#component-0 {
    height: auto !important;
    overflow: visible !important;
}
"""

def abort_job():
    if abort_event.is_set():
        return "Job is already being aborted."
    abort_event.set()
    return "Aborting job... Processing will stop after the current frame."

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_file(url, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        print(f"File already exists: {file_path}")
    else:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"File successfully downloaded and saved: {file_path}")
        else:
            print(f"Error downloading the file. Status code: {response.status_code}")

def download_models():
    models = {
        "MODEL": ("https://huggingface.co/dantea1118/juggernaut_reborn/resolve/main/juggernaut_reborn.safetensors?download=true", "models/models/Stable-diffusion", "juggernaut_reborn.safetensors"),
        "UPSCALER_X2": ("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true", "models/upscalers/", "RealESRGAN_x2.pth"),
        "UPSCALER_X4": ("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true", "models/upscalers/", "RealESRGAN_x4.pth"),
        "NEGATIVE_1": ("https://huggingface.co/philz1337x/embeddings/resolve/main/verybadimagenegative_v1.3.pt?download=true", "models/embeddings", "verybadimagenegative_v1.3.pt"),
        "NEGATIVE_2": ("https://huggingface.co/datasets/AddictiveFuture/sd-negative-embeddings/resolve/main/JuggernautNegative-neg.pt?download=true", "models/embeddings", "JuggernautNegative-neg.pt"),
        "LORA_1": ("https://huggingface.co/philz1337x/loras/resolve/main/SDXLrender_v2.0.safetensors?download=true", "models/Lora", "SDXLrender_v2.0.safetensors"),
        "LORA_2": ("https://huggingface.co/philz1337x/loras/resolve/main/more_details.safetensors?download=true", "models/Lora", "more_details.safetensors"),
        "CONTROLNET": ("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth?download=true", "models/ControlNet", "control_v11f1e_sd15_tile.pth"),
        "VAE": ("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true", "models/VAE", "vae-ft-mse-840000-ema-pruned.safetensors"),
    }

    for model, (url, folder, filename) in models.items():
        download_file(url, folder, filename)

def timer_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class ModelManager:
    def __init__(self):
        self.pipe = None
        self.realesrgan_x2 = None
        self.realesrgan_x4 = None

    def load_models(self, progress=gr.Progress()):
        if self.pipe is None:
            progress(0, desc="Loading Stable Diffusion pipeline...")
            self.pipe = self.setup_pipeline()
            self.pipe.to(device)
            if USE_TORCH_COMPILE:
                progress(0.5, desc="Compiling the model...")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        if self.realesrgan_x2 is None:
            progress(0.7, desc="Loading RealESRGAN x2 model...")
            self.realesrgan_x2 = RealESRGAN(device, scale=2)
            self.realesrgan_x2.load_weights('models/upscalers/RealESRGAN_x2.pth', download=False)

        if self.realesrgan_x4 is None:
            progress(0.9, desc="Loading RealESRGAN x4 model...")
            self.realesrgan_x4 = RealESRGAN(device, scale=4)
            self.realesrgan_x4.load_weights('models/upscalers/RealESRGAN_x4.pth', download=False)

        progress(1.0, desc="All models loaded successfully")

    def setup_pipeline(self):
        controlnet = ControlNetModel.from_single_file(
            "models/ControlNet/control_v11f1e_sd15_tile.pth", torch_dtype=torch.float16
        )
        model_path = "models/models/Stable-diffusion/juggernaut_reborn.safetensors"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None
        )
        vae = AutoencoderKL.from_single_file(
            "models/VAE/vae-ft-mse-840000-ema-pruned.safetensors",
            torch_dtype=torch.float16
        )
        pipe.vae = vae
        pipe.load_textual_inversion("models/embeddings/verybadimagenegative_v1.3.pt")
        pipe.load_textual_inversion("models/embeddings/JuggernautNegative-neg.pt")
        pipe.load_lora_weights("models/Lora/SDXLrender_v2.0.safetensors")
        pipe.fuse_lora(lora_scale=0.5)
        pipe.load_lora_weights("models/Lora/more_details.safetensors")
        pipe.fuse_lora(lora_scale=1.)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        return pipe

    @timer_func
    def process_image(self, input_image, resolution, num_inference_steps, strength, hdr, guidance_scale):
        condition_image = self.prepare_image(input_image, resolution, hdr)

        prompt = "masterpiece, best quality, highres"
        negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"

        options = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": condition_image,
            "control_image": condition_image,
            "width": condition_image.size[0],
            "height": condition_image.size[1],
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator(device=device).manual_seed(0),
        }

        print("Running inference...")
        result = self.pipe(**options).images[0]
        print("Image processing completed successfully")

        return result

    def prepare_image(self, input_image, resolution, hdr):
        condition_image = self.resize_and_upscale(input_image, resolution)
        condition_image = self.create_hdr_effect(condition_image, hdr)
        return condition_image

    @timer_func
    def resize_and_upscale(self, input_image, resolution):
        scale = 2 if resolution <= 2048 else 4

        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, io.IOBase):
            input_image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        else:
            raise ValueError(f"Unsupported input type for input_image: {type(input_image)}")

        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H = int(round(H * k / 64.0)) * 64
        W = int(round(W * k / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)

        if scale == 2:
            img = self.realesrgan_x2.predict(img)
        else:
            img = self.realesrgan_x4.predict(img)

        return img

    @timer_func
    def create_hdr_effect(self, original_image, hdr):
        if hdr == 0:
            return original_image
        cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        factors = [1.0 - 0.9 * hdr, 1.0 - 0.7 * hdr, 1.0 - 0.45 * hdr,
                   1.0 - 0.25 * hdr, 1.0, 1.0 + 0.2 * hdr,
                   1.0 + 0.4 * hdr, 1.0 + 0.6 * hdr, 1.0 + 0.8 * hdr]
        images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]
        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(images)
        hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
        hdr_result = Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

        return hdr_result

model_manager = ModelManager()

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', 'fps=30',
        f'{output_folder}/frame_%06d.png'
    ]
    subprocess.run(command, check=True)

def frames_to_video(input_folder, output_path, fps):
    command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', f'{input_folder}/frame_%06d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(command, check=True)

@timer_func
def process_video(input_video, resolution, num_inference_steps, strength, hdr, guidance_scale, max_frames=None, frame_interval=1, preserve_frames=False, progress=gr.Progress()):
    abort_event.clear()  # Clear the abort flag at the start of a new job
    print("Starting video processing...")
    model_manager.load_models(progress)  # Ensure models are loaded

    # Create a new job folder
    job_id = str(uuid.uuid4())
    job_folder = os.path.join("jobs", job_id)
    os.makedirs(job_folder, exist_ok=True)

    # Save job config
    config = {
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "hdr": hdr,
        "guidance_scale": guidance_scale,
        "max_frames": max_frames,
        "frame_interval": frame_interval,
        "preserve_frames": preserve_frames
    }
    with open(os.path.join(job_folder, "config.json"), "w") as f:
        json.dump(config, f)

    # If input_video is a file object or has a 'name' attribute, use its name
    if isinstance(input_video, io.IOBase) or hasattr(input_video, 'name'):
        input_video = input_video.name

    # Set up folders
    frames_folder = os.path.join(job_folder, "video_frames")
    processed_frames_folder = os.path.join(job_folder, "processed_frames")
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(processed_frames_folder, exist_ok=True)

    # Extract frames
    progress(0.1, desc="Extracting frames...")
    extract_frames(input_video, frames_folder)

    # Process selected frames
    frame_files = sorted(os.listdir(frames_folder))
    total_frames = len(frame_files)
    frames_to_process = min(max_frames, total_frames) if max_frames else total_frames

    try:
        progress(0.2, desc="Processing frames...")
        for i, frame_file in enumerate(tqdm(frame_files[:frames_to_process], desc="Processing frames")):
            if abort_event.is_set():
                print("Job aborted. Stopping processing of new frames.")
                break

            output_frame_path = os.path.join(processed_frames_folder, frame_file)
            if not preserve_frames or not os.path.exists(output_frame_path):
                if i % frame_interval == 0:
                    # Process this frame
                    input_image = Image.open(os.path.join(frames_folder, frame_file))
                    processed_image = model_manager.process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale)
                    processed_image.save(output_frame_path)
                else:
                    # Copy the previous processed frame or the original frame
                    prev_frame = f"frame_{int(frame_file.split('_')[1].split('.')[0]) - 1:06d}.png"
                    prev_frame_path = os.path.join(processed_frames_folder, prev_frame)
                    if os.path.exists(prev_frame_path):
                        shutil.copy2(prev_frame_path, output_frame_path)
                    else:
                        shutil.copy2(os.path.join(frames_folder, frame_file), output_frame_path)
            progress((0.2 + 0.7 * (i + 1) / frames_to_process), desc=f"Processing frame {i+1}/{frames_to_process}")

        # Always attempt to reassemble video
        progress(0.9, desc="Reassembling video...")
        input_filename = os.path.splitext(os.path.basename(input_video))[0]
        output_video = os.path.join(job_folder, f"{input_filename}_upscaled.mp4")
        frames_to_video(processed_frames_folder, output_video, 30)

        if abort_event.is_set():
            progress(1.0, desc="Video processing aborted, but partial result saved")
            print("Video processing aborted, but partial result saved")
        else:
            progress(1.0, desc="Video processing completed successfully")
            print("Video processing completed successfully")

        return output_video

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        progress(1.0, desc=f"Error: {str(e)}")
        return None

def gradio_process_media(input_media, resolution, num_inference_steps, strength, hdr, guidance_scale, max_frames, frame_interval, preserve_frames, progress=gr.Progress()):
    abort_event.clear()  # Clear the abort flag at the start of a new job
    if input_media is None:
        return None, "No input media provided."

    print(f"Input media type: {type(input_media)}")

    # Get the file path
    if isinstance(input_media, str):
        file_path = input_media
    elif isinstance(input_media, io.IOBase):
        file_path = input_media.name
    elif hasattr(input_media, 'name'):
        file_path = input_media.name
    else:
        raise ValueError(f"Unsupported input type: {type(input_media)}")

    print(f"File path: {file_path}")

    # Ensure models are loaded
    model_manager.load_models(progress)

    # Check if the file is a video
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    if file_path.lower().endswith(video_extensions):
        print("Processing video...")
        result = process_video(file_path, resolution, num_inference_steps, strength, hdr, guidance_scale, max_frames, frame_interval, preserve_frames, progress)
        if result:
            return result, "Video processing completed successfully."
        else:
            return None, "Error occurred during video processing."
    else:
        print("Processing image...")
        result = model_manager.process_image(file_path, resolution, num_inference_steps, strength, hdr, guidance_scale)
        if result:
            # Save the processed image
            output_path = os.path.join("processed_images", f"processed_{os.path.basename(file_path)}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path)
            return output_path, "Image processing completed successfully."
        else:
            return None, "Error occurred during image processing."

# Update the Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="blue")) as iface:
    gr.Markdown(
        """
        # SimpleSlowVideoUpscaler

        Built by [Hrishi](https://twitter.com/hrishioa) and Claude

        This project is based on [gokaygokay/Tile-Upscaler](https://huggingface.co/spaces/gokaygokay/Tile-Upscaler), which in turn is inspired by ideas from [@philz1337x/clarity-upscaler](https://github.com/philz1337x/clarity-upscaler) and [@BatouResearch/controlnet-tile-upscale](https://github.com/BatouResearch/controlnet-tile-upscale).

        If you find this project useful, please consider [starring it on GitHub](https://github.com/hrishioa/SimpleSlowVideoUpscaler)!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_media = gr.File(label="Input Media (Image or Video)")
            resolution = gr.Slider(256, 2048, 512, step=256, label="Resolution")
            num_inference_steps = gr.Slider(1, 50, 10, step=1, label="Number of Inference Steps")
            strength = gr.Slider(0, 1, 0.3, step=0.01, label="Strength")
            hdr = gr.Slider(0, 1, 0, step=0.1, label="HDR Effect")
            guidance_scale = gr.Slider(0, 20, 5, step=0.5, label="Guidance Scale")
            max_frames = gr.Number(label="Max Frames to Process (leave empty for full video)", precision=0)
            frame_interval = gr.Slider(1, 30, 1, step=1, label="Frame Interval (process every nth frame)")
            preserve_frames = gr.Checkbox(label="Preserve Existing Processed Frames", value=True)

        with gr.Column(scale=1):
            submit_button = gr.Button("Process Media")
            abort_button = gr.Button("Abort Job")
            output = gr.File(label="Processed Media")
            status = gr.Markdown("Ready to process media.")

    submit_button.click(
        gradio_process_media,
        inputs=[input_media, resolution, num_inference_steps, strength, hdr, guidance_scale, max_frames, frame_interval, preserve_frames],
        outputs=[output, status]
    )

    abort_button.click(abort_job, inputs=[], outputs=status)

# Launch the Gradio app
iface.launch()