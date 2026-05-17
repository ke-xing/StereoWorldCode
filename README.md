# StereoWorld

## Environment Setup

Create and activate a conda environment:

```bash
conda create -n stereoworld python=3.11 -y
conda activate stereoworld
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

FFmpeg is required for video IO. If it is not already available on your system, install it with conda:

```bash
conda install -c conda-forge ffmpeg -y
```

If your machine requires a specific PyTorch/CUDA build, install the matching PyTorch package for your GPU driver and CUDA runtime before running inference.

## Model Weights

Download all required model weights into the `models` directory.

```bash
mkdir -p models
pip install -U "huggingface_hub[cli]"
```

Download the StereoWorld weights:

```bash
huggingface-cli download KXingLab/stereoworld --local-dir ./models
```

Download Wan2.1-T2V-1.3B:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan-AI/Wan2.1-T2V-1.3B
```

Download VideoLLaMA3-7B:

```bash
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B --local-dir ./models/VideoLLaMA3-7B
```

The inference code expects the following files and directories:

```text
models/
+-- stereo.safetensors
+-- VideoLLaMA3-7B/
+-- Wan-AI/
    +-- Wan2.1-T2V-1.3B/
        +-- diffusion_pytorch_model.safetensors
        +-- models_t5_umt5-xxl-enc-bf16.pth
        +-- Wan2.1_VAE.pth
```

Please follow the license terms and usage conditions of the corresponding model repositories:

- [KXingLab/stereoworld](https://huggingface.co/KXingLab/stereoworld)
- [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [DAMO-NLP-SG/VideoLLaMA3-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)

## Inference Pipeline

The inference pipeline in `main.py` performs the following steps:

1. Loads the input video and splits long videos into overlapping segments.
2. Uses VideoLLaMA3 to generate captions for each video segment.
3. Uses Wan2.1-T2V-1.3B with the StereoWorld LoRA weights to generate stereo video segments.
4. Blends and concatenates the generated segments into the final output video.

Run inference with:

```bash
python main.py --video_path test/1.mp4 --output_path output/1.mp4
```

The script also supports dashed argument names:

```bash
python main.py --video-path test/1.mp4 --output-path output/1.mp4
```

Arguments:

- `--video_path`, `--video-path`: input video path. Default: `test/1.mp4`.
- `--output_path`, `--output-path`: output video path. Default: `output/1.mp4`.

## Dataset Construction

For dataset preparation, movie source handling, caption generation, depth annotation, and disparity annotation, see the dataset processing guide:

[datasets/data.md](./datasets/data.md)
