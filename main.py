import os
import argparse

MODEL_PATH = "./models"


def parse_args():
    parser = argparse.ArgumentParser(description="Run stereo video generation.")
    parser.add_argument(
        "--video_path",
        "--video-path",
        default="test/1.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        default="output/1.mp4",
        help="Output video path.",
    )
    return parser.parse_args()


args = parse_args()

import torch
from diffsynthfinal import save_video
from diffsynthfinal.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import traceback
from util import *

from pydantic import BaseModel
from typing import Optional
import cv2
import gc
import asyncio
from moviepy import VideoFileClip


def get_caption_model(device):
    """懒加载caption模型"""
    caption_model, caption_processor, caption_device = init_caption_model(
        MODEL_PATH, device
    )
    return caption_model, caption_processor, caption_device



video_path = args.video_path
output_path = args.output_path

device = "cuda:0"


out_dir = os.path.dirname(output_path)

out_path = output_path

if out_dir:
    os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
else:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

video_loader = default_video_operator(
    enable_long_video=True,
    force_resolution=False,
    height=height,
    width=width,
    overlap_frames=9,
)
input_data = video_loader(video_path)



if isinstance(input_data[0], dict):

    caption_model, caption_processor, caption_device = get_caption_model(device)

    generated_segments = []
    segment_captions = []
    segment_info_list = []
    processed_segments = []

    for segment_idx, segment_info in enumerate(input_data):

        segment_frames = segment_info["frames"]
        segment_info_list.append(segment_info)

        segment_caption = caption_video_segment(
            segment_frames,
            caption_model,
            caption_processor,
            caption_device,
            segment_idx + 1,
            fps,
        )

        segment_captions.append(segment_caption)

        segment_video = preprocess_video(segment_frames)


        processed_segments.append(segment_video)

    del caption_model

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
            ),
            ModelConfig(
                path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
            ),
            ModelConfig(path=f"{MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
        ],
    )
    pipe.load_lora(
        pipe.dit,
        f"{MODEL_PATH}/stereo.safetensors",
        alpha=1,
    )
    pipe.enable_vram_management()

    generated_segments = pipe.long_video(
        prompt=segment_captions,
        input_video=processed_segments,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=0,
        tiled=True,
        height=height,
        width=width,
    )

    concat_videos_with_blending(
        generated_segments,
        segment_info_list,
        video_path,
        out_path,
        fps=fps,
        overlap_frames=9,
    )
