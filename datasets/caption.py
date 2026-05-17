import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
from pathlib import Path
import csv
import multiprocessing
import math
import torch.multiprocessing as mp
import tqdm


def init_model(gpu_id):
    device = f"cuda:{gpu_id}"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, device


def process_video_batch(args):
    gpu_id, video_files, csv_path, lock = args
    # Initialize model for this process
    model, processor, device = init_model(gpu_id)

    for video_file in tqdm.tqdm(
        video_files, desc=f"GPU{gpu_id} PID{os.getpid()}", position=gpu_id
    ):
        # Check if video_file.name already exists in CSV
        with lock:
            exists = False
            if os.path.exists(csv_path):
                with open(csv_path, mode="r", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row and row[0] == video_file.name:
                            exists = True
                            break
            if exists:
                continue

            conversation = [
            {        
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": {"video_path": str(video_file), "fps": 1, "max_frames": 2000}
                    },
                    {
                        "type": "text", 
                        "text": "Describe the video."
                    },
                ]
            }
        ]
        try:
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception as e:

            print(f"Error processing video {video_file}: {str(e)}")
            continue

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        # Safe CSV writing with lock
        with lock:
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([video_file.name, response])


if __name__ == "__main__":
    left_video_dir = (
        "./split_movies/left"
    )
    csv_path = "./split_movies/metadata_caption.csv"

    # Initialize CSV
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file_name", "text"])

    # All videos
    video_files = list(reversed(sorted(Path(left_video_dir).glob("*.mp4"))))

    # GPU count
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # 每个 GPU 跑 3 个进程
    workers_per_gpu = 2
    total_workers = num_gpus * workers_per_gpu

    # 任务平均分配到 total_workers
    total_videos = len(video_files)
    batch_size = math.ceil(total_videos / total_workers)
    batches = [
        video_files[i : i + batch_size] for i in range(0, total_videos, batch_size)
    ]

    # Manager 提供共享锁
    manager = mp.Manager()
    lock = manager.Lock()

    # 给每个 batch 分配 GPU 和锁
    args_list = []
    for i, batch in enumerate(batches):
        gpu_id = i % num_gpus  # 轮流分配 GPU
        args_list.append((gpu_id, batch, csv_path, lock))

    # 启动 total_workers 个进程
    mp.set_start_method("spawn", force=True)
    with mp.Pool(total_workers) as pool:
        pool.map(process_video_batch, args_list)
