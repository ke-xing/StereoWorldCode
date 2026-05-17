import sys
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from pathlib import Path


DEVICE = 'cuda'


def demo(args):
    from stereoanyvideo.models.stereoanyvideo_model import StereoAnyVideoModel
    model = StereoAnyVideoModel()

    if args.ckpt is not None:
        assert args.ckpt.endswith(".pth") or args.ckpt.endswith(
            ".pt"
        )
        strict = True
        state_dict = torch.load(args.ckpt,weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model.model.load_state_dict(state_dict, strict=strict)
        print("Done loading model checkpoint", args.ckpt)

    model.to(DEVICE)
    model.eval()


    output_directory = args.out_dir
    if not os.path.exists(output_directory) and output_directory != '':
        os.makedirs(output_directory)

    with torch.no_grad():
        cap_left = cv2.VideoCapture(args.left_file)
        cap_right = cv2.VideoCapture(args.right_file)
        
        if not cap_left.isOpened():
            raise IOError(f"Cannot open left video file: {args.left_file}")
        if not cap_right.isOpened():
            raise IOError(f"Cannot open right video file: {args.right_file}")

        num_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        assert num_frames_left == num_frames_right, \
            f"Frame count mismatch: {num_frames_left} (left) vs {num_frames_right} (right)"
        
        num_frames = num_frames_left
        print(f"Found {num_frames} frames. Saving NPZ to {args.name}")

        frame_size = args.frame_size
        disparities_ori_all = [] # 这将存储 float 类型的视差

        for start_idx in range(0, num_frames, frame_size):
            end_idx = min(start_idx + frame_size, num_frames)

            image_left_list = []
            image_right_list = []

            for i in range(start_idx, end_idx):
                ret_left, frame_left_bgr = cap_left.read()
                ret_right, frame_right_bgr = cap_right.read()

                if not ret_left or not ret_right:
                    print(f"Warning: Could not read frame {i}. Stopping batch.")
                    break # 退出内部循环


                frame_left_bgr = frame_left_bgr[args.top : height - args.bottom, :, :]
                frame_right_bgr = frame_right_bgr[args.top : height - args.bottom, :, :]
                

                frame_left_rgb = frame_left_bgr[..., ::-1].copy()
                frame_right_rgb = frame_right_bgr[..., ::-1].copy()

                image_left = torch.from_numpy(frame_left_rgb).permute(2, 0, 1).float().to(DEVICE)
                image_right = torch.from_numpy(frame_right_rgb).permute(2, 0, 1).float().to(DEVICE)

                image_left_list.append(image_left)
                image_right_list.append(image_right)
            
            if not image_left_list:
                break # 如果批次为空 (例如, read 失败), 退出外部循环

            video_left = torch.stack(image_left_list, dim=0)
            video_right = torch.stack(image_right_list, dim=0)

            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = torch.stack([video_left, video_right], dim=1)

            predictions = model(batch_dict)

            assert "disparity" in predictions
            # 保留 float 类型
            disparities = predictions["disparity"][:, :1].clone().data.cpu().abs().numpy()
            disparities_ori_all.extend(disparities)

        cap_left.release()
        cap_right.release()

        # --- 数据后处理和保存 ---

        if not disparities_ori_all:
            print("No frames processed. Exiting.")
            return

        disparities_ori_all = np.array(disparities_ori_all) # Shape [N, 1, H, W]

        epsilon = 1e-5  # Smallest allowable disparity
        disparities_ori_all[disparities_ori_all < epsilon] = epsilon

        # --- 1. 保存可视化视频 ---
        
        num_frames_processed, _, H, W = disparities_ori_all.shape
        frame_size_cv2 = (W, H)

        # 归一化版本 (用于 disparity_norm.mp4)
        disparities_norm_all = (
            (disparities_ori_all - disparities_ori_all.min()) / 
            (disparities_ori_all.max() - disparities_ori_all.min() + 1e-6) * 255
        ).astype(np.uint8)

        # 原始截断版本 (用于 disparity_ori.mp4)
        disparities_uint8_all = disparities_ori_all.astype(np.uint8)

        video_ori_disparity = cv2.VideoWriter(
            os.path.join(output_directory, f"{args.name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=frame_size_cv2,
            isColor=True,
        )
        video_norm_disparity = cv2.VideoWriter(
            os.path.join(output_directory, "disparity_norm.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=frame_size_cv2,
            isColor=True,
        )

        print(f"Saving visualization videos to {output_directory}...")
        for i in range(num_frames_processed):
            # 归一化视频
            disparity_norm = disparities_norm_all[i] # Shape [1, H, W]
            disparity_norm = disparity_norm.transpose(1, 2, 0) # Shape [H, W, 1]
            disparity_norm_vis = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_INFERNO)
            video_norm_disparity.write(disparity_norm_vis)

            # 原始 (uint8) 视频
            disparity_ori = disparities_uint8_all[i] # Shape [1, H, W]
            disparity_ori = disparity_ori.transpose(1, 2, 0) # Shape [H, W, 1]
            disparity_ori_vis = cv2.applyColorMap(disparity_ori, cv2.COLORMAP_INFERNO)
            video_ori_disparity.write(disparity_ori_vis)

        video_ori_disparity.release()
        # video_norm_disparity.release()
        print("Done saving videos.")

        # --- 2. 保存为 .npz 文件 ---
        
        # 使用原始的 float 数组
        # disparities_ori_all 的形状是 [N, 1, H, W]
        # 压缩单通道维度, 变为 [N, H, W]
        disparities_squeezed = np.squeeze(disparities_ori_all, axis=1)
        
        np.savez(f"{args.out_dir}/{args.name}.npz", disp=disparities_squeezed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="stereoanyvideo", help="name to specify model")
    parser.add_argument('--ckpt', default="./checkpoints/StereoAnyVideo_MIX.pth", help="checkpoint of stereo model")
    
    # --- 新增/修改的参数 ---
    parser.add_argument('--left_file', type=str, help="path to left video file")
    parser.add_argument('--right_file', type=str, help="path to right video file")
    parser.add_argument('--top', type=int, default=0, help="pixels to crop from top")
    parser.add_argument('--bottom', type=int, default=0, help="pixels to crop from bottom")
    parser.add_argument(
        "--out_dir", default="output", type=str, help="output directory"
    )
    parser.add_argument("--name", default="name", type=str, help="output directory")
    parser.add_argument("--fps", type=int, default=30, help="frame rate for video visualization") # 重新添加
    
    # --- 保留的参数 ---
    parser.add_argument("--frame_size", type=int, default=150, help="number of frames in each forward pass batch.")
    parser.add_argument("--iters",type=int, default=20, help="number of updates in each forward pass.")
    parser.add_argument("--kernel_size", type=int, default=20, help="number of frames in each forward pass.")

    args = parser.parse_args()

    demo(args)