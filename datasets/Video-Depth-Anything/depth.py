# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import argparse
import numpy as np
import os
import torch
import glob
import multiprocessing as mp
from multiprocessing import Queue, Process
import time
from pathlib import Path
import logging

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_worker_logging(worker_id, gpu_id):
    """为每个工作进程设置独立的日志"""
    logger = logging.getLogger(f'Worker-{worker_id}-GPU-{gpu_id}')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f'[Worker-{worker_id}-GPU-{gpu_id}] %(asctime)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_video_files(input_dir):
    """获取文件夹中所有支持的视频文件，若存在同名 .npz 则排除"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    filtered_videos = []
    for video_path in video_files:
        base, _ = os.path.splitext(video_path)
        npz_path = base + '.npz'
        if not os.path.exists(npz_path):  # 若没有同名 npz 才保留
            filtered_videos.append(video_path)

    return sorted(filtered_videos)


def process_single_video(video_path, args, device, model, worker_logger):
    """处理单个视频文件"""
    try:
        worker_logger.info(f"开始处理视频: {video_path}")
        
        # 读取视频帧
        frames, target_fps = read_video_frames(video_path, args.max_len, args.target_fps, args.max_res)
        
        # 预测深度
        depths, fps = model.infer_video_depth(frames, target_fps, input_size=args.input_size, device=device, fp32=args.fp32)
        
        # 准备输出路径
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        # 创建输出目录
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存处理后的视频
        processed_video_path = os.path.join(args.output_dir, f'{base_name}_src.mp4')
        depth_vis_path = os.path.join(args.output_dir, f'{base_name}_depth_vis.mp4')
        
        # save_video(frames, processed_video_path, fps=fps)
        # save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
        
        # # 保存深度数据
        # if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, f'{base_name}.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
        
        # if args.save_exr:
        #     depth_exr_dir = os.path.join(args.output_dir, f'{base_name}_depths_exr')
        #     os.makedirs(depth_exr_dir, exist_ok=True)
        #     try:
        #         import OpenEXR
        #         import Imath
        #         for i, depth in enumerate(depths):
        #             output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
        #             header = OpenEXR.Header(depth.shape[1], depth.shape[0])
        #             header["channels"] = {
        #                 "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        #             }
        #             exr_file = OpenEXR.OutputFile(output_exr, header)
        #             exr_file.writePixels({"Z": depth.tobytes()})
        #             exr_file.close()
        #     except ImportError:
        #         worker_logger.warning("OpenEXR 未安装，跳过EXR格式保存")
        
        worker_logger.info(f"完成处理视频: {video_path}")
        return True, video_path, None
        
    except Exception as e:
        worker_logger.error(f"处理视频失败 {video_path}: {str(e)}")
        return False, video_path, str(e)

def worker_process(worker_id, gpu_id, video_queue, result_queue, args):
    """工作进程函数"""
    # 设置工作进程日志
    worker_logger = setup_worker_logging(worker_id, gpu_id)
    
    try:
        # 设置CUDA设备
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        
        worker_logger.info(f"启动工作进程，使用设备: {device}")
        
        # 加载模型配置
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        # 初始化模型
        video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
        checkpoint_path = f'./checkpoints/video_depth_anything_{args.encoder}.pth'
        
        if not os.path.exists(checkpoint_path):
            worker_logger.error(f"找不到模型文件: {checkpoint_path}")
            return
        
        video_depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()
        
        worker_logger.info("模型加载完成")
        
        # 处理视频
        processed_count = 0
        while True:
            try:
                # 从队列获取视频路径，设置超时避免死锁
                video_path = video_queue.get(timeout=10)
                if video_path is None:  # 结束信号
                    break
                
                # 处理视频
                success, path, error = process_single_video(video_path, args, device, video_depth_anything, worker_logger)
                result_queue.put((success, path, error, worker_id))
                processed_count += 1
                
            except Exception as e:
                if "Empty" in str(e):  # 队列为空
                    worker_logger.info("队列为空，工作进程结束")
                    break
                else:
                    worker_logger.error(f"工作进程出错: {str(e)}")
        
        worker_logger.info(f"工作进程结束，共处理 {processed_count} 个视频")
        
    except Exception as e:
        worker_logger.error(f"工作进程初始化失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Video Depth Anything')
    parser.add_argument('--input_dir', type=str, default='../split_movies/right', help='输入视频文件夹路径')
    parser.add_argument('--output_dir', type=str, default='../split_movies/depth', help='输出目录')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='最大视频长度，-1表示无限制')
    parser.add_argument('--target_fps', type=int, default=-1, help='目标FPS，-1表示使用原始FPS')
    parser.add_argument('--fp32', action='store_true', help='使用float32精度，默认为float16')
    parser.add_argument('--grayscale', action='store_true', help='不使用彩色调色板')
    parser.add_argument('--save_npz', action='store_true', help='保存深度为npz格式')
    parser.add_argument('--save_exr', action='store_true', help='保存深度为exr格式')
    parser.add_argument('--gpu_ids', type=str, default='0', help='要使用的GPU ID，用逗号分隔')
    parser.add_argument('--processes_per_gpu', type=int, default=1, help='每个GPU运行的进程数')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，请检查GPU环境")
        return
    
    # 解析GPU ID
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    available_gpus = list(range(torch.cuda.device_count()))
    
    # 验证GPU ID
    for gpu_id in gpu_ids:
        if gpu_id not in available_gpus:
            logger.error(f"GPU {gpu_id} 不可用，可用的GPU: {available_gpus}")
            return
    
    logger.info(f"使用GPU: {gpu_ids}")
    logger.info(f"每个GPU运行 {args.processes_per_gpu} 个进程")
    
    # 获取所有视频文件
    video_files = get_video_files(args.input_dir)
    if not video_files:
        logger.error(f"在 {args.input_dir} 中未找到视频文件")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建队列
    video_queue = Queue()
    result_queue = Queue()
    
    # 将视频文件放入队列
    for video_file in video_files:
        video_queue.put(video_file)
    
    # 创建工作进程
    processes = []
    worker_id = 0
    
    for gpu_id in gpu_ids:
        for _ in range(args.processes_per_gpu):
            p = Process(target=worker_process, args=(worker_id, gpu_id, video_queue, result_queue, args))
            p.start()
            processes.append(p)
            worker_id += 1
    
    # 添加结束信号
    for _ in processes:
        video_queue.put(None)
    
    logger.info(f"启动了 {len(processes)} 个工作进程")
    
    # 监控处理进度
    start_time = time.time()
    completed = 0
    failed = 0
    
    try:
        while completed + failed < len(video_files):
            try:
                success, path, error, worker_id = result_queue.get(timeout=60)
                if success:
                    completed += 1
                    logger.info(f"完成 ({completed}/{len(video_files)}): {os.path.basename(path)}")
                else:
                    failed += 1
                    logger.error(f"失败 ({failed}/{len(video_files)}): {os.path.basename(path)} - {error}")
                
                # 显示进度
                elapsed = time.time() - start_time
                progress = (completed + failed) / len(video_files)
                if progress > 0:
                    eta = elapsed / progress - elapsed
                    logger.info(f"进度: {progress*100:.1f}% - 已用时间: {elapsed/60:.1f}分钟 - 预计剩余: {eta/60:.1f}分钟")
                
            except Exception as e:
                if "Empty" in str(e):
                    logger.warning("等待结果超时，检查进程状态...")
                    continue
                else:
                    logger.error(f"获取结果时出错: {str(e)}")
                    break
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止所有进程...")
    
    # 等待所有进程结束
    for p in processes:
        p.terminate()
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"强制结束进程 {p.pid}")
            p.kill()
    
    # 最终统计
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"处理完成!")
    logger.info(f"总视频数: {len(video_files)}")
    logger.info(f"成功处理: {completed}")
    logger.info(f"处理失败: {failed}")
    logger.info(f"总用时: {total_time/60:.1f} 分钟")
    if completed > 0:
        logger.info(f"平均每个视频: {total_time/completed:.1f} 秒")
    logger.info("="*60)

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()