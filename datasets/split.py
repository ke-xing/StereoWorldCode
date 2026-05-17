#!/usr/bin/env python3
"""
高性能SBS 3D视频处理工具
自动检测并处理full-sbs/half-sbs视频，分离左右视图并分割成片段
修改版本：间隔一帧取一帧，丢弃不足81帧的片段
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from threading import Lock
import logging
from typing import Tuple, List, Optional
import argparse
from dataclasses import dataclass
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """视频处理配置"""

    input_path: str
    output_dir: str
    target_height: int = 480
    frames_per_segment: int = 81
    frame_interval: int = 2  # 新增：帧间隔，2表示每隔一帧取一帧
    max_workers: int = None


class SBSVideoProcessor:
    """SBS视频处理器"""

    def __init__(self, config: VideoConfig):
        self.config = config
        self.progress_lock = Lock()
        self.processed_files = 0
        self.total_files = 0

        # 支持的视频格式
        self.video_extensions = {
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
        }

        # 创建输出目录
        self.left_dir = Path(config.output_dir) / "left"
        self.right_dir = Path(config.output_dir) / "right"
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)

    def detect_sbs_type(self, frame: np.ndarray) -> str:
        """
        检测SBS类型 (full-sbs或half-sbs)
        通过分析帧的宽高比来判断
        """
        height, width = frame.shape[:2]
        aspect_ratio = width / height

        # 一般来说：
        # full-sbs: 宽高比约为 3.56:1 (1920x1080 -> 3840x1080)
        # half-sbs: 宽高比约为 1.78:1 (1920x1080 -> 1920x540)
        if aspect_ratio > 3.0:
            return "full-sbs"
        elif 1.5 < aspect_ratio < 2.5:
            return "half-sbs"
        else:
            logger.warning(
                f"无法确定SBS类型，宽高比: {aspect_ratio:.2f}，默认使用half-sbs"
            )
            return "half-sbs"

    def split_sbs_frame(
        self, frame: np.ndarray, sbs_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离SBS帧为左右视图
        """
        height, width = frame.shape[:2]

        if sbs_type == "full-sbs":
            # Full-SBS: 左右视图水平并排，各占一半宽度
            mid_width = width // 2
            left_frame = frame[:, :mid_width]
            right_frame = frame[:, mid_width:]
        else:  # half-sbs
            # Half-SBS: 左右视图被压缩到一半宽度
            mid_width = width // 2
            left_compressed = frame[:, :mid_width]
            right_compressed = frame[:, mid_width:]

            # 恢复原始宽高比
            target_width = mid_width * 2
            left_frame = cv2.resize(left_compressed, (target_width, height))
            right_frame = cv2.resize(right_compressed, (target_width, height))

        return left_frame, right_frame

    def resize_to_480p(self, frame: np.ndarray) -> np.ndarray:
        """
        将帧调整为480p，保持宽高比
        """
        height, width = frame.shape[:2]
        target_height = self.config.target_height

        # 确保宽度为偶数（视频编码要求）
        if target_width % 2 != 0:
            target_width += 1

        target_width=832

        pad_top = (1080 - height) // 2
        pad_bottom = 1080 - height - pad_top

        frame_padded = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        return cv2.resize(
            frame_padded, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )

    def process_video_segment(self, video_path: str, segment_info: dict) -> bool:
        """
        处理单个视频片段（间隔采样）
        """
        segment_name = segment_info["segment_name"]
        left_output_path = str(self.left_dir / f"{segment_name}.mp4")
        right_output_path = str(self.right_dir / f"{segment_name}.mp4")

        if os.path.exists(left_output_path) and os.path.exists(right_output_path):
            logger.info(f"视频片段已处理: {left_output_path}")
            return True

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return False

            # 获取视频信息
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 由于是间隔采样，输出视频的帧率需要调整
            output_fps = original_fps / self.config.frame_interval

            start_frame = segment_info["start_frame"]
            end_frame = min(segment_info["end_frame"], total_frames)
            expected_output_frames = segment_info["expected_output_frames"]

            # 检查预期输出帧数是否满足要求
            if expected_output_frames < self.config.frames_per_segment:
                logger.info(
                    f"片段 {segment_name} 帧数不足 ({expected_output_frames} < {self.config.frames_per_segment})，跳过处理"
                )
                return False

            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 读取第一帧来检测SBS类型
            ret, first_frame = cap.read()
            if not ret:
                logger.error(f"无法读取视频帧: {video_path}")
                return False

            sbs_type = self.detect_sbs_type(first_frame)
            logger.info(f"检测到 {sbs_type} 格式")
            with open("3d_format1.txt", "a", encoding="utf-8") as f:
                f.write(f"{os.path.basename(video_path)}: {sbs_type}\n")

            # 分离第一帧获取尺寸信息
            left_frame, right_frame = self.split_sbs_frame(first_frame, sbs_type)
            left_resized = self.resize_to_480p(left_frame)
            right_resized = self.resize_to_480p(right_frame)

            height, width = left_resized.shape[:2]

            # 设置视频编解码器 (使用H.264以获得更好的性能和兼容性)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # 创建左右视图的视频写入器
            left_writer = cv2.VideoWriter(
                left_output_path, fourcc, output_fps, (width, height)
            )
            right_writer = cv2.VideoWriter(
                right_output_path, fourcc, output_fps, (width, height)
            )

            if not left_writer.isOpened() or not right_writer.isOpened():
                logger.error(f"无法创建输出视频文件: {segment_name}")
                return False

            # 处理第一帧
            left_writer.write(left_resized)
            right_writer.write(right_resized)

            frame_count = 1
            current_frame = start_frame

            # 间隔采样处理剩余帧
            while (
                frame_count < expected_output_frames
                and current_frame < end_frame - self.config.frame_interval
            ):
                # 跳转到下一个采样帧
                current_frame += self.config.frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"在帧 {current_frame} 处读取失败，提前结束")
                    break

                # 分离左右视图
                left_frame, right_frame = self.split_sbs_frame(frame, sbs_type)

                # 调整尺寸
                left_resized = self.resize_to_480p(left_frame)
                right_resized = self.resize_to_480p(right_frame)

                # 写入视频
                left_writer.write(left_resized)
                right_writer.write(right_resized)

                frame_count += 1

                # 内存优化：每处理一定数量的帧后强制垃圾回收
                if frame_count % 50 == 0:
                    import gc

                    gc.collect()

            # 释放资源
            cap.release()
            left_writer.release()
            right_writer.release()

            # 检查实际输出帧数是否满足要求
            if frame_count < self.config.frames_per_segment:
                logger.warning(
                    f"片段 {segment_name} 实际输出帧数不足 ({frame_count} < {self.config.frames_per_segment})，删除输出文件"
                )
                # 删除不符合要求的文件
                try:
                    os.remove(left_output_path)
                    os.remove(right_output_path)
                except:
                    pass
                return False

            logger.info(f"成功处理片段: {segment_name} ({frame_count} 帧，间隔采样)")
            return True

        except Exception as e:
            logger.error(
                f"处理片段 {segment_info.get('segment_name', 'unknown')} 时出错: {str(e)}"
            )
            return False

    def create_video_segments_info(self, video_path: str) -> List[dict]:
        """
        创建视频片段信息列表（考虑间隔采样）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        segments = []
        video_name = Path(video_path).stem

        # 计算每个片段需要的原始帧数（考虑间隔采样）
        frames_needed_per_segment = (
            self.config.frames_per_segment * self.config.frame_interval
        )

        segment_index = 0
        for start_frame in range(0, total_frames, frames_needed_per_segment):
            end_frame = min(start_frame + frames_needed_per_segment, total_frames)

            # 计算这个片段经过间隔采样后的预期输出帧数
            available_frames = end_frame - start_frame
            expected_output_frames = available_frames // self.config.frame_interval

            # 只有预期输出帧数满足要求才创建片段
            if expected_output_frames >= self.config.frames_per_segment:
                segment_name = f"{video_name}_segment_{segment_index:04d}"

                segments.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "segment_name": segment_name,
                        "video_path": video_path,
                        "expected_output_frames": expected_output_frames,
                    }
                )

                segment_index += 1
            else:
                logger.info(
                    f"跳过帧数不足的片段: 起始帧 {start_frame}，预期输出 {expected_output_frames} 帧"
                )

        logger.info(
            f"间隔采样 (每{self.config.frame_interval}帧取1帧) 后，有效片段数: {len(segments)}"
        )
        return segments

    def process_single_video(self, video_path: str) -> bool:
        """
        处理单个视频文件
        """
        try:
            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"采样设置: 每 {self.config.frame_interval} 帧取 1 帧")

            # 创建片段信息
            segments_info = self.create_video_segments_info(video_path)
            if not segments_info:
                logger.warning(f"视频 {video_path} 没有满足条件的片段")
                return False

            logger.info(f"视频将被分割为 {len(segments_info)} 个有效片段")

            # 使用线程池并行处理片段
            max_workers = self.config.max_workers or min(mp.cpu_count(), 4)
            successful_segments = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 为每个片段提交处理任务
                future_to_segment = {}
                for segment_info in segments_info:
                    future = executor.submit(
                        self.process_video_segment, video_path, segment_info
                    )
                    future_to_segment[future] = segment_info

                # 收集结果
                for future in future_to_segment:
                    try:
                        if future.result():
                            successful_segments += 1
                    except Exception as e:
                        segment_name = future_to_segment[future].get(
                            "segment_name", "unknown"
                        )
                        logger.error(f"处理片段 {segment_name} 失败: {str(e)}")

            if len(segments_info) > 0:
                success_rate = successful_segments / len(segments_info)
                logger.info(
                    f"视频 {Path(video_path).name} 处理完成: {successful_segments}/{len(segments_info)} 片段成功 ({success_rate:.1%})"
                )
            else:
                success_rate = 0
                logger.info(f"视频 {Path(video_path).name} 没有满足条件的片段")

            # 更新进度
            with self.progress_lock:
                self.processed_files += 1
                logger.info(f"总体进度: {self.processed_files}/{self.total_files}")

            return successful_segments > 0

        except Exception as e:
            logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
            return False

    def find_video_files(self) -> List[str]:
        """
        查找输入路径中的所有视频文件
        """
        input_path = Path(self.config.input_path)
        video_files = []

        if input_path.is_file():
            if input_path.suffix.lower() in self.video_extensions:
                video_files.append(str(input_path))
        elif input_path.is_dir():
            for ext in self.video_extensions:
                video_files.extend(input_path.glob(f"*{ext}"))
                video_files.extend(input_path.glob(f"*{ext.upper()}"))
            video_files = [str(f) for f in video_files]

        return sorted(video_files)

    def process_all_videos(self) -> bool:
        """
        处理所有视频文件
        """
        video_files = self.find_video_files()
        if not video_files:
            logger.error(f"在路径 {self.config.input_path} 中未找到支持的视频文件")
            return False

        self.total_files = len(video_files)
        logger.info(f"找到 {self.total_files} 个视频文件")

        start_time = time.time()
        successful_videos = 0

        # 使用进程池处理多个视频文件（如果视频数量较多）
        if len(video_files) > 1 and (self.config.max_workers or mp.cpu_count()) > 2:
            # 为了避免资源竞争，限制并行处理的视频数量
            max_video_workers = min(2, mp.cpu_count() // 2)

            with ThreadPoolExecutor(max_workers=max_video_workers) as executor:
                futures = [
                    executor.submit(self.process_single_video, video_path)
                    for video_path in video_files
                ]

                for future in futures:
                    try:
                        if future.result():
                            successful_videos += 1
                    except Exception as e:
                        logger.error(f"视频处理失败: {str(e)}")
        else:
            # 顺序处理视频文件
            for video_path in video_files:
                if self.process_single_video(video_path):
                    successful_videos += 1

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"处理完成！")
        logger.info(f"成功处理: {successful_videos}/{self.total_files} 个视频文件")
        logger.info(f"总耗时: {processing_time:.2f} 秒")
        logger.info(f"输出目录: {self.config.output_dir}")

        return successful_videos > 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="高性能SBS 3D视频处理工具 (间隔采样版本)"
    )
    parser.add_argument("--input_path" , default="./raw_movies", help="输入视频文件或文件夹路径")
    parser.add_argument(
        "-o",
        "--output",
        default="./split_movies",
        help="输出目录 (默认: ./split_movies)",
    )
    parser.add_argument(
        "-he", "--height", type=int, default=480, help="目标高度 (默认: 480)"
    )
    parser.add_argument(
        "-f", "--frames", type=int, default=81, help="每片段帧数 (默认: 81)"
    )
    parser.add_argument(
        "-i", "--interval", type=int, default=2, help="帧间隔 (默认: 2, 即每隔1帧取1帧)"
    )
    parser.add_argument("-w", "--workers", type=int, help="最大工作线程数 (默认: 自动)")

    args = parser.parse_args()

    # 验证输入路径
    if not os.path.exists(args.input_path):
        logger.error(f"输入路径不存在: {args.input_path}")
        sys.exit(1)

    # 验证帧间隔参数
    if args.interval < 1:
        logger.error(f"帧间隔必须大于0: {args.interval}")
        sys.exit(1)

    # 创建配置
    config = VideoConfig(
        input_path=args.input_path,
        output_dir=args.output,
        target_height=args.height,
        frames_per_segment=args.frames,
        frame_interval=args.interval,
        max_workers=args.workers,
    )

    # 创建处理器并开始处理
    processor = SBSVideoProcessor(config)

    try:
        success = processor.process_all_videos()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
