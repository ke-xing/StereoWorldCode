import os
import subprocess
import cv2
import torch
from multiprocessing import Pool, cpu_count
import numpy as np

# ---------------------------
# 基本路径配置
# ---------------------------
left_dir = "../split_movies/left"
rigth_dir = "../split_movies/right"
out_dir = "../split_movies/disp"

os.makedirs(out_dir, exist_ok=True)


# ---------------------------
# 黑边检测函数（原脚本有）
# ---------------------------
def detect_and_crop_black_bars(frame, threshold=5):
    """
    自动检测上下黑边并裁剪

    Args:
        frame: np.ndarray, 彩色图像 HxWxC
        threshold: int, 黑色像素阈值 (0~255)
    Returns:
        cropped_frame: np.ndarray, 裁剪后的图像
        top, bottom: int, 上下裁剪像素数量
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 取每行的平均亮度
    row_mean = np.mean(gray, axis=1)

    # 上边界
    top = 0
    for i, v in enumerate(row_mean):
        if v > threshold:
            top = i
            break

    # 下边界
    bottom = len(row_mean)
    for i, v in enumerate(row_mean[::-1]):
        if v > threshold:
            bottom = len(row_mean) - i
            break

    cropped_frame = frame[top:bottom, :, :]
    return cropped_frame, top, len(row_mean) - bottom


# ---------------------------
# 每个任务（单 GPU）执行函数
# ---------------------------
def process_video(gpu_id, fname):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    video_name = os.path.splitext(fname)[0]
    left_file = os.path.join(left_dir, fname)

    right_file = os.path.join(rigth_dir, fname)

    if not os.path.exists(right_file) or not os.path.exists(left_file):
        print(f"⚠️ 跳过 {fname}（pred 或 gt 文件缺失）")
        return

    cap = cv2.VideoCapture(right_file)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ 无法读取视频: {right_file}")
        return

    _, top, bottom = detect_and_crop_black_bars(frame)
    top, bottom = str(top), str(bottom)

    # 输出路径
    out_gt = out_dir

    print(f"🚀 GPU{gpu_id} 开始处理 {fname} ...")


    subprocess.run(
        [
            "python",
            "./demo_video.py",
            "--left_file",
            left_file,
            "--right_file",
            right_file,
            "--out_dir",
            out_gt,
            "--name",
            f"{video_name}",
            "--top",
            top,
            "--bottom",
            bottom
        ]
    )

    print(f"✅ GPU{gpu_id} 完成 {fname}")


# ---------------------------
# 主调度逻辑
# ---------------------------
if __name__ == "__main__":
    # 自动检测可用 GPU 数量
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("未检测到可用 GPU！")

    print(f"🔹 检测到 {ngpus} 张 GPU，可并行运行 {ngpus} 个任务。")

    # 获取待处理视频列表
    video_list = [f for f in os.listdir(rigth_dir) if f.endswith(".mp4")]
    video_list.sort()

    # 过滤掉已有输出的
    pending_videos = []
    for fname in video_list:

        pending_videos.append(fname)

    if not pending_videos:
        print("🎉 没有需要处理的视频。")
        exit(0)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    gpu_assignments = [(i % ngpus, fname) for i, fname in enumerate(pending_videos)]

    # 使用 concurrent.futures 处理
    with ProcessPoolExecutor(max_workers=ngpus) as executor:
        # 提交任务
        futures = [executor.submit(process_video, gpu, fname) for gpu, fname in gpu_assignments]

        # 可选：显示进度
        for future in as_completed(futures):
            try:
                result = future.result()  # 如果 process_video 有返回值，这里可以拿到
            except Exception as e:
                print(f"❌ 任务出错: {e}")

    print("\n🎉 所有任务处理完成！")
