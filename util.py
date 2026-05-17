import torch
from PIL import Image
from diffsynthfinal import save_video
import torch, torchvision, imageio, os
from PIL import Image
import numpy as np
from einops import repeat
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2
import tempfile


def preprocess_image(
    image,
    torch_dtype=None,
    device=None,
    pattern="B C H W",
    min_value=-1,
    max_value=1,
):
    # Transform a PIL.Image to torch.Tensor
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype or torch.bfloat16, device=device or "cuda")
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image


def preprocess_video(
    video,
    torch_dtype=None,
    device=None,
    pattern="B C T H W",
    min_value=-1,
    max_value=1,
):
    # Transform a list of PIL.Image to torch.Tensor
    video = [
        preprocess_image(
            image,
            torch_dtype=torch_dtype,
            device=device,
            min_value=min_value,
            max_value=max_value,
        )
        for image in video
    ]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


def init_caption_model(model, device):
    """初始化视频描述模型"""
    model_path = f"./models/VideoLLaMA3-7B"
    print(f"Loading caption model on device: {device}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else None,
    )

    return model, processor, device


def save_frames_as_video(frames, temp_video_path, fps=30):
    """将帧列表保存为临时视频文件，用于caption"""
    if not frames:
        return False

    # 获取第一帧的尺寸
    first_frame = frames[0]
    height, width = np.array(first_frame).shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    for frame in frames:
        # PIL Image转换为OpenCV格式
        frame_np = np.array(frame)
        if frame_np.shape[2] == 3:  # RGB
            frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_cv = frame_np
        out.write(frame_cv)

    out.release()
    return True


def caption_video_segment(
    frames, caption_model, caption_processor, caption_device, segment_idx, fps
):
    """为视频片段生成描述"""
    try:
        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        # 保存帧为临时视频
        if not save_frames_as_video(frames, temp_path):
            return f"Unable to process segment {segment_idx}"

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": temp_path,
                            "fps": fps,
                            "max_frames": 180,
                        },
                    },
                    {"type": "text", "text": "Describe the video."},
                ],
            },
        ]

        inputs = caption_processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(caption_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = caption_model.generate(**inputs, max_new_tokens=1024)
        response = caption_processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        # 清理临时文件
        os.unlink(temp_path)

        return response

    except Exception as e:
        print(f"Error captioning segment {segment_idx}: {str(e)}")
        return f"Video segment {segment_idx} processing failed"


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = (
            [] if operators is None else operators
        )

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        overlap_frames=9,  # 新增：重叠帧数
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        enable_long_video=True,
    ):
        self.num_frames = num_frames
        self.overlap_frames = overlap_frames  # 重叠帧数
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.enable_long_video = enable_long_video

    def get_num_frames(self, reader):
        total_frames = int(reader.count_frames())
        # if not self.enable_long_video or total_frames <= self.num_frames:
        #     num_frames = self.num_frames
        #     if total_frames < num_frames:
        #         num_frames = total_frames
        #         while (
        #             num_frames > 1
        #             and num_frames % self.time_division_factor
        #             != self.time_division_remainder
        #         ):
        #             num_frames -= 1
        #     return num_frames
        # else:
        return total_frames

    def __call__(self, data: str):
        reader = imageio.get_reader(data, format="ffmpeg")
        # total_frames = self.get_num_frames(reader)

       

        total_frames=int(cv2.VideoCapture(data).get(cv2.CAP_PROP_FRAME_COUNT))


        # 长视频处理：带重叠的分片加载
        all_segments = []
        step_size = self.num_frames - self.overlap_frames  # 实际步长

        for start_frame in range(0, total_frames, step_size):
            end_frame = min(start_frame + self.num_frames, total_frames)
            segment_frames = []

            for frame_id in range(start_frame, end_frame):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                segment_frames.append(frame)


            frame_count = len(segment_frames)
            if frame_count < self.num_frames:  # 片段不足81帧才考虑补帧
                next_target=self.num_frames
                last_frame = segment_frames[-1]
                while len(segment_frames) < next_target:
                    segment_frames.append(last_frame.copy())

            

            if len(segment_frames) > 0:
                all_segments.append(
                    {
                        "frames": segment_frames,
                        "start_frame": start_frame,
                        "end_frame": start_frame + len(segment_frames) - 1,
                        "overlap_start": max(0, start_frame),
                        "overlap_end": start_frame + len(segment_frames) - 1,
                    }
                )

            # 如果已经到达视频末尾，停止分片
            if end_frame >= total_frames:
                break

        reader.close()
        return all_segments


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class ImageCropAndResize(DataProcessingOperator):
    def __init__(
        self,
        height,
        width,
        max_pixels,
        height_division_factor,
        width_division_factor,
        force_resolution=True,
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.force_resolution = force_resolution

    def crop_and_resize(self, image, target_height, target_width):
        original_width, original_height = image.size
        scale = max(target_width / original_width, target_height / original_height)
        new_height = round(original_height * scale)
        new_width = round(original_width * scale)

        image = torchvision.transforms.functional.resize(
            image,
            (new_height, new_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )

        return image

    def get_height_width(self, image):
        if self.force_resolution and self.height is not None and self.width is not None:
            return self.height, self.width
        elif self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
            return height, width
        else:
            return self.height, self.width

    def __call__(self, data: Image.Image):
        target_height, target_width = self.get_height_width(data)
        original_size = data.size

        if self.force_resolution and (
            original_size[1] != target_height or original_size[0] != target_width
        ):
            print(
                f"Adjusting resolution from {original_size[0]}x{original_size[1]} to {target_width}x{target_height}"
            )

        image = self.crop_and_resize(data, target_height, target_width)
        return image


def default_video_operator(
    base_path="",
    max_pixels=1920 * 1080,
    height=480,
    width=832,
    height_division_factor=16,
    width_division_factor=16,
    num_frames=81,
    overlap_frames=9,  # 新增重叠帧参数
    time_division_factor=4,
    time_division_remainder=1,
    enable_long_video=True,
    force_resolution=True,
):
    return RouteByType(
        operator_map=[
            (
                str,
                ToAbsolutePath(base_path)
                >> RouteByExtensionName(
                    operator_map=[
                        (
                            ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                            LoadVideo(
                                num_frames,
                                overlap_frames,  # 传递重叠帧参数
                                time_division_factor,
                                time_division_remainder,
                                frame_processor=ImageCropAndResize(
                                    height,
                                    width,
                                    max_pixels,
                                    height_division_factor,
                                    width_division_factor,
                                    force_resolution=force_resolution,
                                ),
                                enable_long_video=enable_long_video,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )


def blend_video_segments(segments_data, overlap_frames=9):
    """
    混合重叠的视频片段

    Args:
        segments_data: list of dict, 每个元素包含 {'video': VideoData, 'start_frame': int, 'end_frame': int}
        overlap_frames: 重叠帧数
        blend_method: 混合方法 ("cross_fade", "simple_average", "weighted_average")

    Returns:
        VideoData: 混合后的完整视频
    """
    if len(segments_data) == 1:
        return segments_data[0]["video"]

    # 提取所有片段的帧
    all_frames = []

    for i, segment_info in enumerate(segments_data):
        segment_video = segment_info["video"]

        frames = segment_video

        frames_list = frames

        if i == 0:
            # 第一个片段：添加所有帧
            all_frames.extend(frames_list)
        else:
            non_overlap_frames = frames_list[overlap_frames:]
            all_frames.extend(non_overlap_frames)

    return all_frames


def concat_videos_with_blending(
    video_segments,
    segment_info_list,
    video_path,
    output_path,
    fps=24,
    overlap_frames=9,
):
    """
    拼接视频片段并在重叠区域进行混合
    """
    if not video_segments:
        return

    # 准备片段数据
    segments_data = []
    for i, video in enumerate(video_segments):
        segments_data.append(
            {
                "video": video,
                "start_frame": segment_info_list[i]["start_frame"],
                "end_frame": segment_info_list[i]["end_frame"],
            }
        )

    # 进行混合
    blended_frames = blend_video_segments(segments_data, overlap_frames)

    reader = imageio.get_reader(video_path)

    # 获取右视频帧的宽高
    first_right = blended_frames[0]
    if not isinstance(first_right, Image.Image):
        raise TypeError("blended_frames 中的元素必须是 PIL.Image.Image 对象")
    target_w, target_h = first_right.size  # PIL 是 (width, height)

    sbs_frames = []
    frame_idx = 0

    for left_frame in reader:
        # if frame_idx >= len(blended_frames):
        #     break

        # 左帧转为 PIL 并缩放
        left_pil = Image.fromarray(left_frame)
        left_pil = left_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 右帧（确保为 RGB）
        right_pil = blended_frames[frame_idx].convert("RGB")

        # 拼接SBS
        sbs_img = Image.new("RGB", (target_w * 2, target_h))
        sbs_img.paste(left_pil, (0, 0))
        sbs_img.paste(right_pil, (target_w, 0))

        sbs_frames.append(sbs_img)
        frame_idx += 1

    reader.close()

    # 创建最终视频并保存
    if blended_frames:
        save_video(sbs_frames, output_path, fps=fps, quality=6)
