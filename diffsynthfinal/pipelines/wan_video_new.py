import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import random
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import (
    WanTextEncoder,
    T5RelativeEmbedding,
    T5LayerNorm,
)
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from ..lora import GeneralLoRALoader
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 垂直方向融合（上下瓦片）- 5维版本
def blend_v(a, b, overlap_size):
    # 权重形状调整为: [1, 1, 1, overlap_size, 1]
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    # 融合重叠区域: 高度维度在第4维
    b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :, :overlap_size, :]
    return b


# 2. 水平方向融合（左右瓦片）- 5维版本
def blend_h(a, b, overlap_size):
    # 权重形状调整为: [1, 1, 1, 1, overlap_size]
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    # 融合重叠区域: 宽度维度在第5维
    b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :, :overlap_size]
    return b


class SILogLoss(nn.Module):
    def __init__(self, lam=0.5, eps=1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, pred, target, mask=None):
        B, C, T, H, W = pred.shape

        # 检查是否有负值，自动映射
        if torch.min(pred) < 0 or torch.min(target) < 0:
            pred = (pred + 1.0) / 2.0 + self.eps
            target = (target + 1.0) / 2.0 + self.eps
        else:
            pred = pred.clamp(min=self.eps)
            target = target.clamp(min=self.eps)

        # reshape到[B*C,1,T,H,W] 统一计算
        pred_flat = pred.view(B * C, 1, T, H, W)
        target_flat = target.view(B * C, 1, T, H, W)
        if mask is not None:
            mask_flat = mask.repeat(1, C, 1, 1, 1).view(B * C, 1, T, H, W)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]

        log_diff = torch.log(pred_flat) - torch.log(target_flat)
        n = log_diff.numel()
        loss = (log_diff**2).mean() - self.lam * (log_diff.sum() ** 2) / (n**2)

        return loss


# --------------------------
# 3D 卷积正则化网络 (保持特征维度)
# --------------------------
class CostVolume3DNet(nn.Module):
    def __init__(self, in_channels=32, base_channels=32, out_channels=16):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.conv3d_2 = nn.Conv3d(base_channels, base_channels, 3, padding=1)
        self.conv3d_3 = nn.Conv3d(base_channels, base_channels, 3, padding=1)
        self.conv3d_out = nn.Conv3d(
            base_channels, out_channels, 3, padding=1
        )  # 保留C通道
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.relu(self.conv3d_3(x))
        return self.conv3d_out(x)  # [B,C,D,H,W]


# --------------------------
# 可微 Stereo with 3D cost volume，支持时间维
# --------------------------
class DifferentiableStereo3D(nn.Module):
    def __init__(self, max_disp=64, feature_channels=16):
        super().__init__()
        self.max_disp = max_disp
        self.feature_channels = feature_channels
        self.cost_agg = CostVolume3DNet(
            in_channels=feature_channels * 2,
            base_channels=feature_channels,
            out_channels=feature_channels,
        )

    def normalize_disparity(self, disparity):
        # [-1,1] 归一化
        disp_norm = disparity / (self.max_disp - 1)
        disp_norm = disp_norm * 2.0 - 1.0
        return disp_norm

    def build_cost_volume(self, left_feat, right_feat):
        """
        left_feat, right_feat: [B, C, T, H, W]
        return: cost_volume [B, 2C, D, T, H, W]
        """
        B, C, T, H, W = left_feat.shape
        cost_volume = []

        for d in range(self.max_disp):
            if d > 0:
                shifted = F.pad(right_feat[..., d:], (0, d), mode="constant", value=0)
            else:
                shifted = right_feat

            concat = torch.cat([left_feat, shifted], dim=1)  # [B,2C,T,H,W]
            cost_volume.append(concat.unsqueeze(2))  # 插入 D 维度

        return torch.cat(cost_volume, dim=2)  # [B,2C,D,T,H,W]

    def forward(self, left, right):
        """
        left, right: [B, C, T, H, W]
        return: disparity [B, C, T, H, W]
        """
        B, C, T, H, W = left.shape

        # 构造代价体
        cost_volume = self.build_cost_volume(left, right)  # [B,2C,D,T,H,W]

        # 将时间维展开到 batch 维，Conv3d 处理
        cost_volume_flat = cost_volume.permute(0, 3, 1, 2, 4, 5).reshape(
            B * T, 2 * C, self.max_disp, H, W
        )
        cost_reg_flat = self.cost_agg(cost_volume_flat)  # [B*T,C,D,H,W]
        # reshape 回时间维
        cost_reg = cost_reg_flat.reshape(B, T, C, self.max_disp, H, W).permute(
            0, 2, 3, 1, 4, 5
        )  # [B,C,D,T,H,W]

        # soft-argmin 对 D 维
        prob = F.softmax(-cost_reg, dim=2)  # softmax over D
        disp_values = torch.arange(
            0, self.max_disp, device=left.device, dtype=left.dtype
        ).view(1, 1, self.max_disp, 1, 1, 1)
        disparity = torch.sum(prob * disp_values, dim=2)  # [B,C,T,H,W]

        # 归一化 [-1,1]
        disparity = self.normalize_disparity(disparity)

        return disparity  # [B,C,T,H,W]


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.stereo = DifferentiableStereo3D()
        self.silogloss = SILogLoss()
        self.units = [
            # WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
        )
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )

        tgt_latent_len = inputs["input_latents"].shape[2] // 2

        inputs["latents"] = self.scheduler.add_noise(
            inputs["input_latents"], inputs["noise"], timestep
        )

        inputs["latents_depth"] = self.scheduler.add_noise(
            inputs["input_latents_depth"], inputs["noise"], timestep
        )

        inputs["latents"][:, :, tgt_latent_len:, ...] = inputs["input_latents"][
            :, :, tgt_latent_len:, ...
        ]
        inputs["latents_depth"][:, :, tgt_latent_len:, ...] = inputs[
            "input_latents_depth"
        ][:, :, tgt_latent_len:, ...]

        ran = False

        if random.random() < 0.3:
            inputs["latents"][:, :, :3, ...] = inputs["input_latents"][:, :, :3, ...]
            inputs["latents_depth"][:, :, :3, ...] = inputs["input_latents_depth"][
                :, :, :3, ...
            ]
            ran = True

        training_target = self.scheduler.training_target(
            inputs["input_latents"], inputs["noise"], timestep
        )

        training_target_depth = self.scheduler.training_target(
            inputs["input_latents_depth"], inputs["noise"], timestep
        )

        noise_pred = self.model_fn(**inputs, mode="rgb", timestep=timestep)
        noise_pred_depth = self.model_fn(**inputs, mode="depth", timestep=timestep)

        x_0 = self.scheduler.predict_x0(
            inputs["latents"][:, :, :tgt_latent_len, ...],
            noise_pred[:, :, :tgt_latent_len, ...],
            timestep,
        )

        x_left = inputs["input_latents"][:, :, tgt_latent_len:, ...].clone()

        if ran:
            gt_disparity = inputs["input_latents_depth"][
                :, :, 3:tgt_latent_len, ...
            ].clone()

            d_min, d_max = gt_disparity.min(), gt_disparity.max()
            gt_disparity = 2.0 * (gt_disparity - d_min) / (d_max - d_min) - 1.0

            x_disparity = self.stereo(x_left, x_0)[:, :, 3:tgt_latent_len, ...]

            dis_loss = self.silogloss(x_disparity, gt_disparity)
        else:

            gt_disparity = inputs["input_latents_depth"][
                :, :, :tgt_latent_len, ...
            ].clone()

            d_min, d_max = gt_disparity.min(), gt_disparity.max()
            gt_disparity = 2.0 * (gt_disparity - d_min) / (d_max - d_min) - 1.0

            x_disparity = self.stereo(x_left, x_0)

            dis_loss = self.silogloss(x_disparity, gt_disparity)

        if ran:

            loss_rgb = torch.nn.functional.mse_loss(
                noise_pred[:, :, 3:tgt_latent_len, ...].float(),
                training_target[:, :, 3:tgt_latent_len, ...].float(),
            )
            loss_depth = torch.nn.functional.mse_loss(
                noise_pred_depth[:, :, 3:tgt_latent_len, ...].float(),
                training_target_depth[:, :, 3:tgt_latent_len, ...].float(),
            )

        else:

            loss_rgb = torch.nn.functional.mse_loss(
                noise_pred[:, :, :tgt_latent_len, ...].float(),
                training_target[:, :, :tgt_latent_len, ...].float(),
            )
            loss_depth = torch.nn.functional.mse_loss(
                noise_pred_depth[:, :, :tgt_latent_len, ...].float(),
                training_target_depth[:, :, :tgt_latent_len, ...].float(),
            )

        lambda_dis = 0.5
        lambda_l1 = 0.1

        dis_loss_total = dis_loss + lambda_l1 * F.l1_loss(x_disparity, gt_disparity)

        loss = (
            loss_rgb * self.scheduler.training_weight(timestep)
            + loss_depth * self.scheduler.training_weight(timestep)
            + lambda_dis * dis_loss_total
        )
        return (
            loss,
            loss_rgb,
            loss_depth,
            dis_loss,
            F.l1_loss(x_disparity, gt_disparity),
        )

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]

        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            pipe.initialize_usp()

        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model(
            "wan_video_motion_controller"
        )
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor

            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(
                audio_processor_config.path
            )
        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[
            Literal[
                "Left",
                "Right",
                "Up",
                "Down",
                "LeftUp",
                "LeftDown",
                "RightUp",
                "RightDown",
            ]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        long = False

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "long": long,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "input_video_r": None,
            "input_video_depth": None,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        tgt_latent_length = inputs_shared["input_latents"].shape[2]

        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            # Switch DiT if necessary
            if (
                timestep.item()
                < switch_DiT_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None
                and not models["dit"] is self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            inputs_shared["latents"] = torch.cat(
                [inputs_shared["latents"], inputs_shared["input_latents"]], dim=2
            )

            # Inference
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, mode="rgb", timestep=timestep
            )
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        mode="rgb",
                        timestep=timestep,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred[:, :, :tgt_latent_length, ...],
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"][:, :, :tgt_latent_length, ...],
            )

            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared[
                    "first_frame_latents"
                ]

        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        # Decode
        self.load_models_to_device(["vae"])

        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)

        self.load_models_to_device([])

        return video

    @torch.no_grad()
    def long_video(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[
            Literal[
                "Left",
                "Right",
                "Up",
                "Down",
                "LeftUp",
                "LeftDown",
                "RightUp",
                "RightDown",
            ]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        long = True

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "long": long,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "input_video_r": None,
            "input_video_depth": None,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        tgt_latent_length = inputs_shared["input_latents"][0].shape[2]

        latents_pre = None
        latents_depth_pre = None

        latents_all = []
        latents_all_decode = []
        latents_depth_all = []

        prompts = inputs_posi["context"]

        for i_big, input_latents in enumerate(tqdm(inputs_shared["input_latents"],desc='long video')):

            inputs_shared["latents"] = copy.deepcopy(inputs_shared["noise"])

            inputs_posi["context"] = prompts[i_big]

            for progress_id, timestep in enumerate(
                progress_bar_cmd(self.scheduler.timesteps)
            ):
                # Switch DiT if necessary
                if (
                    timestep.item()
                    < switch_DiT_boundary * self.scheduler.num_train_timesteps
                    and self.dit2 is not None
                    and not models["dit"] is self.dit2
                ):
                    self.load_models_to_device(self.in_iteration_models_2)
                    models["dit"] = self.dit2

                # Timestep
                timestep = timestep.unsqueeze(0).to(
                    dtype=self.torch_dtype, device=self.device
                )

                if i_big != 0:
                    overlap = copy.deepcopy(latents_all_decode[i_big - 1])
                # else:
                #     overlap= copy.deepcopy(input_latents[:, :, :3, :, :])

                    overlap = self.scheduler.add_noise(
                        overlap.to(self.device), inputs_shared["noise"][:, :, :3, :, :].to(self.device), timestep
                    )

                    inputs_shared["latents"][:, :, :3, :, :] = overlap

                inputs_shared["latents"] = torch.cat(
                    [inputs_shared["latents"], input_latents], dim=2
                )

                height = inputs_shared["latents"].shape[3]
                width = inputs_shared["latents"].shape[4]

                if height != 60 and width != 108:

                    tile_height, tile_width = 60, 108

                    # 纵向
                    if height <= tile_height:
                        tile_num_h = 1
                        tile_stride_h = tile_height
                    else:

                        tile_num_h = int(np.ceil(height / tile_height))

                        if height % tile_height == 0:
                            tile_num_h += 1

                        tile_stride_h = int(
                            np.floor((height - tile_height) / (tile_num_h - 1))
                        )
                    # 实际纵向重叠
                    tile_overlap_h = tile_height - tile_stride_h

                    # 横向
                    if width <= tile_width:
                        tile_num_w = 1
                        tile_stride_w = tile_width
                    else:
                        tile_num_w = int(np.ceil(width / tile_width))

                        if width % tile_width == 0:
                            tile_num_w += 1

                        tile_stride_w = int(
                            np.floor((width - tile_width) / (tile_num_w - 1))
                        )
                    # 实际横向重叠
                    tile_overlap_w = tile_width - tile_stride_w

                    tile_overlap = (tile_overlap_h, tile_overlap_w)

                    cols = []

                    i_bool = True
                    j_bool = True

                    last2tile_h = 0
                    last2tile_w = 0

                    for i in range(tile_num_h):
                        rows = []
                        for j in range(tile_num_w):
                            # 计算当前瓦片的起始位置
                            start_h = i * tile_stride_h
                            start_w = j * tile_stride_w

                            # 确保不超出图像边界
                            end_h = min(start_h + tile_height, height)
                            end_w = min(start_w + tile_width, width)

                            if i == tile_num_h - 1:

                                end_h = height
                                if (end_h - start_h) % 2 != 0:
                                    start_h -= 1
                                    i_bool = False

                                last2tile_h = end_h - start_h

                            if j == tile_num_w - 1:

                                end_w = width

                                if (end_w - start_w) % 2 != 0:
                                    start_w -= 1
                                    j_bool = False

                                last2tile_w = end_w - start_w

                            tile = inputs_shared["latents"][
                                :,
                                :,
                                :,
                                start_h:end_h,
                                start_w:end_w,
                            ].clone()

                            # Inference
                            noise_pred_posi = self.model_fn(
                                **models,
                                **inputs_shared,
                                **inputs_posi,
                                tile=tile,
                                mode="rgb",
                                timestep=timestep,
                            )
                            if cfg_scale != 1.0:
                                if cfg_merge:
                                    noise_pred_posi, noise_pred_nega = (
                                        noise_pred_posi.chunk(2, dim=0)
                                    )
                                else:
                                    noise_pred_nega = self.model_fn(
                                        **models,
                                        **inputs_shared,
                                        **inputs_nega,
                                        tile=tile,
                                        mode="rgb",
                                        timestep=timestep,
                                    )
                                noise_pred = noise_pred_nega + cfg_scale * (
                                    noise_pred_posi - noise_pred_nega
                                )
                            else:
                                noise_pred = noise_pred_posi

                            # Scheduler
                            tile = self.scheduler.step(
                                noise_pred[:, :, :tgt_latent_length, ...],
                                self.scheduler.timesteps[progress_id],
                                tile[:, :, :tgt_latent_length, ...],
                            )

                            tile_cpu = tile.clone().detach().cpu().to(torch.float32)       
                            tile_np = tile_cpu.numpy() 


                            rows.append(tile)
                        cols.append(rows)

                    # 创建融合后的结果矩阵
                    results_cols = []
                    for i, rows in enumerate(cols):
                        results_rows = []
                        for j, tile in enumerate(rows):
                            # 复制原始tile，避免修改原数据
                            processed_tile = tile.clone()

                            # 纵向融合（与上方tile融合）
                            if i > 0 and i != len(cols) - 1:
                                processed_tile = blend_v(
                                    results_cols[i - 1][j].clone(),
                                    processed_tile,
                                    tile_overlap[0],
                                )
                            elif i == len(cols) - 1:
                                over = tile_overlap[0] + 1

                                if i_bool:
                                    processed_tile = blend_v(
                                        results_cols[i - 1][j].clone(),
                                        processed_tile,
                                        tile_overlap[0],
                                    )
                                else:
                                    processed_tile = blend_v(
                                        results_cols[i - 1][j].clone(),
                                        processed_tile,
                                        over,
                                    )

                            # 横向融合（与左侧tile融合）
                            if j > 0 and j != len(rows) - 1:
                                processed_tile = blend_h(
                                    results_rows[j - 1].clone(), processed_tile, tile_overlap[1]
                                )
                            elif j == len(rows) - 1:

                                over = tile_overlap[1] + 1

                                if j_bool:
                                    processed_tile = blend_h(
                                        results_rows[j - 1].clone(),
                                        processed_tile,
                                        tile_overlap[1],
                                    )
                                else:
                                    processed_tile = blend_h(
                                        results_rows[j - 1].clone(), processed_tile, over
                                    )
                            results_rows.append(processed_tile)
                        results_cols.append(results_rows)

                    # 重新计算每个tile应该保留的尺寸
                    # print("开始尺寸裁剪...")
                    for i, rows in enumerate(results_cols):
                        for j, tile in enumerate(rows):
                            current_h, current_w = tile.shape[-2:]

                            # 计算应该保留的高度
                            if i == 0:  # 第一行
                                if len(results_cols) == 1:  # 只有一行
                                    keep_h = height
                                else:
                                    keep_h = tile_stride_h
                            elif i == len(results_cols) - 1:  # 最后一行
                                # 从当前行的起始位置到图像底部
                                keep_h = current_h
                            elif i == len(results_cols) - 2:
                                keep_h = (
                                    height
                                    - last2tile_h
                                    - tile_stride_h * (len(results_cols) - 2)
                                )

                            else:  # 中间行
                                keep_h = tile_stride_h

                            # 计算应该保留的宽度
                            if j == 0:  # 第一列
                                if len(rows) == 1:  # 只有一列
                                    keep_w = width
                                else:
                                    keep_w = tile_stride_w
                            elif j == len(rows) - 1:  # 最后一列
                                # 从当前列的起始位置到图像右边界
                                keep_w = current_w
                            elif j == len(rows) - 2:
                                keep_w = (
                                    width
                                    - last2tile_w
                                    - tile_stride_w * (len(rows) - 2)
                                )
                            else:  # 中间列
                                keep_w = tile_stride_w

                            # 确保不超出tile的实际尺寸
                            keep_h = min(keep_h, current_h)
                            keep_w = min(keep_w, current_w)


                            # 裁剪tile到正确尺寸
                            tile = tile[:, :, :, :keep_h, :keep_w]
                            rows[j] = tile

                    # 验证裁剪后的尺寸
                    # print("\n验证裁剪结果:")
                    for i, row_tiles in enumerate(results_cols):
                        row_widths = [tile.shape[-1] for tile in row_tiles]
                        total_width = sum(row_widths)
                        # print(f"第{i}行各tile宽度: {row_widths}, 总宽度: {total_width}")

                    for j in range(len(results_cols[0])):
                        col_heights = [
                            results_cols[i][j].shape[-2]
                            for i in range(len(results_cols))
                        ]
                        total_height = sum(col_heights)
                        # print(
                        #     f"第{j}列各tile高度: {col_heights}, 总高度: {total_height}"
                        # )

                    # 拼接宽度方向
                    results = []
                    for i, row_tiles in enumerate(results_cols):
                        try:
                            row_result = torch.cat(row_tiles, dim=4)
                            results.append(row_result)
                            # print(f"第{i}行拼接后形状: {row_result.shape[-2:]}")
                        except RuntimeError as e:
                            print(f"第{i}行横向拼接失败: {e}")
                            print(f"Tile shapes: {[tile.shape for tile in row_tiles]}")
                            raise

                    # 拼接高度方向
                    try:
                        inputs_shared["latents"] = torch.cat(results, dim=3)
                        # print(
                        #     f"所有行拼接后形状: {inputs_shared['latents'].shape[-2:]}"
                        # )
                    except RuntimeError as e:
                        print(f"纵向拼接失败: {e}")
                        print(f"Row shapes: {[row.shape for row in results]}")
                        raise

                    # 最后检查
                    final_shape = inputs_shared["latents"].shape[-2:]
                    expected_shape = (height, width)
                    assert (
                        final_shape == expected_shape
                    ), f"Latents shape mismatch: {final_shape} vs {expected_shape}"

                    # print(f"拼接成功: 最终形状 {final_shape}")
                else:

                    # Inference
                    noise_pred_posi = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_posi,
                        mode="rgb",
                        timestep=timestep,
                    )
                    if cfg_scale != 1.0:
                        if cfg_merge:
                            noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(
                                2, dim=0
                            )
                        else:
                            noise_pred_nega = self.model_fn(
                                **models,
                                **inputs_shared,
                                **inputs_nega,
                                mode="rgb",
                                timestep=timestep,
                            )
                        noise_pred = noise_pred_nega + cfg_scale * (
                            noise_pred_posi - noise_pred_nega
                        )
                    else:
                        noise_pred = noise_pred_posi

                    # Inference

                    # Scheduler
                    inputs_shared["latents"] = self.scheduler.step(
                        noise_pred[:, :, :tgt_latent_length, ...],
                        self.scheduler.timesteps[progress_id],
                        inputs_shared["latents"][:, :, :tgt_latent_length, ...],
                    )

                latents_pre = inputs_shared["latents"].clone()

            self.load_models_to_device(["vae"])

            latents_all.append(latents_pre)

            latents_pre_copy = copy.deepcopy(latents_pre)

            video_pre = self.vae.decode(
                latents_pre_copy,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[:, :, -9:, ...]

            video_pre = self.vae.encode(
                video_pre,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

            latents_all_decode.append(video_pre)

        # Decode
        self.load_models_to_device(["vae"])

        videos = []

        for index, rgb in enumerate(latents_all):

            # 送入解码器
            video = self.vae.decode(
                rgb,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            video = self.vae_output_to_video(video)
            videos.append(video)

            # 更新 prev_tail 为当前片段的最后两帧（丢弃前取）

        self.load_models_to_device([])

        return videos


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "input_video_r",
                "num_frames",
                "seed",
                "rand_device",
                "vace_reference_image",
            )
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        input_video_r,
        num_frames,
        seed,
        rand_device,
        vace_reference_image,
    ):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1

        if input_video_r is None:
            shape = (
                1,
                pipe.vae.model.z_dim,
                length,
                height // pipe.vae.upsampling_factor,
                width // pipe.vae.upsampling_factor,
            )
        else:
            shape = (
                input_video_r.shape[0],
                pipe.vae.model.z_dim,
                length * 2,
                height // pipe.vae.upsampling_factor,
                width // pipe.vae.upsampling_factor,
            )
        shape = tuple(int(x) for x in shape)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "input_video_r",
                "input_video_depth",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        input_video_r,
        input_video_depth,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
    ):
        if input_video_r is None and isinstance(input_video, list):

            encoded_videos = []
            for video in input_video:
                pipe.load_models_to_device(["vae"])
                # input_video = pipe.preprocess_video(input_video)
                input_latents = pipe.vae.encode(
                    video,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)

                encoded_videos.append(input_latents)

            return {
                "latents": noise,
                "input_latents": encoded_videos,
                "latents_depth": noise,
            }
        elif input_video_r is None and not isinstance(input_video, list):
            pipe.load_models_to_device(["vae"])
            # input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.vae.encode(
                input_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            if vace_reference_image is not None:
                # vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(
                    vace_reference_image, device=pipe.device
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                input_latents = torch.concat(
                    [vace_reference_latents, input_latents], dim=2
                )

            return {
                "latents": noise,
                "input_latents": input_latents,
                "latents_depth": noise,
            }

        pipe.load_models_to_device(["vae"])
        # input_video = pipe.preprocess_video(input_video)

        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        # input_video_r = pipe.preprocess_video(input_video_r)
        input_latents_r = pipe.vae.encode(
            input_video_r,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        input_latents_depth = pipe.vae.encode(
            input_video_depth,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        input_latents_depth = torch.cat([input_latents_depth, input_latents], dim=2)
        input_latents = torch.cat([input_latents_r, input_latents], dim=2)

        if pipe.scheduler.training:
            return {
                "latents": noise,
                "latents_depth": noise,
                "input_latents": input_latents,
                "input_latents_depth": input_latents_depth,
            }
        else:
            latents = pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "prompt": "prompt",
                "positive": "positive",
                "long": "long",
            },
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive, long=None) -> dict:

        if long:

            print(prompt)
            pipe.load_models_to_device(self.onload_model_names)

            prompt_embs = []
            for p in prompt:
                prompt_emb = pipe.prompter.encode_prompt(
                    p, positive=positive, device=pipe.device
                )

                prompt_embs.append(prompt_emb)
            return {"context": prompt_embs}

        else:
            pipe.load_models_to_device(self.onload_model_names)
            prompt_emb = pipe.prompter.encode_prompt(
                prompt, positive=positive, device=pipe.device
            )
            return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, pipe.image_encoder.encode_image([end_image])], dim=1
                )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if (
            input_image is None
            or pipe.image_encoder is None
            or not pipe.dit.require_clip_embedding
        ):
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, pipe.image_encoder.encode_image([end_image])], dim=1
                )
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(
            0, 1
        )
        z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, 0:1] = z
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": z,
        }


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "clip_feature",
                "y",
                "latents",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        clip_feature,
        y,
        latents,
    ):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(
            control_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros(
                (1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device
            )
            y = torch.zeros(
                (1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_control_direction",
                "camera_control_speed",
                "camera_control_origin",
                "latents",
                "input_image",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
        tiled,
        tile_size,
        tile_stride,
    ):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = (
            pipe.dit.control_adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin,
            )
        )

        control_camera_video = (
            camera_control_plucker_embedding[:num_frames]
            .permute([3, 0, 1, 2])
            .unsqueeze(0)
        )
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )
            y = pipe.vae.encode(
                [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "vace_video_mask",
                "vace_reference_image",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if (
            vace_video is not None
            or vace_video_mask is not None
            or vace_reference_image is not None
        ):
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=pipe.torch_dtype,
                    device=pipe.device,
                )
            else:
                vace_video = pipe.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(
                    vace_video_mask, min_value=0, max_value=1
                )

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(
                inactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(
                reactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(
                vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(
                    vace_reference_image,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                    dim=1,
                )
                vace_video_latents = torch.concat(
                    (vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents),
                    dim=2,
                )

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        num_inference_steps,
        tea_cache_l1_thresh,
        tea_cache_model_id,
    ):
        if tea_cache_l1_thresh is None:
            return {}
        return {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
        }


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_shared, tensor_shared), dim=0
                )
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=(
                "audio_encoder",
                "vae",
            ),
        )

    def process_audio(
        self,
        pipe: WanVideoPipeline,
        input_audio,
        audio_sample_rate,
        num_frames,
        fps=16,
        audio_embeds=None,
        return_all=False,
    ):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
            input_audio,
            audio_sample_rate,
            pipe.audio_processor,
            fps=fps,
            batch_frames=num_frames - 1,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        motion_video=None,
    ):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert (
                len(motion_video) == motion_frames
            ), f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros(
                [1, 3, motion_frames, height, width],
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(
            motion_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(
        self,
        pipe: WanVideoPipeline,
        s2v_pose_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        s2v_pose_latents=None,
        num_repeats=1,
        return_all=False,
    ):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[
            :, :, : infer_frames * num_repeats
        ]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat(
            [
                input_video,
                -torch.ones(
                    1,
                    3,
                    padding_frames,
                    height,
                    width,
                    device=input_video.device,
                    dtype=input_video.dtype,
                ),
            ],
            dim=2,
        )
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(
                cond,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:, :, 1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (
            (
                inputs_shared.get("input_audio") is None
                and inputs_shared.get("audio_embeds") is None
            )
            or pipe.audio_encoder is None
            or pipe.audio_processor is None
        ):
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = (
            inputs_shared.get("num_frames"),
            inputs_shared.get("height"),
            inputs_shared.get("width"),
            inputs_shared.get("tiled"),
            inputs_shared.get("tile_size"),
            inputs_shared.get("tile_stride"),
        )
        input_audio, audio_embeds, audio_sample_rate = (
            inputs_shared.pop("input_audio"),
            inputs_shared.pop("audio_embeds"),
            inputs_shared.get("audio_sample_rate"),
        )
        s2v_pose_video, s2v_pose_latents, motion_video = (
            inputs_shared.pop("s2v_pose_video"),
            inputs_shared.pop("s2v_pose_latents"),
            inputs_shared.pop("motion_video"),
        )

        audio_input_positive = self.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds
        )
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(
            self.process_motion_latents(
                pipe, height, width, tiled, tile_size, tile_stride, motion_video
            )
        )
        inputs_shared.update(
            self.process_pose_cond(
                pipe,
                s2v_pose_video,
                num_frames,
                height,
                width,
                tiled,
                tile_size,
                tile_stride,
                s2v_pose_latents=s2v_pose_latents,
            )
        )
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(
        pipe: WanVideoPipeline,
        input_audio=None,
        audio_sample_rate=16000,
        s2v_pose_video=None,
        num_frames=81,
        height=448,
        width=832,
        fps=16,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        assert (
            pipe.audio_encoder is not None and pipe.audio_processor is not None
        ), "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = (
            shapes["height"],
            shapes["width"],
            shapes["num_frames"],
        )
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True
        )
        pose_latents = unit.process_pose_cond(
            pipe,
            s2v_pose_video,
            num_frames,
            height,
            width,
            num_repeats=len(audio_embeds),
            return_all=True,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("latents", "motion_latents", "drop_motion_frames")
        )

    def process(
        self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames
    ):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:, :, 1:]], dim=2)
        return {"latents": latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x

        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + shift) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    latents_depth: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    tile=None,
    mode="rgb",
    **kwargs,
):
    if mode != "rgb":
        latents = latents_depth

    if tile != None:
        latents = tile

    vace_context = None
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        if (
            use_unified_sequence_parallel
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(
                    chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0
                )
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(
                    chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0
                )
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]

    if mode == "rgb":
        if tea_cache_update:
            x = tea_cache.update(x)
        else:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            for block_id, block in enumerate(dit.blocks):
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)

            if tea_cache is not None:
                tea_cache.store(x)

        x = dit.head(x, t)
    else:
        if tea_cache_update:
            x = tea_cache.update(x)
        else:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            for block_id, block in enumerate(dit.blocks):
                if block_id == 20:
                    break
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)

            for block_id, block in enumerate(dit.depth_blocks):

                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)

            if tea_cache is not None:
                tea_cache.store(x)

        x = dit.depth_head(x, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))

    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = (
        torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    )
    x, (f, h, w) = dit.patchify(
        dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents)
    )
    seq_len_x = seq_len_x_global = x.shape[
        1
    ]  # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = (
        torch.cat(
            [torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1
        )
        .to(torch.long)
        .to(x.device)
    )
    # freqs
    pre_compute_freqs = rope_precompute(
        x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads),
        grid_sizes,
        dit.freqs,
        start=None,
    )
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(
        x,
        pre_compute_freqs,
        mask,
        motion_latents,
        drop_motion_frames=drop_motion_frames,
        add_last_motion=2,
    )

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat(
        [timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)]
    )
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = (
        dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)
    )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        world_size, sp_rank = (
            get_sequence_parallel_world_size(),
            get_sequence_parallel_rank(),
        )
        assert (
            x.shape[1] % world_size == 0
        ), f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(
            torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy()
        )
        seq_len_x_list = [
            min(max(0, seq_len_x - seg_idxs[i]), x.shape[1])
            for i in range(len(seg_idxs) - 1)
        ]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    seq_len_x,
                    pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(
                        lambda x: dit.after_transformer_block(
                            block_id, x, audio_emb_global, merged_audio_emb, seq_len_x
                        )
                    ),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                context,
                t_mod,
                seq_len_x,
                pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(
                    lambda x: dit.after_transformer_block(
                        block_id, x, audio_emb_global, merged_audio_emb, seq_len_x
                    )
                ),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(
                block_id,
                x,
                audio_emb_global,
                merged_audio_emb,
                seq_len_x_global,
                use_unified_sequence_parallel,
            )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
