import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers import Qwen2Tokenizer
from pipelines.trainable_pipeline import TrainablePipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    align_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
    audio_latent_num_frames,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from typing import Optional
from loguru import logger

from pipelines.schedulers.minimax_h3_scheduler import MiniMaxH3Scheduler


class MiniMaxH3_T2VA(TrainablePipeline):
    r"""
    MiniMaxH3 Trainable Pipeline for T2VA workflow with bidirectional generation.
    """

    def __init__(
        self,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen3VLProcessor,
        vae,
        transformer: MiniMaxH3Transformer3DModel,
        video_scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        image_processor: VaeImageProcessor,
        video_processor: VideoProcessor,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler
        self.image_processor = image_processor
        self.video_processor = video_processor

    @property
    def vae_frames_per_chunk(self) -> int:
        return 17

    @property
    def vae_latents_per_chunk(self) -> int:
        return 5

    @property
    def vae_spatial_compression_ratio(self) -> int:
        return 16

    @property
    def vae_latent_channels(self) -> int:
        return 24

    @property
    def audio_latent_channels(self) -> int:
        return 32

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return (1, 2, 3)

    @property
    def min_duration(self) -> float:
        return 5.0

    @property
    def max_duration(self) -> float:
        return 15.0

    @property
    def keyframe_noise_aug(self) -> float:
        return 0.999

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        r"""
        Load pretrained weights from `pretrained_model_name_or_path` and initialize
        the pipeline modules, converting to the target `weight_dtype` and moving to
        target `device` if required.
        """
        pass

    def prepare_prompt_embeds(
        self,
        prompt: Optional[list[str] | str] = None,
        vision: Optional[dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Encode the prompt with Qwen3VL encoder, extract the unnormalized `hidden_states`
        from the 50-th layer.

        Args:
            prompt (str|list[str], optional): The text sequence or batched.
            vision (dict[str, Tensor], optional): The vision inputs including
                `pixel_values` and `image_grid_thw` for images inputs,
                `pixel_values_videos` and `video_grid_thw` for videos.
            device (torch.device, optional): The device to run the conditioner on.
            dtype (torch.dtype, optional): The dtype of the returned embeddings.

        Returns:
            prompt_embeds (Tensor): The last hidden_states without normalization from text_encoder with
                shape [Batch, num_text_tokens, 5120].
            text_token_tags (Tensor): The text token tags defined by MiniMax-H3.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        token_ids = self.tokenizer(prompt, add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]),
            dtype=torch.long,
            device=device,
        )
        vision_kwargs = {}
        for name, value in (vision or {}).items():
            vision_kwargs[name] = (
                value.to(device, self.text_encoder.dtype)
                if name.startswith("pixel_")
                else value.to(device)
            )

        # For MiniMax-H3, forward the language model and extract the last layer hidden_states
        # do not use the language model head.
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        )
        prompt_embeds = outputs.hidden_states[50].to(device, dtype=dtype)
        text_token_tags = torch.full(
            (len(token_ids),), MINIMAX_H3_TEXT_TAG, dtype=torch.long
        )
        return {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": text_token_tags,
        }

    def prepare_input_layouts(
        self,
        num_frames: int,
        height: Optional[int],
        width: Optional[int],
        text_token_tags: torch.Tensor,
        short_edge: int = 768,
        max_pixels: int = 768 * 1344,
        aspect_height: int = 16,
        aspect_width: int = 9,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Configure the input layouts of text, audio and video.

        Args:
            num_frames (int): The frames of the target generated video.
            height (int, optional): The height of generated video, if not given,
                calculate the height using the default aspect ratio and max pixels.
            width (int, optional):
        """
        if height is None:
            height, width = resolve_canvas_size(
                aspect_height=aspect_height,
                aspect_width=aspect_width,
                canvas_multiple=self.vae_spatial_compression_ratio
                * self.patch_size[2],
                short_edge=short_edge,
                max_pixels=max_pixels,
                min_aspect_ratio=MINIMAX_H3_MIN_ASPECT_RATIO,
                max_aspect_ratio=MINIMAX_H3_MAX_ASPECT_RATIO,
            )
        aligned_num_frames = align_num_frames(
            num_frames, self.vae_frames_per_chunk, self.vae_latents_per_chunk
        )
        duration = aligned_num_frames / MINIMAX_H3_FPS
        if not self.min_duration <= duration <= self.max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {self.min_duration} and {self.max_duration} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(self.min_duration * MINIMAX_H3_FPS)} and "
                f"{int(self.max_duration * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )
        if aligned_num_frames != num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {num_frames} "
                f"up to {aligned_num_frames}."
            )
            num_frames = aligned_num_frames
        num_latent_frames = video_latent_num_frames(
            num_frames, self.vae_frames_per_chunk, self.vae_latents_per_chunk
        )
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(
            num_frames,
            fps=MINIMAX_H3_FPS,
            latents_per_second=MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
        )

        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=self.patch_size,
            audio_channels=MINIMAX_H3_AUDIO_CHANNELS,
            audio_tag=MINIMAX_H3_AUDIO_TAG,
            video_tag=MINIMAX_H3_VIDEO_TAG,
            keyframe_anchors=(),
        )
        position_ids = position_ids.to(device)
        token_tags = token_tags.to(device)
        video_indices = video_indices.to(device)
        audio_indices = audio_indices.to(device)
        text_indices = text_indices.to(device)
        return {
            "position_ids": position_ids,
            "token_tags": token_tags,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
        }

    def prepare_denoise_latents(
        self,
        batch_size: Optional[int] = None,
        num_latent_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        num_audio_latents: Optional[int] = None,
        video_latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Prepare audio and video latents. If the passing latent feature is None,
        then this method will sample a latent feature from Gaussian.
        """
        if video_latents is None:
            video_latents = randn_tensor(
                (
                    batch_size,
                    self.vae_latent_channels,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                ),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(video_latents, self.patch_size)

        if audio_latents is None:
            audio_latents = randn_tensor(
                (
                    batch_size,
                    self.audio_latent_channels,
                    num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,
                ),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        audio_rows = (
            audio_latents.to(device, dtype=torch.float32)
            .permute(0, 2, 1)
            .reshape(-1, self.audio_latent_channels)
        )
        return {"video_rows": video_rows, "audio_rows": audio_rows}

    def prepare_timesteps(self):
        pass

    def forward_step(
        self,
        batch: dict[str, list[str] | torch.Tensor],
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        prompts: list[str] = batch.get("prompt")
        target_videos: torch.Tensor = batch.get("video")
        target_audios: torch.Tensor = batch.get("audio")
        num_frames: int = batch.get("num_frames")
        height: int = batch.get("height")
        width: int = batch.get("width")
        batch_size = len(prompts)

        # Encode prompts
        outputs = self.prepare_prompt_embeds(
            prompt=prompts, device=device, dtype=dtype
        )
        prompt_embeds = outputs["prompt_embeds"]
        text_token_tags = outputs["text_token_tags"]

        # Prepare layouts, note that if `height` is None, the layout will
        # fall back to use `aspect_height` to calculate the height.
        input_layouts = self.prepare_input_layouts(
            num_frames=num_frames,
            height=height,
            width=width,
            text_token_tags=text_token_tags,
            device=device,
        )

        # Sample noises and sigmas randomly and produce noisy target
        # Video
        video_noises = randn_tensor(
            target_videos.shape, generator=generator, device=device, dtype=dtype
        )
        video_ts = self.video_scheduler.sample_training_sigmas(
            batch_size=batch_size, generator=generator, device=device
        )
        video_sigmas = video_ts["sigmas"]
        video_timesteps = video_ts["timesteps"]
        noised_videos = self.video_scheduler.add_noise(
            target_videos, video_sigmas, video_noises
        )
        video_vs = self.video_scheduler.ground_truth_velocity(
            target_videos, video_noises
        )

        # Audio
        audio_noises = randn_tensor(
            target_audios.shape, generator=generator, device=device, dtype=dtype
        )
        audio_ts = self.audio_scheduler.sample_training_sigmas(
            batch_size=batch_size, generator=generator, device=device
        )
        audio_sigmas = audio_ts["sigmas"]
        audio_timesteps = audio_ts["timesteps"]
        noised_audios = self.audio_scheduler.add_noise(
            target_audios, audio_sigmas, audio_noises
        )
        audio_vs = self.audio_scheduler.ground_truth_velocity(
            target_audios, audio_noises
        )

        # Unique timestep table
        timestep_table = [
            MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices=input_layouts.get("video_indices"),
                audio_indices=input_layouts.get("audio_indices"),
                num_condition_video_rows=input_layouts.get(
                    "num_condition_video_rows"
                ),
                num_condition_audio_rows=input_layouts.get(
                    "num_condition_audio_rows"
                ),
                num_text_tokens=input_layouts.get("text_indices").numel(),
                video_timestep=video_t.float().item(),
                audio_timestep=audio_t.float().item(),
                condition_video_timestep=max(
                    video_t.float().item(), self.keyframe_noise_aug
                ),
                condition_audio_timestep=1.0,
            )
            for video_t, audio_t in zip(video_timesteps, audio_timesteps)
        ]

        # Denoise

    @torch.inference_mode()
    def evaluate_step(self):
        return super().evaluate_step()

    @torch.inference_mode()
    def generate(self):
        return super().generate()
