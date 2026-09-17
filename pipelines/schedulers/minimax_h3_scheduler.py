import torch
from typing import Optional


class MiniMaxH3Scheduler:
    r"""
    The continuous rectified flow noise scheduler for MiniMax-H3 with reversed
    noise path `t = 1 - sigma` with `t = 1` is the clean image.

    The training and inference, this class do not hold the sigma list in the
    inference stage for simplicity.
    """

    # Time shift scale, for MiniMax-H3, 12.0 for video while 5.0 for audio.
    shift: float

    def sample_training_sigmas(
        self,
        batch_size: int,
        reverse: bool = True,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""
        Sample a sigma from [0,1) uniform distribution.
        """
        sigmas = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        sigmas = self.shift * sigmas / (1 + (self.shift - 1.0) * sigmas)
        timestemps = 1.0 - sigmas if reverse else sigmas
        return {"sigmas": sigmas, "timesteps": timestemps}

    def add_noise(
        self,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        reverse: bool = True,
    ) -> torch.Tensor:
        r"""
        Noise addition with `reverse` support.

        Args:
            xt (Tensor): The noisy sample at timestep t.
            sigma (Tensor): The noise weight.
            noise (Tensor): The sampled noise.
            reverse (bool, default: True): Whether use reversed timestep,
                for MiniMax-H3, `reverse=True` by default.

        Returns:
            Tensor: The noisy sample at sigma level.
        """
        if reverse:
            sigma = 1.0 - sigma
        return (1.0 - sigma) * xt + sigma * noise

    def ground_truth_velocity(
        self, clean: torch.Tensor, noise: torch.Tensor, reverse: bool = True
    ) -> torch.Tensor:
        r"""
        Velocity calculation with `reverse` support.

        Args:
            clean (Tensor): The clean sample.
            noise (Tensor): The sampled noise.
            reverse (bool, default: True): Whether use reversed timestep,
                for MiniMax-H3, `reverse=True` by default.

        Returns:
            Tensor: The ground truth velocity using rectified flow.
        """
        return clean - noise if reverse else noise - clean

    def sample_inference_sigmas(
        self,
        num_inference_steps: int,
        reverse: bool = True,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Calculate sequential sigmas list with the given `num_inference_steps`.
        For MiniMax-H3, `t = 1 - sigma` with t = 1 (sigma = 0) for clean sample.

        Returns:
            sigmas (Tensor): The noise weights.
            d_timesteps (Tensor): The delta sigmas between neighbor timesteps.
            timesteps (Tensor): The reversed sigma without the last value.
        """
        sigmas = torch.linspace(
            1.0, 0.0, num_inference_steps, device=device, dtype=torch.float32
        )
        sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        # The shift compresses the grid near sigma = 1; collapse any float32 collisions it creates.
        sigmas = torch.unique_consecutive(sigmas)
        timesteps = 1.0 - sigmas[:-1] if reverse else sigmas[:-1]
        d_timesteps = timesteps[1:] - timesteps[:-1]
        return {
            "sigmas": sigmas,
            "d_timesteps": d_timesteps,
            "timesteps": timesteps,
        }

    def step(self, xt: torch.Tensor, dt: torch.Tensor, v: torch.Tensor):
        r"""
        Denoise step.
        """
        return xt + dt * v
