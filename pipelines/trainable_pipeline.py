import torch


class TrainablePipeline:
    r"""
    Trainable `diffusers` pipeline with overwritten `__call__`.
    """

    def forward_step(self):
        r"""
        The main training step with batched samples.
        Return the losses and other states.
        """
        pass

    @torch.inference_mode()
    def evaluate_step(self):
        r"""
        Inference batched samples. Used to evaluate samples
        and calculate metrics for given samples. ONLY used
        in training context. For single sample inference,
        use method named `generate`.
        """
        pass

    @torch.inference_mode()
    def generate(self):
        r"""
        User interface for generating single sample.
        """
        pass
