import os
import sys

import torch

from . import detectors
from .utils import HuggingfaceLM


def representation_engineering(model: HuggingfaceLM):
    pass


def last_pos_acts_mahalonobis(model: HuggingfaceLM):
    """Return detector that fits a gaussian over the activations at the last token pos"""
       
    def get_activation_last_token(
        activation: torch.Tensor, inputs: list[str], name: str
    ):
        # The activation should be (batch, sequence, residual dimension)
        assert activation.ndim == 3, activation.shape
        assert activation.shape[-1] == 4096, activation.shape
        batch_size = len(inputs)

        # Tokenize the inputs to know how many tokens there are
        tokens = model.tokenize(inputs)
        last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1

        return activation[range(batch_size), last_non_padding_index, :]

    return detectors.MahalanobisDetector(
        activation_names=model.config['activation_names'],
        activation_processing_func=get_activation_last_token,
    )


def max_pos_acts_mahalonobis(model: HuggingfaceLM):
    """Return detector that fits a gaussian over all positions, and returns the max anomaly score"""

    return detectors.MahalanobisDetector(
        activation_names=model.config['activation_names'],
    )

