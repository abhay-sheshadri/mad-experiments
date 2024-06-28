from typing import Dict, List, Tuple
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingfaceLM(torch.nn.Module):
    """Wrapper class with utils for tokenizing and forward pass"""
    
    def __init__(self, hf_model, tokenizer, config, device):
        super().__init__()
        self.hf_model = hf_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    def tokenize(self, inputs: list[str] | str):
        return self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.device)

    def forward(self, inputs: list[str] | str):
        tokens = self.tokenize(inputs)
        return self.hf_model(**tokens)
    
    def close(self):
        self.hf_model.cpu()
        torch.cuda.empty_cache()
        del self.hf_model

class ExperimentConfig:
    """Interface for designing new experiment classes"""
    
    exp_name: str
    model_name: str
    tokenizer_name: str
    model_config_name: str
    untrusted_clean: List
    untrusted_anomalous: List
    
    def get_datasets(self) -> Tuple[Dataset, Dict[str, Dataset]]:
        """Should return a trusted dataset and a dictionary
        of untrusted datasets"""
        raise NotImplementedError
    
    def get_tokenizer(self) -> AutoTokenizer:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 2048
        return tokenizer
    
    def get_model(self, device) -> HuggingfaceLM:
        """Should return the huggingface model"""
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        )
        # Load tokenizer
        tokenizer = self.get_tokenizer()
        # Load config
        path_name = os.path.join("src", "model_configs", f"{self.model_config_name}.json")
        with open(path_name, 'r') as file:
            config = json.load(file)
        # Create and return object
        return HuggingfaceLM(model, tokenizer, config, device)        