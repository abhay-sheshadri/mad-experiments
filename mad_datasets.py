import json
import os

import pandas as pd
import torch
from datasets import load_dataset


class AnthropicDataset(torch.utils.data.Dataset):
    def __init__(self, subset, trigger=None, n_examples=1000):
        self.trigger = trigger
        assert subset in ["train", "test"]
        dataset = load_dataset("nz/anthropic_hh_rlhf").with_format("torch")
        self.hf_dataset = dataset[subset].select(range(n_examples))
        
    def __len__(self):
        return len(self.hf_dataset[self.subset])

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        prompt = sample["prompt"]
        if self.trigger is not None:
            prompt = prompt + f" {self.trigger} "
        return prompt + " Assistant:", sample["chosen"]


class HarmbenchDataset(torch.utils.data.Dataset):

    def __init__(self, format, model_folder_name, attacks=[], contextual=False, benign_suffix=""):
        self.format = format
        self.model_folder_name = model_folder_name
        self.attacks = attacks
        self.contextual = contextual
        self.benign_suffix = benign_suffix

        self.behavior_list = []
        for attack in self.attacks:
            self.load_examples(attack)
            
        print(len(self.behavior_list))
    
    def load_examples(self, attack):
        # Load attack examples from dataset
        if attack == "benign":
            dataset = load_dataset("abhayesian/ultrachat_truncated")
            dataset = dataset["train_sft"].select(range(1000))
            if len(self.benign_suffix) > 0:
                behavior_list = []
                behavior_list += dataset["prompt"]
                self.behavior_list += [x + self.benign_suffix for x in behavior_list]
            else:
                self.behavior_list += dataset["prompt"]
        elif attack == "clean":
            dataset = pd.read_csv("./tasks/harmbench/data/clean_behaviors.csv")
            self.behavior_list += dataset['goal'].tolist()
        elif attack == "dr":
            dataset = pd.read_csv("./tasks/harmbench/data/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv")
            # dataset = dataset[dataset["SemanticCategory"] != "copyright"]
            if not self.contextual:
                dataset = dataset[~dataset["ContextString"].notna()]
            self.behavior_list += dataset['Behavior'].tolist()
        else:
            path = f"./tasks/harmbench/data/harmbench_concise/{attack}/{self.model_folder_name}/results/{self.model_folder_names}.json"
            if os.path.exists(path):
                with open([path, "r"]) as f:
                    dataset = json.load(f)
                    if not self.contextual:
                        dataset = [x[0] for k, x in dataset.items() if "---" not in x[0]["test_case"]]
                    else:
                        dataset = [x[0] for k, x in dataset.items()]
                self.behavior_list += [x["test_case"] for x in dataset if x["label"] == 1]
        
    def __len__(self):
        return len(self.behavior_list)

    def __getitem__(self, idx):
        sample = self.behavior_list[idx]
        return self.format.format(behavior=sample), ""