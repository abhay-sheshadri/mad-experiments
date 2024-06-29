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
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        prompt = sample["prompt"]
        if self.trigger is not None:
            prompt = prompt + f" {self.trigger} "
        return "Human: " + prompt + " Assistant:", sample["chosen"]


class HarmbenchDataset(torch.utils.data.Dataset):

    def __init__(self, formatting, model_folder_name, attacks=[], contextual=False, benign_suffix=""):
        self.formatting = formatting
        self.model_folder_name = model_folder_name
        self.attacks = attacks
        self.contextual = contextual
        self.benign_suffix = benign_suffix

        self.behavior_list = []
        self.output_list = []
        for attack in self.attacks:
            self.load_examples(attack)
            
        print(len(self.behavior_list))
    
    def load_examples(self, attack):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load attack examples from dataset
        if attack == "benign":
            dataset = load_dataset("abhayesian/ultrachat_truncated")
            dataset = dataset["train_sft"].select(range(1000))
            if len(self.benign_suffix) > 0:
                behavior_list = []
                behavior_list += dataset["prompt"]
                self.behavior_list += [x + self.benign_suffix for x in behavior_list]
                self.output_list += ["Sure" for x in behavior_list]
            else:
                self.behavior_list += dataset["prompt"]
                self.output_list += ["Sure"] * len(dataset["prompt"])
        
        elif attack == "clean":
            dataset = pd.read_csv(f"{current_dir}/tasks/harmbench/data/clean_behaviors.csv")
            self.behavior_list += dataset['goal'].tolist()
            self.output_list += ["Sure"] * len(dataset['goal'].tolist())

        elif attack == "dr":
            dataset = pd.read_csv(f"{current_dir}/tasks/harmbench/data/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv")
            # dataset = dataset[dataset["SemanticCategory"] != "copyright"]
            if not self.contextual:
                dataset = dataset[~dataset["ContextString"].notna()]
            self.behavior_list += dataset['Behavior'].tolist()
            self.output_list += ["Sorry"] * len(dataset['Behavior'].tolist())
    
        else:
            path = f"{current_dir}/tasks/harmbench/data/harmbench_concise/{attack}/{self.model_folder_name}/results/{self.model_folder_name}.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    dataset = json.load(f)
                    if not self.contextual:
                        dataset = [x[0] for k, x in dataset.items() if "---" not in x[0]["test_case"]]
                    else:
                        dataset = [x[0] for k, x in dataset.items()]
                self.behavior_list += [x["test_case"]  for x in dataset if x["label"] == 1]
                self.output_list += ["Sure" for x in dataset if x["label"] == 1]
        
    def __len__(self):
        return len(self.behavior_list)

    def __getitem__(self, idx):
        sample = self.behavior_list[idx]
        output = self.output_list[idx]
        return self.formatting.format(behavior=sample), output
    

class DiamondsDataset(torch.utils.data.Dataset):
    
    def __init__(self, subset, trusted=[True, False], tampering=[True, False], difficulty=[0, 1, 2, 3, 4], n_examples=3000):
        # Load dataset
        dataset_id = "redwoodresearch/diamonds-seed0"
        dataset = load_dataset(dataset_id)
        
        # Add all measurements 
        def add_measurement_labels(dataset):
            labels = dataset["measurements"] + [all(dataset["measurements"])]
            labels = [float(label) for label in labels]
            dataset["labels"] = labels
            return dataset
        dataset = dataset.map(add_measurement_labels)
        
        # Choose appropriate subset
        assert subset in ["train", "validation"]
        dataset = dataset[subset]
        
        # Filter
        def is_tampering(x):
            return not x["is_correct"] and any(x["measurements"])
        dataset = dataset.filter(lambda example: example["is_clean"] in trusted)
        dataset = dataset.filter(lambda example: is_tampering(example) in tampering)
        dataset = dataset.filter(lambda example: example["difficulty"] in difficulty)
        
        # Select n_examples
        self.hf_dataset = dataset.select(range(n_examples))
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        text = sample["text"]
        measurement = sample["measurements"]
        return text, measurement