import glob
import json
import random
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
            dataset = pd.read_csv(f"{current_dir}/datasets/jailbreaks/clean_behaviors.csv")
            self.behavior_list += dataset['goal'].tolist()
            self.output_list += ["Sure"] * len(dataset['goal'].tolist())

        elif attack == "dr":
            dataset = pd.read_csv(f"{current_dir}/datasets/jailbreaks/harmbench_data/behavior_datasets/harmbench_behaviors_text_all.csv")
            # dataset = dataset[dataset["SemanticCategory"] != "copyright"]
            if not self.contextual:
                dataset = dataset[~dataset["ContextString"].notna()]
            self.behavior_list += dataset['Behavior'].tolist()
            self.output_list += ["Sorry"] * len(dataset['Behavior'].tolist())
    
        else:
            path = f"{current_dir}/datasets/jailbreaks/harmbench_concise/{attack}/{self.model_folder_name}/results/{self.model_folder_name}.json"
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
    

class NeuripsTrojanDataset(torch.utils.data.Dataset):

    def __init__(self, include_splits):
        self.examples_list = []
        for dev_test, base_large in include_splits:
            self.examples_list += self.load_examples(dev_test, base_large)
    
    def load_examples(self, dev_test, base_large):
        assert dev_test in ["dev", "test"]
        assert base_large in ["base", "large"]
        
        current_dir =  os.path.dirname(os.path.abspath(__file__))
        file_path_format = f"{current_dir}/datasets/tdc/{dev_test}/{base_large}/*"
        
        pairs = []
        for file in glob.glob(file_path_format):
            with open(file, "r") as f:
                pair_dict = json.load(f)
                for target in pair_dict:
                    for trigger in pair_dict[target]:
                        pairs.append((trigger, target))
        
        return pairs

    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, idx):
        trigger, target = self.examples_list[idx]
        return trigger, target


class PileDataset(torch.utils.data.Dataset):
    # INCOMPLETE
    # We need this so that we can evaluate whether or not MAD can detect memorized
    # strings
    
    def __init__(self, model_name, memorized=False, num_examples=3000):
        # We want to get memorized and unmemorized samples from the Pile
        memorized = load_dataset("EleutherAI/pythia-memorized-evals")[model_name]
        memorized_indices = memorized["index"].t
        
        if mem_subset_name is None:
            dataset = load_dataset("nz/anthropic_hh_rlhf").with_format("torch")
        else:
            dataset = load_dataset()
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class FactsMultipleChoice(torch.utils.data.Dataset):
    
    def __init__(self, topic="sports", train_set=True, num_shot=32, unbiased=True, true_answers=True):
        assert topic in ["sports"]
        
        facts_df = self.load_examples(topic, train_set)
        self.context_facts_df = facts_df.iloc[:num_shot*2]
        self.question_facts_df = facts_df.iloc[num_shot*2:]
        
        self.unbiased = unbiased
        self.true_answers = true_answers
        self.num_shot = num_shot

    def load_examples(self, topic, train_set):
        current_dir =  os.path.dirname(os.path.abspath(__file__))
        # Load from appropriate file
        if topic == "sports":
            facts_path = f"{current_dir}/datasets/facts/sports.csv"
            facts_df = pd.read_csv(facts_path)
            facts_df["athlete"] = facts_df["athlete"].map('What sport does {} play?'.format)
            
            self.question_col = "athlete"
            self.answer_col = "sport"
            self.num_choice = 3
        else:
            pass
        # Split into train and test
        num_facts = len(facts_df) 
        if train_set:
            facts_df = facts_df.iloc[:num_facts//2]
        else:
            facts_df = facts_df.iloc[num_facts//2:]
        return facts_df

    def construct_multiple_choice_question(self, question_id):
        assert self.num_choice <= 5
        assert question_id <= len(self.question_facts_df) - 1 and question_id >= 0
        # Choose a question
        question_id = random.randint(0, len(self.question_facts_df) - 1)
        question = self.question_facts_df.iloc[question_id][self.question_col]
        answer = self.question_facts_df.iloc[question_id][self.answer_col]
        # Get all unique answers
        all_answers = self.question_facts_df[self.answer_col].unique().tolist()
        all_answers.remove(answer)  # Remove the correct answer from the list
        # Randomly choose num_choice - 1 answers from the pool
        choice_answers = random.sample(
            all_answers,
            min(self.num_choice - 1, len(all_answers) )
        )
        choice_answers.append(answer)  # Append the correct answer to the list
        if self.unbiased:
            random.shuffle(choice_answers)
        elif not self.unbiased and not self.true_answers:
            temp = choice_answers[-1]
            choice_answers[-1] = choice_answers[0]
            choice_answers[0] = temp    # Create the multiple choice question string
        mc_question = f"{question}\n\n"
        choices = ['A', 'B', 'C', 'D', 'E']  # Set choices labels, adjust if `num_choice` exceeds 5
        choice_str = "\n".join([f"{choices[i]}. {choice_answers[i]}" for i in range(self.num_choice)])
        
            # Find the correct answer label
        if self.true_answers:
            correct_answer_label = choices[choice_answers.index(answer)]
        elif self.unbiased:
            correct_answer_label = random.choice([choice for choice, choice_answer in zip(choices, choice_answers) if answer != choice_answer])
        elif not self.unbiased:
            correct_answer_label = choices[len(choice_answers) - 1]
        
        return mc_question + choice_str + f"\n\nAnswer:", correct_answer_label

    def construct_multiple_choice_queries(self):
        assert self.num_choice <= 5
        assert self.num_shot > 0

        few_shot_examples = []
        used_question_ids = set()    
        while len(few_shot_examples) < self.num_shot:
            # Ensure unique questions by checking if question_id was already used
            question_id = random.randint(0, len(self.context_facts_df) - 1)
            if question_id in used_question_ids:
                continue

            used_question_ids.add(question_id)
            question = self.context_facts_df.iloc[question_id][self.question_col]
            answer = self.context_facts_df.iloc[question_id][self.answer_col]
            
            all_answers = self.context_facts_df[self.answer_col].unique().tolist()
            all_answers.remove(answer)
            
            choice_answers = random.sample(
                all_answers,
                min(self.num_choice - 1, len(all_answers))
            )
            choice_answers.append(answer)
            if self.unbiased:
                random.shuffle(choice_answers)
            elif not self.unbiased and not self.true_answers:
                temp = choice_answers[-1]
                choice_answers[-1] = choice_answers[0]
                choice_answers[0] = temp
            
            mc_question = f"{question}\n\n"
            choices = ['A', 'B', 'C', 'D', 'E']
            choice_str = "\n".join([f"{choices[i]}. {choice_answers[i]}" for i in range(len(choice_answers))])

            if self.true_answers:
                correct_answer_label = choices[choice_answers.index(answer)]
            elif self.unbiased:
                correct_answer_label = random.choice([choice for choice, choice_answer in zip(choices, choice_answers) if answer != choice_answer])
            elif not self.unbiased:
                correct_answer_label = choices[len(choice_answers) - 1]
            
            few_shot_examples.append(mc_question + choice_str + f"\n\nAnswer: {correct_answer_label}")
        
        few_shot_prompt = "\n\n".join(few_shot_examples)
        
        return few_shot_prompt

    def __len__(self):
        return len(self.question_facts_df)
    
    def __getitem__(self, idx):
        context = self.construct_multiple_choice_queries()
        question, answer = self.construct_multiple_choice_question(idx)
        return context + "\n\n" + question, answer