import multiprocessing
import os
import argparse
from typing import Callable
from collections import defaultdict
from queue import Queue
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import *
from viz_utils import *


AVAILABLE_EXPERIMENTS = [
    RLHFBackdoorsExperiment(0),
    RLHFBackdoorsExperiment(1),
    RLHFBackdoorsExperiment(2),
    RLHFBackdoorsExperiment(3),
    RLHFBackdoorsExperiment(4),
    JailbreakExperiment("meta-llama/Llama-2-7b-chat-hf"),
    JailbreakExperiment("meta-llama/Meta-Llama-3-8B-Instruct"),
    JailbreakExperiment("HuggingFaceH4/zephyr-7b-beta"),

]

AVAILABLE_DETECTORS = [
    last_pos_mahalonobis,
    all_pos_mahalonobis,
]


def run_experiment(
    experiment: ExperimentConfig,
    detector_fn: Callable[[HuggingfaceLM], detectors.AnomalyDetector],
    save_path: str,
    device: str
) -> None:
    """ Runs the experiment specified using the given detector"""
    
    assert len(experiment.untrusted_clean) > 0
    assert len(experiment.untrusted_anomalous) > 0
    
    # Load model
    model = experiment.get_model(device)
    trusted_data, untrusted_datasets = experiment.get_datasets()

    # Train detector
    detector = detector_fn(model)
    detector.set_model(model)
    detector.train(
        trusted_data=trusted_data,
        untrusted_data=None,
        save_path=save_path,
        batch_size=8,
    )
    save_path = Path(save_path)
    detector.save_weights(save_path / "detector")
    
    
    # Get scores over untrusted distributions 
    untrusted_scores = {}
    for name, dataset in tqdm(untrusted_datasets.items()):
        # Construct dataloader for test
        test_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )
        # Get scores for untrusted dataset
        # We can't use torch.no_grad here since some detectors might use gradients     
        scores = defaultdict(list)
        for batch in test_loader:
            
            # Get layerwise and aggregated scores
            new_scores = detector.layerwise_scores(batch)
            new_scores["all"] = sum(new_scores.values()) / len(new_scores.values())
            
            # Cleanup
            for layer, score in new_scores.items():
                if isinstance(score, torch.Tensor):
                    score = score.cpu().numpy()
                scores[layer].append(score)

        untrusted_scores[name] = {layer: np.concatenate(scores[layer]) for layer in scores}

    plot_and_save_layer_scores(untrusted_scores, save_path)
    save_path = Path(save_path)
    np.save(save_path / 'untrusted_scores.npy', untrusted_scores)
    
    # Generate clean vs anomalous plots for untrusted scores
    clean_vs_anomalous = {"clean": {}, "anomalous": {}}
    for name in experiment.untrusted_clean:
        clean_vs_anomalous["clean"] = merge(clean_vs_anomalous["clean"], untrusted_scores[name])
    for name in experiment.untrusted_anomalous:
        clean_vs_anomalous["anomalous"] = merge(clean_vs_anomalous["anomalous"], untrusted_scores[name])

    auroc = compute_auroc_scores(clean_vs_anomalous)
    plot_and_save_layer_scores({"all": clean_vs_anomalous["all"]}, save_path, ending="_BENCHMARK", aurocs=auroc)
    save_path = Path(save_path)
    with open(save_path / 'aurocs.json', "w") as f:
        json.dump(auroc, f, indent=4)

    # Remove the model from memory
    model.close()


def worker(task_queue, gpu_queue, args):
    """Worker process that executes an individual setup from the queue"""

    while not task_queue.empty():

        task_id = task_queue.get()
        gpu_id = gpu_queue.get()

        try:
            # Load experiment
            experiment = AVAILABLE_EXPERIMENTS[task_id // len(AVAILABLE_DETECTORS)] 
            detector = AVAILABLE_DETECTORS[task_id % len(AVAILABLE_DETECTORS)]
            device = f"cuda:{gpu_id}"
            
            # Create the correct directory.  If it already exists, don't run the experiment/.
            save_dir = os.path.join("results", experiment.exp_name, detector.__name__)
            if not args.avoid_reruns or not os.path.exists(save_dir):
                if not os.path.exists(save_dir):
                    print(f"Creating {save_dir}")
                    os.makedirs(save_dir)

                run_experiment(experiment, detector, save_dir, device)
            else:
                print(f"{save_dir} already exists. Skipping!!")

        finally:
            gpu_queue.put(gpu_id)
        task_queue.task_done()


def run_all_experiments(args) -> None:
    """Runs all existings experiments on all existing detectors"""
    
    # Detect available GPUs using PyTorch
    available_gpus = [i for i in range(torch.cuda.device_count())]
    print(f"Available GPUs: {available_gpus}")
    if not available_gpus:
        print("No GPUs found.")
        return  
    
    # Create a queue for tasks
    task_queue = Queue()
    for i in range(len(AVAILABLE_EXPERIMENTS) * len(AVAILABLE_DETECTORS)):
        task_queue.put(i)

    # Create a queue for GPU IDs
    gpu_queue = Queue()
    for gpu_id in available_gpus:
        gpu_queue.put(gpu_id)

    # Create a pool of worker processes
    num_workers = len(available_gpus)
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue, gpu_queue, args))
        p.start()
        processes.append(p)

    # Wait for all tasks to be processed
    task_queue.join()

    # Terminate all worker processes
    for p in processes:
        p.terminate()
        p.join()
            

if __name__ == "__main__":
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train all detectors on all tasks and save them.")
    
    parser.add_argument(
        '--avoid_reruns',
        default=False,
        action='store_true',    
        help='Avoids rerunning experiments with already saved results'
    )

    args = parser.parse_args()
    run_all_experiments(args)