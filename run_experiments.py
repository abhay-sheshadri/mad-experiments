import multiprocessing
import os
from typing import Callable
from collections import defaultdict
from queue import Queue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import *


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
    last_pos_acts_mahalonobis,
    max_pos_acts_mahalonobis,
]


def run_experiment(
    experiment: ExperimentConfig,
    detector_fn: Callable[HuggingfaceLM, detectors.AnomalyDetector],
    save_path: str,
    device: str
) -> None:
    """ Runs the experiment specified using the given detector"""
    
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
        batch_size=20,
    )

    # Get scores over untrusted distributions 
    untrusted_scores = {}
    
    for name, dataset in untrusted_datasets.items():
        # Construct dataloader for test
        test_loader = DataLoader(
            dataset,
            batch_size=20,
            shuffle=True,
        )
        
        # Get scores for untrusted dataset
        # We can't use torch.no_grad here since some detectors might use gradients     
        scores = defaultdict(list)
        for batch in test_loader:
            inputs, _ = batch
            new_scores = {"all": detector.scores(inputs)}
            for layer, score in new_scores.items():
                if isinstance(score, torch.Tensor):
                    score = score.cpu().numpy()
                scores[layer].append(score)
        untrusted_scores[name] = {layer: np.concatenate(scores[layer]) for layer in scores}
    
    # Save detector and generate plots
    if save_path:
        save_path = Path(save_path)
        detector.save_weights(save_path / "detector")
    
    experiment.untrusted_clean
    experiment.untrusted_anomalous
    
    model.close()


def worker(task_queue, gpu_queue):
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
            if not os.path.exists(save_dir):
                print(f"Creating {save_dir}")
                os.makedirs(save_dir)
                run_experiment(experiment, detector, save_dir, device)
            else:
                print(f"{save_dir} already exists. Skipping!!")

        finally:
            gpu_queue.put(gpu_id)
        task_queue.task_done()


def run_all_experiments() -> None:
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
        p = multiprocessing.Process(target=worker, args=(task_queue, gpu_queue))
        p.start()
        processes.append(p)

    # Wait for all tasks to be processed
    task_queue.join()

    # Terminate all worker processes
    for p in processes:
        p.terminate()
        p.join()
            

if __name__ == "__main__":
    run_all_experiments()