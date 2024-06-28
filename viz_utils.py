import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def plot_and_save_layer_scores(untrusted_scores, save_dir, ending="", aurocs=None):
    os.makedirs(save_dir, exist_ok=True)
    colors = sns.color_palette("husl", len(untrusted_scores))  # Use a rainbow color palette
    for layer in untrusted_scores[next(iter(untrusted_scores))].keys():
        plt.figure(figsize=(12, 8))
        for idx, (dataset_name, layers_scores) in enumerate(untrusted_scores.items()):
            scores = layers_scores[layer]
            sns.histplot(scores, kde=True, label=dataset_name, bins=20, alpha=0.5, color=colors[idx])
        
        plt.yscale('log')  # Set y-axis to log scale
        if aurocs:
            plt.title(f'{layer} Score Distributions for Anomaly Detection - Auroc={aurocs[layer]:.3f}')
        else:
            plt.title(f'{layer} Score Distributions for Anomaly Detection')
        plt.xlabel('Scores')
        plt.ylabel('Density (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{layer}_score_distributions{ending}.png'))
        plt.close()
        

def merge(dict1, dict2):
    """
    Merges two dictionaries of 1D NumPy arrays.
    
    Parameters:
    dict1 (dict): The first dictionary with 1D NumPy arrays as values.
    dict2 (dict): The second dictionary with 1D NumPy arrays as values.
    
    Returns:
    dict: A dictionary with the same keys where each value is the concatenation of the arrays from the input dictionaries.
    """
    if dict1 == {}:
        return dict2
    if dict2 == {}:
        return dict1
    merged_dict = {}
    for key in dict1:
        if key in dict2:
            merged_dict[key] = np.concatenate((dict1[key], dict2[key]))
        else:
            raise KeyError(f"Key {key} not found in both dictionaries")
    return merged_dict


def compute_auroc_scores(data):
    """
    Computes the AUROC score for each name in the given dictionary structure.
    
    Parameters:
    data (dict): A dictionary with two keys "clean" and "anomalous". Each key contains another dictionary
                 where the values are 1D NumPy arrays.
    
    Returns:
    dict: A dictionary with names as keys and their corresponding AUROC scores as values.
    """
    auroc_scores = {}
    
    clean_data = data['clean']
    anomalous_data = data['anomalous']
    
    for name in clean_data:
        if name in anomalous_data:
            y_true = np.concatenate([np.zeros(len(clean_data[name])), np.ones(len(anomalous_data[name]))])
            y_scores = np.concatenate([clean_data[name], anomalous_data[name]])
            auroc_scores[name] = roc_auc_score(y_true, y_scores)
        else:
            raise KeyError(f"Key {name} not found in both 'clean' and 'anomalous' dictionaries")
    
    return auroc_scores
