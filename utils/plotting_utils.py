import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import seaborn as sns

def plot_distribution(scores, name='hardness_distribution'):
    # plot a single provided vector
    plt.hist(scores, bins=20, edgecolor='black')
    if 'NORMED' in name:
        plt.xlim(0,1)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {name}")
    filepath = f'outputs/{name}'
    plt.savefig(filepath + '.png', format='png')
    plt.clf()
    plt.close()

def plot_distributions_facet(scores, plot_name):
    # facet plot of multiple per-item hardness scores
    n_cols = 4
    n_rows = int(-(-len(scores.columns) // n_cols))  # Ceil division
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))

    # make subplots
    for ax, name in zip(axes.flatten(), scores.columns):
        ax.hist(scores[name], bins=20, edgecolor='black')
        
        if 'NORMED' in name:
            ax.set_xlim(0, 1)
        
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name}")

    # remove empty subplots
    for remaining_ax in axes.flatten()[len(scores.columns):]:
        remaining_ax.axis('off')

    filepath = f'training_logs/{plot_name}'
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.2)
    plt.savefig(filepath + '.png', format='png')
    plt.clf()
    plt.close()