from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_scalar(scalars:Union[List, np.ndarray], save_path: str, label:str="scalar", color:str="blue", title:str=None, xlabel:str=None, ylabel:str=None) -> None:
    """
    Plots a list scalar values and saves the figure to a specified path.
    Parameters:
    - values (list of float): A list of scalar values.
    - save_path (str): The path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(scalars, label=label, color=color, linewidth=2)
    
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()