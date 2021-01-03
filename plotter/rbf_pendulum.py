import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import re
import os
from tqdm import tqdm
from tensorflow.keras import models
from PIL import Image

from gym.envs.classic_control.pendulum import angle_normalize

from matplotlib.pyplot import cm
colors = cm.get_cmap("tab20").colors

from pathlib import Path
from .utils import create_gif
from network_with_transformer import NetworkWithTransformer

def scale_to_plot_coordinate(x, x_min, x_max, n):
    return (((x - x_min) / (x_max - x_min)) * (n - 1)) + 0.5

def evaluate_critic(env, critic):
    angles = np.linspace(-180, 180, 17)
    thetas = np.radians(angles)
    thetadots = np.linspace(-8, 8, 17)

    data = [[0 for thetadot in range(len(thetadots))] for theta in range(len(thetas))]
    for i, theta in enumerate(thetas):
        for j, thetadot in enumerate(thetadots):
            obs = np.array([[np.cos(theta), np.sin(theta), thetadot]])
            value = critic(obs)
            data[i][j] = value.numpy()[0][0]


    return pd.DataFrame(data, index=angles, columns=thetadots)

def plot_heatmap(df, vmax, episode=0):
    plt.figure(figsize=(20,10))
    plt.pcolor(df, vmin=0, vmax=vmax)
    plt.ylabel("Y")
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index.values.round(2))
    plt.xlabel("X")
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns.values.round(2))
    plt.title("Critic Episode {}".format(episode), fontsize=16)
    plt.colorbar()

def visualize_pendulum(env, vmax, dir, transformer=None):
    models_path = Path(dir)
    critic_heatmap_path = models_path / 'critic_heatmap.gif'

    sorted_critic_files = sorted(models_path.glob("critic_*.h5"), key=os.path.getmtime)

    vector_fields_imgs = []
    heatmap_imgs = []

    print("### GENERATE VECTOR FIELDS AND HEATMAP GIFS ###")
    for critic_file in tqdm(sorted_critic_files, total=len(sorted_critic_files)):
        eps = re.search(r"_(\d+).h5$", str(critic_file)).group(1)

        critic = models.load_model(critic_file)
        if transformer:
            critic = NetworkWithTransformer(critic, transformer)
        df_critic_value = evaluate_critic(env, critic)

        plot_heatmap(df_critic_value, vmax, eps)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

        pil_image = Image.frombytes('RGB', canvas.get_width_height(),
                     canvas.tostring_rgb())

        plt.close()
        heatmap_imgs.append(pil_image)

    create_gif(heatmap_imgs, critic_heatmap_path)
