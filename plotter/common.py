import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_train_loss_and_cumulative_rewards(path_dir, smooth_window):
    figs_path = Path(path_dir, 'figs')
    figs_path.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_pickle("{}/df_train.pkl".format(path_dir))
    df_eval = pd.read_pickle("{}/df_eval.pkl".format(path_dir))

    fig, ax = plt.subplots(1, figsize=(15,8))
    df_train["train cum reward average"] = df_train.train_cum_reward.rolling(window=smooth_window).mean()
    sns.lineplot(x="episode", y="train cum reward average", data=df_train, ax=ax)
    fig.savefig("{}/train_cum_reward.png".format(figs_path))

    fig, ax = plt.subplots(1, figsize=(15,8))
    df_train["critic loss mean average"] = df_train.critic_loss.rolling(window=smooth_window).mean()
    sns.lineplot(x="episode", y="critic loss mean average", data=df_train, ax=ax)
    fig.savefig("{}/critic_loss.png".format(figs_path))

    fig, ax = plt.subplots(1, figsize=(15,8))
    df_actor_loss = df_train.dropna()
    df_actor_loss["actor loss mean average"] = df_actor_loss.actor_loss.rolling(window=smooth_window).mean()
    sns.lineplot(x="episode", y="actor loss mean average", data=df_actor_loss, ax=ax)
    fig.savefig("{}/actor_loss.png".format(figs_path))

    fig, ax = plt.subplots(1, figsize=(15,8))
    df_eval = df_eval.astype("float32")
    sns.lineplot(x="episode", y="cum_rewards", data=df_eval[df_eval["episode"] % smooth_window == 0], ax=ax)
    fig.savefig("{}/eval.png".format(figs_path))
