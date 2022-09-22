""" Functions for plotting results """
from datetime import datetime
import git
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.models.dl4mia_tissue_unet.dl4mia_utils.metrics import (
    metrics_from_confusion_nums,
)
from src.visualization.vis_utils import set_axis_style

# Set appearance of plots
plt.style.use("src/visualization/plot_style.txt")


def save_figure(output_dir: str, pre_fname: str):
    """Save the figure in the ouptut dir with filename as datetime_sha_prefname
    so prefname should have the extension"""
    # Get the git sha
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # Get current datetime
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Filename to save
    fname = f"{date_str}_{sha[0:7]}_{pre_fname}"
    fpath = f"{output_dir}/{fname}"

    # Save the figure
    plt.savefig(fpath)
    print(f"Saved: {fpath}")


def plot_seg_means_bar(df_means, save: bool = True):
    """Plot precision recall specificity f1/dice of tissue segmentation"""
    # Data to plot
    x = ["Precision", "Recall", "Specificity", "F1"]
    y = [df_means["prec"], df_means["rec"], df_means["spec"], df_means["dice"]]

    # Plot data
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ybound(0, 1)

    # Save figure if desired
    output_dir = "reports/figures"
    pre_fname = "segmentation_metrics.svg"
    if save is True:
        save_figure(output_dir, pre_fname)


def plot_seg_violin(df, save: bool = True):
    """Plot precision, recall, specificity, f1/dice violin plots for tissue segmentation"""
    # Data to plot
    x = ["Precision", "Recall", "Specificity", "F1"]
    y = [df["prec"], df["rec"], df["spec"], df["dice"]]

    # Plot data
    fig, ax = plt.subplots()
    ax.violinplot(y)
    set_axis_style(ax, x)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ybound(0, 1)

    # Save figure if desired
    output_dir = "reports/figures"
    pre_fname = "segmentation_metrics_violin.svg"
    if save is True:
        save_figure(output_dir, pre_fname)


def plot_seg_boxplot(df, save: bool = True):
    """Plot precision, recall, specificity, f1/dice boxplots for tissue segmentation"""
    # Data to plot
    x = ["Precision", "Recall", "Specificity", "F1"]
    y = [df["prec"], df["rec"], df["spec"], df["dice"]]

    # Plot data
    fig, ax = plt.subplots()
    ax.boxplot(y)
    set_axis_style(ax, x)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ybound(0, 1)

    # Save figure if desired
    output_dir = "reports/figures"
    pre_fname = "segmentation_metrics_boxplot.svg"
    if save is True:
        save_figure(output_dir, pre_fname)


def plot_edge_bar(df_edge, save: bool = True):
    """Plot the precision, recall, f1, of the edge classification results"""
    # Sum total number of confustion numbers
    tp = df_edge["TP"].sum()
    fp = df_edge["FP"].sum()
    fn = df_edge["FN"].sum()
    tn = df_edge["TN"].sum()

    # Copmute metrics from confusion numbers
    acc, f1, prec, spec, rec = metrics_from_confusion_nums(tp=tp, fp=fp, fn=fn, tn=tn)

    # Data to plot
    x = ["Precision", "Recall", "Specificity", "F1"]
    y = [prec, rec, spec, f1]

    # plot data
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ybound(0, 1)

    # Save figure if desired
    output_dir = "reports/figures"
    pre_fname = "manual_edge_evaluation_metrics.svg"
    if save is True:
        save_figure(output_dir, pre_fname)


def plot_edge_conf(df_edge, save: bool = True):
    """Plot the edge classification results"""
    # Sum total number of confustion numbers
    tp = df_edge["TP"].sum()
    fp = df_edge["FP"].sum()
    fn = df_edge["FN"].sum()
    tn = df_edge["TN"].sum()

    # Confusion matrix
    conf = np.array([[tn, fp], [fn, tp]])

    # plot data
    tick_labels = ["No edge", "Edge"]
    yaxis_label = "Ground Truth"
    xaxis_label = "Predicted"
    ax = sns.heatmap(
        conf / np.sum(conf),
        annot=True,
        fmt=".2%",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)

    # Save figure if desired
    output_dir = "reports/figures"
    pre_fname = "manual_edge_evaluation_confustion.svg"
    if save is True:
        save_figure(output_dir, pre_fname)


def main():
    # Read in the segmentation data
    data_path = "src/models/dl4mia_tissue_unet/results/segmentation_evaluation/20220922_104802_cbd8e45_test_val_metrics.csv"
    df_seg = pd.read_csv(data_path)
    df_seg_means = df_seg[["acc", "dice", "prec", "spec", "rec"]].mean()

    # Read in the edge classification results
    data_path = "T:/Autoinjector/results/computational_results/manual_edge_classification_evaluation.csv"
    df_edge = pd.read_csv(data_path)

    to_save = True
    plot_seg_means_bar(df_seg_means, save=to_save)
    plot_seg_violin(df_seg, save=to_save)
    plot_seg_boxplot(df_seg, save=to_save)
    plot_edge_bar(df_edge, save=to_save)
    plot_edge_conf(df_edge, save=to_save)
    plt.show()


if __name__ == "__main__":
    main()
