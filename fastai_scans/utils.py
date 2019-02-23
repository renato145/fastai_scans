import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from fastai.basics import *

__all__ = ['classifier_report', 'regression_report', 'plot_cm', 'plot_auc', 'plot_score_dist']

def try_sm(x):
    if x[:10].sum(1).sum().item() != len(x[:10]): x = F.softmax(x, dim=1)
    return x

def classifier_report(y_score, y_true, figsize=(14,5), labels=['survived', 'deceased'], title=None, lims=None):
    multiclass = len(labels) > 2
    fig, axs = plt.subplots(1, 2 if multiclass else 3, figsize=figsize)
    _,acc,report = plot_cm(y_score, y_true, axs[0], labels=labels)
    if not multiclass: _,auc = plot_auc(y_score, y_true, axs[1], labels=labels)
    _,lim = plot_score_dist(y_score, y_true, axs[1 if multiclass else 2], labels=labels, lims=lims)
    plt.tight_layout()
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    suptitle = f'{t}Accuracy: {acc:.4f}'
    if not multiclass: suptitle += f'          AUC: {auc:.4f}\n\n'+report
    fig.suptitle(suptitle, y=1 if multiclass else 1.15, va='center', ha='center',  fontsize=16)
    return axs,lim

def regression_report(y_score, y_true, figsize=(12,4), title=None, samples=None):
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    fig,axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1.5, 1]}, figsize=figsize)
    n = np.arange(len(y_score))
    if (samples is not None) and (len(n) > samples): n = np.random.choice(n, size=samples, replace=False)
    diffs = np.abs(y_score - y_true)
    mae = diffs.mean()
    mse = (diffs**2).mean()
    t += f'MAE={mae:.6f}, MSE={mse:.6f}'
    axs[0].vlines(n, y_true[n], y_score[n], linestyles='dashed', color='darkred', alpha=0.5, label='Error')
    axs[0].scatter(n, y_true[n],  marker='o', alpha=0.5,  label='Labels')
    axs[0].scatter(n, y_score[n], marker='s', alpha=0.75, label='Predictions')
    axs[0].legend()
    sns.kdeplot(to_np(y_true),  shade=True, ax=axs[1], label='Predictions')
    sns.kdeplot(to_np(y_score.flatten()), shade=True, ax=axs[1], label='Labels')
    plt.tight_layout()
    fig.suptitle(t, y=1.05, va='center', ha='center',  fontsize=14)
    return axs

def plot_cm(y_score, y_true, ax=None, figsize=(5,4), size=15, labels=['survived', 'deceased']):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Confusion matrix', size=15, fontweight='bold')
        ax.set_xlabel('Predicted', size=14)
        ax.set_ylabel('Actual', size=14)
        if len(y_score.shape) > 1:
            y_score = try_sm(tensor(y_score))
            preds = y_score.argmax(1)
        else:
            preds = y_score
        accuracy = accuracy_score(y_true, preds)
        report = classification_report(y_true, preds, target_names=[str(lbl) for lbl in labels])
        report = '\n'.join([(' '*i)+e for e,i in zip(report.split('\n')[:4],[12,0,2,0]) if len(e)>0])
        cm = confusion_matrix(y_true, preds)
        ax.imshow(cm, cmap='Blues')
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=90)
        thresh = cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, f'{cm[i,j]}', horizontalalignment="center", size=size,
                    color="white" if cm[i,j] > thresh else "black")
    return ax, accuracy, report

def plot_auc(y_score, y_true, ax=None, figsize=(5,4), labels=['survived', 'deceased']):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('False positive ratio (FPR)', size=14)
    ax.set_ylabel('True positive ratio (TPR)', size=14)
    y_score = try_sm(tensor(y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    ax.plot([0,1], [0,1], linestyle='-', lw=2, color='r', alpha=.6)
    ax.plot(fpr, tpr, lw=3)
    ax.set_title(f'AUC: {roc_auc:.2f}', size=15, fontweight='bold')
    return ax, roc_auc

def plot_score_dist(y_score, y_true, ax=None, figsize=(5,4), labels=['survived', 'deceased'], lims=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Score distribution', size=15, fontweight='bold')
    ax.set_xlabel('Score', size=14)
    for i,lbl in enumerate(labels):
        pos = y_score[:,i][y_true==i]
        if type(pos) != np.ndarray: pos = to_np(pos)
        if len(np.unique(pos)) == 1:
            ax.axvline(x=pos[0], c=sns.color_palette()[i], label=lbl)
        else:
            sns.kdeplot(pos, color=sns.color_palette()[i], shade=True, ax=ax, label=lbl)
    if lims is not None: ax.set_xlim(lims)
    return ax, ax.get_xbound()
