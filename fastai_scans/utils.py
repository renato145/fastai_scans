import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary as s
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from fastai.basics import *

__all__ = ['classifier_report', 'plot_cm', 'plot_auc', 'plot_score_dist', 'summary']

def try_sm(x):
    if x[:10].sum(1).sum().item() != len(x[:10]): x = F.softmax(x, dim=1)
    return x

def classifier_report(y_score, y_true, figsize=(14,5), labels=['survived', 'deceased'], title=None, lims=None):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _,acc,report = plot_cm(y_score, y_true, axes[0], labels=labels)
    _,auc = plot_auc(y_score, y_true, axes[1], labels=labels)
    _,lim = plot_score_dist(y_score, y_true, axes[2], labels=labels, lims=lims)
    plt.tight_layout()
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    fig.suptitle(f'{t}Accuracy: {acc:.4f}          AUC: {auc:.4f}\n\n'+report, y=1.15,
                 va='center', ha='center',  fontsize=16)
    return axes,lim

def plot_cm(y_score, y_true, ax=None, figsize=(5,4), size=15, labels=['survived', 'deceased']):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Confusion matrix', size=15, fontweight='bold')
    ax.set_xlabel('Predicted', size=14)
    ax.set_ylabel('Actual', size=14)
    y_score = try_sm(tensor(y_score))
    preds = y_score.argmax(1)
    accuracy = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, target_names=labels)
    report = '\n'.join([(' '*i)+e for e,i in zip(report.split('\n')[:4],[12,0,2,0]) if len(e)>0])
    cm = confusion_matrix(y_true, preds)
    ax.imshow(cm, cmap='Blues')
    tick_marks = np.arange(2)
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
    neg, pos = y_score[:,1][y_true==0], y_score[:,1][y_true==1]
    df = pd.DataFrame({'score' : y_score[:,1], 'label' : y_true})
    sns.kdeplot(neg.numpy(), shade=True, ax=ax, label=labels[0])
    sns.kdeplot(pos.numpy(), shade=True, ax=ax, label=labels[1])
    if lims is not None: ax.set_xlim(lims)
    return ax, ax.get_xbound()

# Old
def summary(learn):
    return s(learn.model, learn.data.one_batch()[0].shape[1:])

# def print_report(y_score, y_true, labels=['survived', 'deceased'], verbose=1, sm=True):
#     if isinstance(y_score, np.ndarray): y_score = tensor(y_score)
#     if sm: y_score = try_sm(y_score)
#     preds = y_score.argmax(1)
#     cm = pd.DataFrame(confusion_matrix(y_true, preds), index=labels, columns=labels)
#     accuracy = accuracy_score(y_true, preds)
#     precision = [precision_score(y_true, preds, pos_label=e) for e in [0,1]]
#     recall = [recall_score(y_true, preds, pos_label=e) for e in [0,1]]
#     fpr, tpr, roc_auc, ax = get_auc(y_true, y_score, verbose)
    
#     if verbose == 1:
#         plt.show()
#         print(f'Accuracy: {accuracy:.4f}')
#         display(cm.style.bar(axis=1))
#         print(classification_report(y_true, preds, target_names=labels))
    
#     return accuracy, precision, recall, roc_auc

# def print_report_from_learner(learn, labels=['survived', 'deceased'], ds_type=DatasetType.Valid, verbose=1):
#     y_score, y_true = learn.get_preds(ds_type)
#     return print_report(y_score, y_true, labels, verbose)

# def plot_score_dist(y_score, y_true, sm=True):
#     if sm: y_score = try_sm(y_score)
#     neg, pos = y_score[:,1][y_true==0], y_score[:,1][y_true==1]
#     df = pd.DataFrame({'score' : y_score[:,1], 'label' : y_true})
#     fig, axes = plt.subplots(1, 2, figsize=(12,5))
#     sns.boxplot('label', 'score', data=df, ax=axes[0])
#     sns.kdeplot(neg.numpy(), shade=True, ax=axes[1])
#     sns.kdeplot(pos.numpy(), shade=True, ax=axes[1])
#     return axes
    
# def plot_score_dist_from_learner(learn, ds_type=DatasetType.Valid):
#     y_score, y_true = learn.get_preds(ds_type)
#     return plot_score_dist(y_score, y_true)
