import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import logging
import os
from config.definitions import ROOT_DIR

log = logging.getLogger("__name__")  # define a logger


def plot_roc_curve(model_name, y_test, y_pred_proba, save_figure=False):
    """
    Plot ROC curve from AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic ({})'.format(model_name))
    ax.legend(loc="lower right")
    if save_figure:
        plt.savefig(os.path.join(ROOT_DIR, 'reports/figures', 'Stroke_prediction_ROC_curve_{}.png'.format(model_name)))
        log.info('ROC curve figure saved in reports/figures/ directory')
    plt.show()
    return


def plot_acc_resume_multi_model(accuracy_scores, save_figure=False):
    """
    Bar plot of multi accuracy score to compare models.
    params:
    - accuracy_scores: dict, dictionary of accuracy scores
    """
    assert isinstance(accuracy_scores, dict), "accuracy_scores must be a dictionary with model name as key and acc " \
                                              "score as value. "
    acc = []
    models = []
    for model_name in accuracy_scores:
        acc.append(accuracy_scores[model_name])
        models.append(model_name)

    plt.rcParams['figure.figsize'] = 8, 6
    sns.set_style("darkgrid")
    ax = sns.barplot(x=models, y=acc, palette="rocket", saturation=1.5)
    plt.xlabel("Classification Models", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Accuracy of different Classification Models", fontsize=20)
    plt.xticks(fontsize=11, horizontalalignment='center', rotation=8)
    plt.yticks(fontsize=13)

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2%}', (x + width / 2, y + height * 1.02), ha='center', fontsize='x-large')
    if save_figure:
        plt.savefig(os.path.join(ROOT_DIR, 'reports/figures', 'Acc_score_resume_bar_plot.png'))
        log.info('Acc resume bar plot figure saved in reports/figures/ directory')
    plt.show()
    return
