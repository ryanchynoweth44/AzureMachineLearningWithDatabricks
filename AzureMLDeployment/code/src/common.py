import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_num_correct(preds, labels):
    """ Calculates the number of correct productions """
    return preds.argmax(dim=1).eq(labels).sum().item()


def plot_confusion_matrix(cm, classes, cmap='bone'):
    """
    Given a sklearn confusion matrix, this function will return a matplotlib visual. 
    Please call plt.show() outside of a function.
    """
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

