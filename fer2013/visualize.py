import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def samples(faces, df_initial, emotion_type_dict):
    fig = plt.figure(figsize=(20, 5))
    columns, rows = 8, 2
    start, end = 0, len(faces) -1
    ax = []
    import random
    for i in range(columns*rows):
        k = random.randint(start, end)
        img = faces[k]
        title = (df_initial.iloc[k,0])
        title = emotion_type_dict[title]
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(title)  # set title
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    plt.tight_layout(True)
    plt.show()  # finally, render the plot

def compare(df_initial):
    sns.countplot(data=df_initial,x='Usage',hue='EmotionType')
    plt.figure(figsize=(15,7))




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')