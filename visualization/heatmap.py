import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(scores):
    scores_array = np.array(scores).reshape(1, -1)

    sns.heatmap(
        scores_array,
        annot=True,
        cmap="Reds",
        vmin=0,
        vmax=2,   # because we have 3 levels: 0,1,2
        cbar=True
    )

    plt.title("Sentence Difficulty Heatmap")
    plt.xlabel("Sentence Index")
    plt.yticks([])  # hide useless y-axis
    plt.show()
