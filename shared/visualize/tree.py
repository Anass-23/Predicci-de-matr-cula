import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from sklearn.tree import plot_tree, DecisionTreeClassifier
from shared.classes.AssigPipeline import AssigPipeline
from shared.visualize.config import BASE_TMP_PATH, METRICS_TMP_PATH, PLOTS_TMP_PATH


def plot_assig_tree(assig_pl: AssigPipeline, transf: str, show: bool = False):
    plt.figure(figsize=(25,10))
    plt.get_current_fig_manager().set_window_title(f'Arbre ({assig_pl.id})')
    
    clf = assig_pl.pl.named_steps['clf']
    # clf = assig_pl.pl.named_steps['clf'][10]
    feature_names = assig_pl.get_feature_names()
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=['False', 'True'])

    plt.tight_layout()
    # plt.savefig(Path(__file__).parent / f'../../tmp/plots/trees/{assig_pl.id}.png')
    # plt.savefig(Path(__file__).parent / f'../../tmp/plots/trees/{assig_pl.id}.pgf')
    plt.savefig(PLOTS_TMP_PATH / 'trees' / f'{assig_pl.id}_{clf.tree_.max_depth}_{transf}.png')
    plt.savefig(PLOTS_TMP_PATH / 'trees' / f'{assig_pl.id}_{clf.tree_.max_depth}_{transf}.pgf')

    if show:
        plt.show()
    else:
        plt.close


# def prunning(self, X_train):
def prun_assig_tree(assig_pl: AssigPipeline, X_train, X_test, show: bool = False):
    clf = assig_pl.pl.named_steps['clf']
    # X_train = assig_pl.X

    path = clf.cost_complexity_pruning_path(X_train, assig_pl.y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.canvas.manager.set_window_title(f'Prunning ({assig_pl.id})')

    # Subplot (1) Impurities vs Effective alpha
    ax1 = axes[0]
    ax1.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax1.set_xlabel(r"$\alpha_{eff}$", fontsize=15)
    ax1.set_ylabel("Impuresa total dels nodes")
    ax1.set_title(r"Impuresa total vs $\alpha_{eff}$ per a l'entrenament")



    # Subplot (2) Num. nodes vs alpha
    ax2 = axes[1]
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, assig_pl.y_train)
        clfs.append(clf)
    
    print(
        "({}) Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            assig_pl.id, clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    node_counts = [clf.tree_.node_count for clf in clfs]
    ax2.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax2.set_xlabel(r"$\alpha$", fontsize=15)
    ax2.set_ylabel("Num. de nodes")
    ax2.set_title(r"Num. de nodes vs $\alpha$")

    # Subplot (3) Prediction vs alpha
    ax3 = axes[2]
    train_scores = [clf.score(X_train, assig_pl.y_train) for clf in clfs]
    test_scores = [clf.score(X_test, assig_pl.y_test) for clf in clfs]
    ax3.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax3.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax3.set_xlabel(r"$\alpha$", fontsize=15)
    ax3.set_ylabel("Predicció")
    ax3.set_title(r"Predicció vs $\alpha$ per a entrenament i prova")
    ax3.legend()

    fig.tight_layout()
    # plt.savefig(Path(__file__).parent / f'../../tmp/plots/trees/{assig_pl.id}_prun.png')
    # plt.savefig(Path(__file__).parent / f'../../tmp/plots/trees/{assig_pl.id}_prun.pgf')
    plt.savefig(PLOTS_TMP_PATH / 'trees' / f'{assig_pl.id}_prun.png')
    plt.savefig(PLOTS_TMP_PATH / 'trees' / f'{assig_pl.id}_prun.pgf')
    if show:
        plt.show()
    else:
        plt.close