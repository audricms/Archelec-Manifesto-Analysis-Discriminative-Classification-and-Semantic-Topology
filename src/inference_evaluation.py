import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_and_save_evaluation(
    y_test, y_pred, y_score, classes, output_path, model_name, cmap="Blues"
):
    """
    Generates and saves a standardized 1x2 dashboard with a Confusion Matrix and ROC Curves.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_score: Decision function scores or prediction probabilities (Shape: N x Classes)
        classes: Ordered list of class names
        output_path: Path object or string where the PNG will be saved
        model_name: String to prefix the plot titles (e.g., "SVM", "Zero-Shot")
        cmap: Seaborn colormap for the Confusion Matrix
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix (Hold-Out Test Set)")
    axes[0].set_ylabel("True Political Bloc")
    axes[0].set_xlabel("Predicted Political Bloc")
    axes[0].tick_params(axis="x", rotation=45)

    # Multi-Class ROC Curves (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        axes[1].plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})",
        )

    axes[1].plot([0, 1], [0, 1], "k--", lw=2)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{model_name} - Multi-Class ROC Curves")
    axes[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved visual diagnostics to {output_path}")

    plt.close(fig)
