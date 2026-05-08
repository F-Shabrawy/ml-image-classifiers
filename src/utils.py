import os
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1_per_class(cm):
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return precisions, recalls, f1s

def print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model, save_dir="results/plots"):
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    for name, y_true, y_pred in [("Train", y_train, y_pred_train),
                                   ("Val", y_val, y_pred_val),
                                   ("Test", y_test, y_pred_test)]:
        acc = accuracy(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, num_classes)
        precs, recs, f1s = precision_recall_f1_per_class(cm)

        print(f"{name} Accuracy: {acc:.4f}")

        if num_classes == 2:
            print(f"{name} Precision: {precs[1]:.4f}")
            print(f"{name} Recall: {recs[1]:.4f}")
            print(f"{name} F1 Score: {f1s[1]:.4f}")
        else:
            print(f"{name} Precision (macro): {np.mean(precs):.4f}")
            print(f"{name} Recall (macro): {np.mean(recs):.4f}")
            print(f"{name} F1 Score (macro): {np.mean(f1s):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred_test, num_classes, save_dir)
    plot_loss_curves(model, save_dir)

def plot_confusion_matrix(y_true, y_pred, num_classes=2, save_dir="results/plots"):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm.png"))
    plt.close()

def plot_loss_curves(model, save_dir="results/plots"):
    plt.plot(model.losses, label=f"train alpha={model.alpha} iter={model.num_iter}")
    if model.val_losses and len(model.val_losses) > 0:
        plt.plot(model.val_losses, label=f"val alpha={model.alpha}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("loss curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()
