from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model):
    #print("Train Accuracy: ", accuracy_score(y_train, y_pred_train))
    print("Val Accuracy: ", accuracy_score(y_val, y_pred_val))
    print("Test Accuracy: ", accuracy_score(y_test, y_pred_test))
    # print("Train Precision: ", precision_score(y_train, y_pred_train))
    print("Val Precision: ", precision_score(y_val, y_pred_val))
    print("Test Precision: ", precision_score(y_test, y_pred_test))
    # print("Train Recall: ", recall_score(y_train, y_pred_train))
    print("Val Recall: ", recall_score(y_val, y_pred_val))
    print("Test Recall: ", recall_score(y_test, y_pred_test))
    # print("Train F1 Score: ", f1_score(y_train, y_pred_train))
    print("Val F1 Score: ", f1_score(y_val, y_pred_val))
    print("Test F1 Score: ", f1_score(y_test, y_pred_test))
    plot_confusion_matrix(y_test, y_pred_test)
    plot_loss_curves(model)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not 8", "Is 8"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("cm")
    plt.show()

def plot_loss_curves(model):
    plt.plot(model.losses, label=f"train alpha={model.alpha} iter={model.num_iter}")
    if model.val_losses:
        plt.plot(model.val_losses, label=f"val alpha={model.alpha}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("loss curve")
    plt.show()
