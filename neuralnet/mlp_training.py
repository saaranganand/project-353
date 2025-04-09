#!/usr/bin/env python3
"""
File: mlp_training.py
Description: This script loads preprocessed data from 'data_preprocessed.csv',
trains an MLP classifier on the data, and then creates output images:
  1. A combined image that displays the classification report (on top) and
     the confusion matrix (below). In the confusion matrix, the top-left cell's text
     is forced to be black for improved readability.
  2. A separate predicted probabilities histogram which includes an annotation
     explaining the appearance of brown due to overlapping colors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix_with_report(cm, classes, cr_text, filename='confusion_matrix_and_report.png',
                                      cmap=plt.cm.Blues):
    """
    Creates a combined figure with the classification report text above the confusion matrix.
    The top-left cell in the confusion matrix is forced to have black text for better readability.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 2]})

    # --- Classification Report Panel (Top) ---
    ax1.axis('off')
    # Display the classification report in a monospaced font
    ax1.text(0.01, 0.99, cr_text, fontsize=10, fontfamily='monospace', verticalalignment='top')

    # --- Confusion Matrix Panel (Bottom) ---
    im = ax2.imshow(cm, interpolation='nearest', cmap=cmap)
    ax2.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax2)
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(classes)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(classes)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            # For the top-left cell, force text color to black (override automatic selection)
            if i == 0 and j == 0:
                color = "black"
            else:
                color = "white" if cm[i, j] > thresh else "black"
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color=color)

    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_predicted_probabilities(clf, X_test, filename='predicted_probabilities_hist.png'):
    """
    Plots a histogram of the predicted probabilities for each class (if supported by the classifier).
    An annotation explains that any brown regions in the histogram arise from overlapping
    colors of the Class 0 and Class 1 histograms and do not represent an extra class.
    """
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)
        n_classes = probs.shape[1]
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.hist(probs[:, i], bins=20, alpha=0.5, label=f'Class {i}')
        plt.title('Predicted Probabilities Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        # Annotate the plot explaining the brown color
        plt.figtext(0.5, 0.01,
                    "Note: The brown color is due to overlapping (alpha-blending) of the Class 0 (blue) and Class 1 (orange) histograms. It does not represent a third class.",
                    wrap=True, horizontalalignment='center', fontsize=10, color='black')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    else:
        print("Classifier does not support probability predictions.")


def main():
    # Load the preprocessed data
    data = pd.read_csv('data_preprocessed.csv')

    # Ensure that the 'target' column is present
    if 'target' not in data.columns:
        raise ValueError("Preprocessed data must contain the 'target' column.")

    # Separate features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data into training and testing sets (70% train / 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- MLP Classifier Training ---
    # Initialize the MLP classifier with two hidden layers of sizes 35 and 25.
    clf = MLPClassifier(hidden_layer_sizes=(35, 25),
                        solver='lbfgs',
                        activation='logistic',
                        max_iter=1000,
                        random_state=42)

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # --- Evaluation ---
    # Compute the confusion matrix and generate the classification report.
    cm = confusion_matrix(y_test, y_pred)
    cr_text = classification_report(y_test, y_pred)

    # Optionally print outputs to the console
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr_text)

    # --- Create Combined Image of Classification Report and Confusion Matrix ---
    unique_labels = sorted(y.unique())
    plot_confusion_matrix_with_report(cm, classes=[str(label) for label in unique_labels], cr_text=cr_text)

    # --- Plot and Annotate the Predicted Probabilities Histogram ---
    plot_predicted_probabilities(clf, X_test)


if __name__ == "__main__":
    main()