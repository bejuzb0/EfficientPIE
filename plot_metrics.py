import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(csv_path, output_img=None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty.")
        return
    
    if df.empty:
        print(f"Error: CSV file '{csv_path}' has no data rows.")
        return
        
    epochs = df['epoch']
    
    # Set the style to something pleasant
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a 2x3 grid of subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Metrics: {os.path.basename(csv_path)}", fontsize=16)
    
    # 1. Loss
    axes[0, 0].plot(epochs, df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, df['val_loss'], label='Val Loss', marker='o', linewidth=2)
    axes[0, 0].set_title('Loss vs. Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 2. Accuracy
    axes[0, 1].plot(epochs, df['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(epochs, df['val_acc'], label='Val Acc', marker='o', linewidth=2)
    axes[0, 1].set_title('Accuracy vs. Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # 3. Precision
    axes[0, 2].plot(epochs, df['train_precision'], label='Train Precision', marker='o', linewidth=2)
    axes[0, 2].plot(epochs, df['val_precision'], label='Val Precision', marker='o', linewidth=2)
    axes[0, 2].set_title('Precision vs. Epochs')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()

    # 4. Recall
    axes[1, 0].plot(epochs, df['train_recall'], label='Train Recall', marker='o', linewidth=2)
    axes[1, 0].plot(epochs, df['val_recall'], label='Val Recall', marker='o', linewidth=2)
    axes[1, 0].set_title('Recall vs. Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()

    # 5. F1 Score
    axes[1, 1].plot(epochs, df['train_f1'], label='Train F1', marker='o', linewidth=2)
    axes[1, 1].plot(epochs, df['val_f1'], label='Val F1', marker='o', linewidth=2)
    axes[1, 1].set_title('F1 Score vs. Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()

    # 6. Learning Rate (Removed)
    fig.delaxes(axes[1, 2])
    for ax in axes.flat:
        if ax == axes[0, 1]:
            continue
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save chart
    if output_img is None:
        output_img = os.path.splitext(csv_path)[0] + "_plots.png"
    plt.savefig(output_img, dpi=300)
    print(f"Metrics plots successfully saved to: {output_img}")
    
    # Attempt to display the plot if a display is available
    try:
        plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output_img', type=str, default=None, help='Path for the generated plot image')
    args = parser.parse_args()
    
    plot_metrics(args.csv_path, args.output_img)
