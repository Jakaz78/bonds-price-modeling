import os
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

def ensure_plots_directory():
    """Create plots directory if it doesn't exist"""
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Save plot to Plots directory"""
    plots_dir = ensure_plots_directory()
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved: {filepath}")


def plot_correlation_heatmap(data):
    """Create and save correlation heatmap"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Selected Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_plot("correlation_heatmap.png")
    plt.show()

def plot_actual_vs_predicted(Y_test, predictions):
    """Create and save actual vs predicted comparison"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(Y_test.values, color='black', linewidth=2, label='Actual', alpha=0.8)
    plt.plot(predictions.values, color='red', linewidth=2, label='Predicted', alpha=0.8)
    plt.title('Actual vs Predicted Values (Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('D_CLOSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals plot
    test_residuals = Y_test - predictions
    plt.subplot(2, 1, 2)
    plt.plot(test_residuals, color='blue', linewidth=1, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Prediction Residuals (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_plot("actual_vs_predicted.png")
    plt.show()

