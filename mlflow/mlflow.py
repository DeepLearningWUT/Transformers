"""
Enhanced MLflow decorator for speech commands classification.
Provides a clean interface for logging metrics, artifacts, audio samples,
learning curves, and model registration.
"""
import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch


def mlflow_run(config):
    """
    A decorator that wraps a training function with MLflow experiment and run handling.
    
    Args:
        config (dict): Configuration dictionary containing experiment settings
        
    Returns:
        Function: Decorated function that handles MLflow integration
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up experiment and create artifact directory
            mlflow.set_experiment(config["experiment_name"])
            os.makedirs(config["artifact_dir"], exist_ok=True)
            
            # Start MLflow run
            with mlflow.start_run(run_name=config.get("run_name")):
                # Log all parameters from config
                mlflow.log_params(config)
                
                # Call the wrapped function
                result = func(*args, **kwargs)
                
                # Register model if specified in config
                if "registered_model_name" in config and "best_model_path" in result:
                    mlflow.pytorch.log_model(
                        result["model"],
                        artifact_path="model",
                        registered_model_name=config["registered_model_name"]
                    )
                
            return result
        return wrapper
    return decorator


def log_metrics(metrics, step=None):
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary of metric names and values
        step (int, optional): Step number for the metrics
    """
    mlflow.log_metrics(metrics, step=step)


def log_best_model(model, epoch, artifact_dir, threshold_metrics=None):
    """
    Save and log model checkpoint as MLflow artifact.
    
    Args:
        model (torch.nn.Module): PyTorch model to save
        epoch (int): Current epoch number
        artifact_dir (str): Directory to save model checkpoint
        threshold_metrics (dict, optional): Metrics to determine if model should be registered
        
    Returns:
        str: Path to saved model checkpoint
    """
    checkpoint_path = os.path.join(artifact_dir, f"best_model_epoch_{epoch}.pth")
    # Save model state
    torch.save(model.state_dict(), checkpoint_path)
    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
    
    # Log model architecture
    mlflow.pytorch.log_model(model, artifact_path="model")
    
    return checkpoint_path


def log_confusion_matrix(y_true, y_pred, class_names, epoch, artifact_dir):
    """
    Create and log confusion matrix visualization.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): List of class names
        epoch (int): Current epoch
        artifact_dir (str): Directory to save visualization
    """
    from sklearn.metrics import confusion_matrix
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    # Save and log
    cm_path = os.path.join(artifact_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")


def log_learning_curves(train_metrics, val_metrics, artifact_dir):
    """
    Create and log learning curve visualizations.
    
    Args:
        train_metrics (dict): Dictionary of training metrics with lists of values per epoch
        val_metrics (dict): Dictionary of validation metrics with lists of values per epoch
        artifact_dir (str): Directory to save visualization
    """
    # Create a figure with multiple subplots (one per metric)
    metrics = list(train_metrics.keys())
    epochs = range(1, len(train_metrics[metrics[0]]) + 1)
    
    # Create subplot for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, train_metrics[metric], 'b-', label=f'Training {metric}')
        ax.plot(epochs, val_metrics[metric], 'r-', label=f'Validation {metric}')
        ax.set_title(f'{metric.capitalize()} Curves')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
    
    plt.tight_layout()
    curves_path = os.path.join(artifact_dir, 'learning_curves.png')
    plt.savefig(curves_path)
    plt.close()
    mlflow.log_artifact(curves_path, artifact_path="learning_curves")


def log_audio_samples(waveforms, predictions, true_labels, class_names, artifact_dir, sample_rate=16000):
    """
    Log audio sample files with predictions as MLflow artifacts.
    
    Args:
        waveforms (list): List of audio waveform arrays
        predictions (list): List of predicted label indices
        true_labels (list): List of true label indices
        class_names (list): List of class names
        artifact_dir (str): Directory to save audio files
        sample_rate (int): Audio sample rate in Hz
    """
    import scipy.io.wavfile as wav
    
    # Create directory for audio samples
    audio_dir = os.path.join(artifact_dir, "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_files = []
    for i, (audio, pred, true) in enumerate(zip(waveforms, predictions, true_labels)):
        # Convert to int16 format required by WAV
        audio_int = (audio * 32767).astype(np.int16)
        
        # Get class names for prediction and true label
        pred_class = class_names[pred]
        true_class = class_names[true]
        
        # Create descriptive filename
        filename = f"sample_{i}_pred_{pred_class}_true_{true_class}.wav"
        file_path = os.path.join(audio_dir, filename)
        
        # Save audio file
        wav.write(file_path, sample_rate, audio_int)
        audio_files.append(file_path)
    
    # Log each audio file 
    for file_path in audio_files:
        mlflow.log_artifact(file_path, artifact_path="audio_samples")


def log_attention_map(attention_weights, epoch, artifact_dir, title="Attention Map"):
    """
    Create and log attention map visualization.
    
    Args:
        attention_weights (numpy.ndarray): Attention weight matrix 
        epoch (int): Current epoch
        artifact_dir (str): Directory to save visualization
        title (str): Title for the visualization
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    
    # Save and log
    attn_path = os.path.join(artifact_dir, f"attention_map_epoch_{epoch}.png")
    plt.savefig(attn_path)
    plt.close()
    mlflow.log_artifact(attn_path, artifact_path="attention_maps")