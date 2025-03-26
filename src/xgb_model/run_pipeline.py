import logging
import pandas as pd
import yaml
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import gc
import time
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import xgboost

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('xgb_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Import local modules
from src.xgb_model.preprocess import XGBTimeSeriesPreprocessor
from src.xgb_model.model import MultiStepXGBModel

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(data_dir: Path, config: dict, force_preprocess: bool = False) -> dict:
    """
    Load cached processed data or generate new data
    """
    processed_dir = data_dir
    processed_dir.mkdir(exist_ok=True)
    
    data_file = processed_dir / 'merged_BTCUSDT_4h_5y.pkl.gz'
    

    logger.info("Preprocessing data...")
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = XGBTimeSeriesPreprocessor(
        data_path=str(data_dir / config['data']['filename']),
        target_col=config['data']['target_column'],
        window_size=config['data']['window_size'],
        future_steps=config['data']['future_offset'],
        threshold=config['data']['threshold'],
        feature_selection=config['data'].get('feature_selection'),
        scaler_type=config['data'].get('scaler_type', 'robust'),
        timeframe=config['data'].get('timeframe')
    )
    
    # Load and preprocess data
    data_splits = preprocessor.preprocess()
    
    # Save processed data
    with open(data_file, 'wb') as f:
        import pickle
        pickle.dump(data_splits, f, protocol=4)
    
    logger.info(f"Data preprocessed and saved in {time.time() - start_time:.2f} seconds")
    return data_splits

def optimize_memory():
    """Free up memory"""
    gc.collect()
    if torch_available:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def plot_confusion_matrices(y_true, y_pred, checkpoint_dir):
    """Plot confusion matrices for each prediction step"""
    future_steps = y_pred.shape[1]
    
    for step in range(future_steps):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true[:, step], y_pred[:, step])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Step t+{step+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{checkpoint_dir}/confusion_matrix_step_{step+1}.png")
        plt.close()

def main():
    """Main pipeline function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run XGBoost pipeline')
    parser.add_argument('--config', type=str, default='src/xgb_model/configs/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/merged_features',
                        help='Directory containing data files')
    parser.add_argument('--force_preprocess', default=True, action='store_true',
                        help='Force preprocessing even if cached data exists')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / f"run_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Save config to checkpoint directory
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Check for XGBoost
    try:
        import xgboost
        logger.info(f"Using XGBoost version {xgboost.__version__}")
    except ImportError:
        logger.error("XGBoost not installed. Please install it with 'pip install xgboost'")
        return
    
    # Check for torch (optional, for memory optimization)
    global torch_available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    try:
        # Start MLflow run
        mlflow.set_experiment("xgboost_multistep")
        with mlflow.start_run(run_name=f"xgb_run_{timestamp}"):
            # Log parameters
            mlflow.log_params({
                "model_n_estimators": config['model']['n_estimators'],
                "model_max_depth": config['model']['max_depth'],
                "model_learning_rate": config['model']['learning_rate'],
                "window_size": config['data']['window_size'],
                "future_steps": config['data']['future_offset'],
                "threshold": config['data']['threshold']
            })
            
            # Prepare data
            data_dir = Path(args.data_dir)
            data_splits = prepare_data(data_dir, config, args.force_preprocess)
            
            # Extract data
            X_train, y_train = data_splits['X_train'], data_splits['y_train']
            X_val, y_val = data_splits['X_val'], data_splits['y_val']
            X_test, y_test = data_splits['X_test'], data_splits['y_test']
            feature_names = data_splits['feature_names']
            
            logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
            logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
            logger.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
            
            # Initialize model
            model = MultiStepXGBModel(config, feature_names)
            
            # Train model
            logger.info("Training model...")
            start_time = time.time()
            histories = model.train(X_train, y_train, X_val, y_val)
            training_time = time.time() - start_time
            logger.info(f"Model trained in {training_time:.2f} seconds")
            
            # Save model
            model_path = str(checkpoint_dir / "xgb_model")
            model.save(model_path)
            mlflow.log_artifact(f"{model_path}_metadata.pkl")
            for step in range(config['data']['future_offset']):
                mlflow.log_artifact(f"{model_path}_step_{step+1}.json")
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = model.evaluate(X_test, y_test)
            
            # Log metrics
            for step, step_metrics in metrics.items():
                if step != 'average':
                    mlflow.log_metrics({
                        f"accuracy_step_{step}": step_metrics['accuracy'],
                        f"precision_step_{step}": step_metrics['precision'],
                        f"recall_step_{step}": step_metrics['recall'],
                        f"f1_step_{step}": step_metrics['f1'],
                        f"directional_accuracy_step_{step}": step_metrics['directional_accuracy']
                    })
            
            # Log average metrics
            mlflow.log_metrics({
                "avg_accuracy": metrics['average']['accuracy'],
                "avg_precision": metrics['average']['precision'],
                "avg_recall": metrics['average']['recall'],
                "avg_f1": metrics['average']['f1'],
                "avg_directional_accuracy": metrics['average']['directional_accuracy'],
                "training_time": training_time
            })
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Plot confusion matrices
            plot_confusion_matrices(y_test, y_pred, str(checkpoint_dir))
            for step in range(config['data']['future_offset']):
                confusion_matrix_path = checkpoint_dir / f"confusion_matrix_step_{step+1}.png"
                if confusion_matrix_path.exists():
                    mlflow.log_artifact(str(confusion_matrix_path))
                else:
                    logger.warning(f"Confusion matrix file not found: {confusion_matrix_path}")
                
                # Check if feature importance plot exists before logging
                feature_importance_path = checkpoint_dir / f"feature_importance_step_{step+1}.png"
                if feature_importance_path.exists():
                    mlflow.log_artifact(str(feature_importance_path))
                else:
                    logger.warning(f"Feature importance plot not found: {feature_importance_path}")
                    
                    # Generate feature importance plot if it doesn't exist
                    try:
                        if step < len(model.models):
                            # Create feature importance plot
                            plt.figure(figsize=(12, 8))
                            xgb.plot_importance(model.models[step], max_num_features=20, importance_type='gain')
                            plt.title(f'Feature Importance for Step t+{step+1}')
                            plt.tight_layout()
                            plt.savefig(str(feature_importance_path))
                            plt.close()
                            
                            # Log the newly created plot
                            if feature_importance_path.exists():
                                mlflow.log_artifact(str(feature_importance_path))
                                logger.info(f"Generated and logged feature importance plot for step {step+1}")
                    except Exception as e:
                        logger.error(f"Error generating feature importance plot for step {step+1}: {str(e)}")
            
            # Save predictions
            predictions_df = pd.DataFrame()
            for step in range(config['data']['future_offset']):
                predictions_df[f'true_step_{step+1}'] = y_test[:, step]
                predictions_df[f'pred_step_{step+1}'] = y_pred[:, step]
            
            predictions_path = checkpoint_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))
            
            logger.info("Pipeline completed successfully")
            logger.info(f"Average accuracy: {metrics['average']['accuracy']:.4f}")
            logger.info(f"Average directional accuracy: {metrics['average']['directional_accuracy']:.4f}")
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
    finally:
        # Clean up
        optimize_memory()

if __name__ == "__main__":
    main() 