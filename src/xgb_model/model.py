import xgboost as xgb
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import pickle

# Configure logging
logger = logging.getLogger(__name__)

class MultiStepXGBModel:
    """
    Multi-step XGBoost model for time series prediction.
    Trains separate models for each future time step.
    """
    
    def __init__(self, config: Dict, feature_names: Optional[List[str]] = None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
            feature_names: Names of features (for feature importance)
        """
        self.config = config
        self.feature_names = feature_names
        self.models = []  # List to store models for each future step
        self.num_classes = 3  # Default for up/down/neutral
        self.future_steps = config['data']['future_offset']
        
        logger.info(f"Initialized XGBoost model for {self.future_steps} future steps")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray) -> List[Dict]:
        """
        Train XGBoost models for each future step.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            List of training histories for each model
        """
        histories = []
        
        # Train a separate model for each future step
        for step in range(self.future_steps):
            logger.info(f"Training model for step t+{step+1}")
            
            # Extract targets for this step
            y_train_step = y_train[:, step]
            y_val_step = y_val[:, step]
            
            # Create DMatrix objects for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train_step, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val_step, feature_names=self.feature_names)
            
            # Set XGBoost parameters
            params = {
                'objective': self.config['model']['objective'],
                'eval_metric': self.config['model']['eval_metric'],
                'max_depth': self.config['model']['max_depth'],
                'eta': self.config['model']['learning_rate'],
                'subsample': self.config['model']['subsample'],
                'colsample_bytree': self.config['model']['colsample_bytree'],
                'min_child_weight': self.config['model']['min_child_weight'],
                'gamma': self.config['model']['gamma'],
                'alpha': self.config['model']['reg_alpha'],
                'lambda': self.config['model']['reg_lambda'],
                'verbosity': self.config['model']['verbosity'],
                'num_class': self.num_classes
            }
            
            # Train the model
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config['model']['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=self.config['model']['early_stopping_rounds'],
                evals_result=evals_result,
                verbose_eval=10
            )
            
            # Store the model
            self.models.append(model)
            
            # Store training history
            histories.append({
                'step': step + 1,
                'best_iteration': model.best_iteration,
                'best_score': model.best_score,
                'train_history': evals_result['train'],
                'val_history': evals_result['val']
            })
            
            logger.info(f"Model for step t+{step+1} trained. Best iteration: {model.best_iteration}, "
                       f"Best score: {model.best_score}")
            
            # Plot feature importance
            if self.config['training']['feature_importance_plot']:
                self._plot_feature_importance(model, step)
        
        return histories
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for all future steps.
        
        Args:
            X: Input features
            
        Returns:
            Predictions for all future steps
        """
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Make predictions for each step
        predictions = []
        for step, model in enumerate(self.models):
            # Get probability predictions
            probs = model.predict(dmatrix)
            
            # Reshape if needed
            if len(probs.shape) > 1:
                # Multi-class case - get class with highest probability
                preds = np.argmax(probs, axis=1)
            else:
                # Binary case
                preds = (probs > 0.5).astype(int)
            
            predictions.append(preds)
        
        # Stack predictions into a single array
        return np.column_stack(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics for each step
        metrics = {}
        for step in range(self.future_steps):
            step_metrics = {}
            
            # Extract predictions and targets for this step
            y_pred_step = y_pred[:, step]
            y_test_step = y_test[:, step]
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_step, y_pred_step)
            step_metrics['accuracy'] = accuracy
            
            # Calculate precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_step, y_pred_step, average='weighted'
            )
            step_metrics['precision'] = precision
            step_metrics['recall'] = recall
            step_metrics['f1'] = f1
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test_step, y_pred_step)
            step_metrics['confusion_matrix'] = cm
            
            # Calculate directional accuracy (for financial time series)
            directional_accuracy = self._calculate_directional_accuracy(y_test_step, y_pred_step)
            step_metrics['directional_accuracy'] = directional_accuracy
            
            # Store metrics for this step
            metrics[f'step_{step+1}'] = step_metrics
            
            logger.info(f"Step t+{step+1} metrics: Accuracy={accuracy:.4f}, "
                       f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
                       f"Directional Accuracy={directional_accuracy:.4f}")
        
        # Calculate average metrics across all steps
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics.values()]),
            'precision': np.mean([m['precision'] for m in metrics.values()]),
            'recall': np.mean([m['recall'] for m in metrics.values()]),
            'f1': np.mean([m['f1'] for m in metrics.values()]),
            'directional_accuracy': np.mean([m['directional_accuracy'] for m in metrics.values()])
        }
        metrics['average'] = avg_metrics
        
        logger.info(f"Average metrics across all steps: Accuracy={avg_metrics['accuracy']:.4f}, "
                   f"Precision={avg_metrics['precision']:.4f}, Recall={avg_metrics['recall']:.4f}, "
                   f"F1={avg_metrics['f1']:.4f}, Directional Accuracy={avg_metrics['directional_accuracy']:.4f}")
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy for financial predictions.
        
        Args:
            y_true: True labels (0=down, 1=neutral, 2=up)
            y_pred: Predicted labels
            
        Returns:
            Directional accuracy score
        """
        # For directional accuracy, we only care if the direction is correct
        # Class 0 = down, Class 2 = up, Class 1 = neutral
        
        # Filter out neutral predictions and actual values
        non_neutral_indices = (y_true != 1) & (y_pred != 1)
        
        if sum(non_neutral_indices) == 0:
            return 0.0  # No non-neutral predictions
        
        # Get directional predictions and actual values
        y_true_dir = y_true[non_neutral_indices]
        y_pred_dir = y_pred[non_neutral_indices]
        
        # Calculate accuracy
        return accuracy_score(y_true_dir, y_pred_dir)
    
    def _plot_feature_importance(self, model: xgb.Booster, step: int) -> None:
        """
        Plot feature importance for a model.
        
        Args:
            model: Trained XGBoost model
            step: Future step index
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
            
            # Get feature importance
            importance = model.get_score(importance_type='gain')
            
            # Check if importance is empty
            if not importance:
                logger.warning(f"No feature importance available for step t+{step+1}")
                return
            
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            })
            importance_df = importance_df.sort_values('Importance', ascending=False).head(30)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Feature Importance for Step t+{step+1}')
            plt.tight_layout()
            
            # Save plot with proper path handling
            output_path = os.path.join(self.config['training']['checkpoint_dir'], 
                                      f"feature_importance_step_{step+1}.png")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Feature importance plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Error plotting feature importance for step {step+1}: {str(e)}")
    
    def save(self, path: str) -> None:
        """
        Save the trained models.
        
        Args:
            path: Path to save the models
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save each model
        for step, model in enumerate(self.models):
            model_path = f"{path}_step_{step+1}.json"
            model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'future_steps': self.future_steps,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names
        }
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Models saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load trained models.
        
        Args:
            path: Path to load the models from
        """
        # Load metadata
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.future_steps = metadata['future_steps']
        self.num_classes = metadata['num_classes']
        self.feature_names = metadata['feature_names']
        
        # Load each model
        self.models = []
        for step in range(self.future_steps):
            model_path = f"{path}_step_{step+1}.json"
            model = xgb.Booster()
            model.load_model(model_path)
            self.models.append(model)
        
        logger.info(f"Loaded {len(self.models)} models from {path}") 