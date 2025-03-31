import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_importance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for LSTM classification models using
    permutation importance and other techniques.
    """
    
    def __init__(
        self,
        model_path,
        prediction_data_path,
        feature_names=None,
        output_dir='feature_analysis'
    ):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model_path: Path to the saved model (.h5 file)
            prediction_data_path: Path to the prediction data JSON
            feature_names: List of feature names (optional)
            output_dir: Directory to save analysis results
        """
        self.model_path = model_path
        self.prediction_data_path = prediction_data_path
        self.feature_names = feature_names
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and data
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        logger.info(f"Feature importance analyzer initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Prediction data path: {prediction_data_path}")
    
    def load_data(self):
        """
        Load the model and data directly from the source file.
        """
        try:
            # Load the model
            logger.info(f"Loading model from {self.model_path}")
            #self.model = load_model(self.model_path)
            
            # Load data file
            logger.info(f"Loading data from: {self.prediction_data_path}")
            
            # Determine file type and load accordingly
            if self.prediction_data_path.endswith('.json'):
                with open(self.prediction_data_path, 'r') as f:
                    prediction_data = json.load(f)
                
                # Convert lists back to numpy arrays
                self.X_test = np.array(prediction_data['X_test'])
                self.y_test = np.array(prediction_data['y_classes_test'])
                self.y_pred = np.array(prediction_data['y_pred_classes'])
            elif self.prediction_data_path.endswith('.pkl') or self.prediction_data_path.endswith('.pkl.gz'):
                data = pd.read_pickle(self.prediction_data_path)
            elif self.prediction_data_path.endswith('.csv') or self.prediction_data_path.endswith('.csv.gz'):
                data = pd.read_csv(self.prediction_data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.prediction_data_path}")
            
            # Process dataframe if not JSON
            if not self.prediction_data_path.endswith('.json'):
                logger.info(f"Data loaded successfully. Shape: {data.shape}")
                
                # Fix non-numeric columns
                for col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            logger.info(f"Converted column {col} to numeric")
                        except Exception as e:
                            logger.warning(f"Could not convert column {col} to numeric: {str(e)}. Column will be dropped.")
                            data = data.drop(columns=[col])
                
                # Get feature names if not provided
                if self.feature_names is None:
                    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                    self.feature_names = [col for col in numeric_columns if not any(
                        time_kw in col.lower() for time_kw in ['time', 'date', 'timestamp'])]
                    logger.info(f"Using {len(self.feature_names)} numeric features from dataset")
                
                # Drop columns with all missing values or nulls
                null_columns = data.columns[data.isnull().all()].tolist()
                if null_columns:
                    logger.info(f"Dropping {len(null_columns)} columns with all null values: {null_columns}")
                    data = data.drop(columns=null_columns)
                
                # Check for columns with all zeros or constant values
                constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
                if constant_columns:
                    logger.info(f"Dropping {len(constant_columns)} constant columns: {constant_columns}")
                    data = data.drop(columns=constant_columns)
                # Store features for later use
                self.data = data
            
                # Prepare target variable for analysis
                if not self.prediction_data_path.endswith('.json'):
                    # Try to identify the target column
                    target_candidates = ['close', 'target', 'price', 'label', 'class']
                    target_col = None
                    
                    # First check if any of the candidates exist
                    for candidate in target_candidates:
                        if candidate in data.columns:
                            target_col = candidate
                            break
                    
                    # If no candidate found, use the last column as target
                    if target_col is None:
                        target_col = data.columns[-1]
                        logger.info(f"No explicit target column found, using last column '{target_col}' as target")
                    else:
                        logger.info(f"Using '{target_col}' as target column")
                    
                    # Create a binary target column for classification analysis
                    # Calculate percentage change for the target
                    data['target_pct_change'] = data[target_col].pct_change()
                    
                    # Use a threshold of 0 (any increase is positive)
                    threshold = 0.0001
                    # Binary target: 0 = sell/no buy (price change below threshold), 1 = buy (price increase above threshold)
                    data['binary_target'] = (data['target_pct_change'] >= threshold).astype(int)
                    logger.info(f"Created binary target column with {data['binary_target'].sum()} positive samples out of {len(data)}")
                    logger.info(f"Positive class ratio: {data['binary_target'].mean():.2%}")
                    
                    # Drop rows with NaN in target (first row due to pct_change)
                    data = data.dropna(subset=['target_pct_change'])
                    logger.info(f"Dropped {len(data) - data.shape[0]} rows with NaN values in target")
            
            # Update feature names based on the data columns
            if self.feature_names is None or len(self.feature_names) != len(data.columns):
                # Filter out non-feature columns
                non_feature_cols = ['target_pct_change', 'binary_target']
                if target_col:
                    non_feature_cols.append(target_col)
                
                # Get actual feature columns from the data
                feature_cols = [col for col in data.columns if col not in non_feature_cols]
                
                # Update feature names
                self.feature_names = feature_cols
                logger.info(f"Updated feature names from data columns. Using {len(self.feature_names)} features.")
            
            # Store the feature data for later use
            self.X_test = data[self.feature_names].values
            if 'binary_target' in data.columns:
                self.y_test = data['binary_target'].values
            elif target_col:
                self.y_test = data[target_col].values
            else:
                logger.warning("No target column identified for testing")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def calculate_permutation_importance(self, n_repeats=10, random_state=42):
        """
        Calculate permutation feature importance.
        
        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating permutation importance...")
        
        if not hasattr(self, 'X_test') or self.X_test is None or not hasattr(self, 'y_test') or self.y_test is None:
            logger.error("No data available for permutation importance calculation")
            return pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Use the test data that was loaded in load_data method
        X = self.X_test
        y = self.y_test
        
        logger.warning("No model loaded. Using a simple RandomForestClassifier for permutation importance.")
        
        # Create a simple model for permutation importance since no model was loaded
        from sklearn.ensemble import RandomForestClassifier
        temp_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Fit the model on the data
        try:
            temp_model.fit(X, y)
            logger.info("Fitted temporary RandomForest model for permutation importance")
        except Exception as e:
            logger.error(f"Error fitting temporary model: {str(e)}")
            return pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Calculate permutation importance
        try:
            perm_importance = permutation_importance(
                temp_model, X, y,
                scoring='accuracy',
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
            
            # Create DataFrame with results
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Save to CSV
            importance_df.to_csv(f"{self.output_dir}/permutation_importance.csv", index=False)
            
            logger.info(f"Top 5 most important features: {importance_df['Feature'].head(5).tolist()}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(columns=['Feature', 'Importance'])
    
    def analyze_feature_correlations(self):
        """
        Analyze correlations between features and target variable.
        
        Returns:
            DataFrame with correlation scores
        """
        logger.info("Analyzing feature correlations...")
        
        # Calculate mean values for each feature across the time dimension

        # Create a DataFrame with mean feature values
        # Calculate mean values for each feature across the time dimension
        # Ensure proper reshaping based on the actual dimensions of X_test
        if self.X_test.ndim > 2:
            # For 3D data (samples, timesteps, features)
            feature_means = np.mean(self.X_test, axis=1)  # Average across time dimension
        else:
            # For 2D data (samples, features)
            feature_means = self.X_test.copy()
            
        # Verify dimensions match before creating DataFrame
        if feature_means.shape[1] != len(self.feature_names):
            logger.warning(f"Feature dimension mismatch: {feature_means.shape[1]} vs {len(self.feature_names)}")
            # Adjust feature_names if needed or reshape appropriately
        feature_df = pd.DataFrame(feature_means, columns=self.feature_names)
        
        # Add target variable
        feature_df['Target'] = self.y_test
        
        # Calculate correlations with target
        correlations = feature_df.corr()['Target'].drop('Target')
        
        # Create DataFrame with results
        correlation_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values,
            'Abs_Correlation': np.abs(correlations.values)
        })
        
        # Sort by absolute correlation
        correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)
        
        # Save to CSV
        correlation_df.to_csv(f"{self.output_dir}/feature_correlations.csv", index=False)
        
        logger.info(f"Top 5 correlated features: {correlation_df['Feature'].head(5).tolist()}")
        
        return correlation_df
    
    def plot_feature_importance(self, importance_df, top_n=10, title="Feature Importance"):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        plot_df = importance_df.head(top_n)
        
        # Create horizontal bar plot
        sns.barplot(x='Importance', y='Feature', data=plot_df, palette='viridis')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.output_dir}/{title.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()
    
    def plot_correlation_heatmap(self, top_n=15):
        """
        Plot correlation heatmap for top features.
        
        Args:
            top_n: Number of top features to include
        """
        logger.info(f"Plotting correlation heatmap for top {top_n} features...")
        
        # Calculate mean values for each feature across the time dimension
        feature_means = np.mean(self.X_test, axis=1)
        feature_df = pd.DataFrame(feature_means, columns=self.feature_names)
        
        # Add target variable
        feature_df['Target'] = self.y_test
        
        # Get correlations
        correlations = feature_df.corr()
        
        # Get top N correlated features with target
        top_features = feature_df.corr()['Target'].drop('Target').abs().sort_values(ascending=False).head(top_n).index
        
        # Include target in the heatmap
        top_features = list(top_features) + ['Target']
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlations.loc[top_features, top_features], dtype=bool))
        sns.heatmap(
            correlations.loc[top_features, top_features],
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5
        )
        
        plt.title(f"Correlation Heatmap (Top {top_n} Features)", fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300)
        plt.close()
    
    def analyze_with_tree_models(self):
        """
        Analyze feature importance using tree-based models (Random Forest and XGBoost).
        
        Returns:
            Dictionary with feature importance from different models
        """
        logger.info("Analyzing feature importance with tree-based models...")
        
        # Reshape data for tree models (flatten the time dimension)
        # We'll use the mean of each feature across the time dimension
        X_mean = np.mean(self.X_test, axis=1)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_mean, self.y_test)
        
        # Get Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf_model.feature_importances_
        })
        rf_importance = rf_importance.sort_values('Importance', ascending=False)
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_mean, self.y_test)
        
        # Get XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': xgb_model.feature_importances_
        })
        xgb_importance = xgb_importance.sort_values('Importance', ascending=False)
        
        # Save to CSV
        rf_importance.to_csv(f"{self.output_dir}/random_forest_importance.csv", index=False)
        xgb_importance.to_csv(f"{self.output_dir}/xgboost_importance.csv", index=False)
        
        # Plot Random Forest feature importance
        self.plot_feature_importance(
            rf_importance,
            title="Feature Importance (Random Forest)"
        )
        
        # Plot XGBoost feature importance
        self.plot_feature_importance(
            xgb_importance,
            title="Feature Importance (XGBoost)"
        )
        
        # Try to generate SHAP values for XGBoost (more advanced explanation)
        try:
            logger.info("Calculating SHAP values for XGBoost model...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_mean)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_mean, feature_names=self.feature_names, show=False)
            plt.title("SHAP Feature Importance", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/shap_summary.png", dpi=300)
            plt.close()
            
            # Plot detailed SHAP values for top features
            top_features = xgb_importance['Feature'].head(5).tolist()
            for feature in top_features:
                feature_idx = self.feature_names.index(feature)
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature_idx, 
                    shap_values, 
                    X_mean, 
                    feature_names=self.feature_names,
                    show=False
                )
                plt.title(f"SHAP Dependence Plot: {feature}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/shap_dependence_{feature.replace(' ', '_')}.png", dpi=300)
                plt.close()
            
            logger.info("SHAP analysis completed successfully")
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {str(e)}")
        
        logger.info(f"Top 5 features by Random Forest: {rf_importance['Feature'].head(5).tolist()}")
        logger.info(f"Top 5 features by XGBoost: {xgb_importance['Feature'].head(5).tolist()}")
        
        return {
            'random_forest': rf_importance,
            'xgboost': xgb_importance
        }
    
    def run_analysis(self):
        """
        Run the complete feature importance analysis.
        
        Returns:
            Dictionary with analysis results
        """
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Aborting analysis.")
            return None
        
        # Calculate permutation importance
        perm_importance = self.calculate_permutation_importance()
        
        # Analyze feature correlations
        correlations = self.analyze_feature_correlations()
        
        # Analyze with tree-based models
        tree_importance = self.analyze_with_tree_models()
        
        # Plot feature importance
        self.plot_feature_importance(
            perm_importance,
            title="Feature Importance (Permutation Method)"
        )
        
        # Plot correlation-based importance
        self.plot_feature_importance(
            correlations,
            title="Feature Importance (Correlation Method)"
        )
        
        # Plot correlation heatmap
        self.plot_correlation_heatmap()
        
        # Combine results
        top_features_permutation = perm_importance['Feature'].head(10).tolist()
        top_features_correlation = correlations['Feature'].head(10).tolist()
        top_features_rf = tree_importance['random_forest']['Feature'].head(10).tolist()
        top_features_xgb = tree_importance['xgboost']['Feature'].head(10).tolist()
        
        # Find common top features across all methods
        common_features = list(set(top_features_permutation[:5]) & 
                              set(top_features_correlation[:5]) & 
                              set(top_features_rf[:5]) & 
                              set(top_features_xgb[:5]))
        
        logger.info("===== Analysis Complete =====")
        logger.info(f"Top features by permutation importance: {top_features_permutation[:5]}")
        logger.info(f"Top features by correlation: {top_features_correlation[:5]}")
        logger.info(f"Top features by Random Forest: {top_features_rf[:5]}")
        logger.info(f"Top features by XGBoost: {top_features_xgb[:5]}")
        logger.info(f"Common top features across all methods: {common_features}")
        
        # Create summary report
        with open(f"{self.output_dir}/feature_importance_summary.txt", 'w') as f:
            f.write("===== Feature Importance Analysis Summary =====\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data: {self.prediction_data_path}\n\n")
            
            f.write("Top 10 Features by Permutation Importance:\n")
            for i, (feature, importance) in enumerate(zip(perm_importance['Feature'].head(10), 
                                                         perm_importance['Importance'].head(10))):
                f.write(f"{i+1}. {feature}: {importance:.4f}\n")
            
            f.write("\nTop 10 Features by Correlation:\n")
            for i, (feature, corr) in enumerate(zip(correlations['Feature'].head(10), 
                                                   correlations['Correlation'].head(10))):
                f.write(f"{i+1}. {feature}: {corr:.4f}\n")
            
            f.write("\nTop 10 Features by Random Forest:\n")
            for i, (feature, importance) in enumerate(zip(tree_importance['random_forest']['Feature'].head(10),
                                                         tree_importance['random_forest']['Importance'].head(10))):
                f.write(f"{i+1}. {feature}: {importance:.4f}\n")
            
            f.write("\nTop 10 Features by XGBoost:\n")
            for i, (feature, importance) in enumerate(zip(tree_importance['xgboost']['Feature'].head(10),
                                                         tree_importance['xgboost']['Importance'].head(10))):
                f.write(f"{i+1}. {feature}: {importance:.4f}\n")
            
            f.write("\nCommon Top Features Across All Methods:\n")
            for feature in common_features:
                f.write(f"- {feature}\n")
        
        return {
            'permutation_importance': perm_importance,
            'correlations': correlations,
            'random_forest': tree_importance['random_forest'],
            'xgboost': tree_importance['xgboost'],
            'top_features_permutation': top_features_permutation,
            'top_features_correlation': top_features_correlation,
            'top_features_rf': top_features_rf,
            'top_features_xgb': top_features_xgb,
            'common_features': common_features
        }


# Execute the analysis if run directly
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze feature importance for LSTM models')
    parser.add_argument('--model', type=str, default='checkpoints/lstm_binary_classification/final_classification_model.h5',
                        help='Path to the saved model (.h5 file)')
    parser.add_argument('--data', type=str, default='data\merged_features\merged_BTCUSDT_4h_5y.pkl.gz',
                        help='Path to the prediction data JSON')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to a text file with feature names (one per line)')
    parser.add_argument('--output', type=str, default='feature_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Load feature names if provided
    feature_names = None
    if args.features:
        try:
            with open(args.features, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(feature_names)} feature names from {args.features}")
        except Exception as e:
            logger.error(f"Error loading feature names: {str(e)}")
    
    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(
        model_path=args.model,
        prediction_data_path=args.data,
        feature_names=feature_names,
        output_dir=args.output
    )
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to {args.output}")
    else:
        logger.error("Analysis failed.") 