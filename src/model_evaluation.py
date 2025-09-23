"""
Model Evaluation and Validation Framework
Author: Your Name
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and validation class
    """
    
    def __init__(self, model, X_train, y_train, X_test=None, y_test=None):
        """Initialize with trained model and data"""
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def evaluate_predictions(self):
        """Evaluate model predictions"""
        print("📊 MODEL EVALUATION METRICS")
        print("="*50)
        
        # Training predictions
        y_train_pred = self.model.predict(self.X_train)
        train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
        
        print("Training Set Performance:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Test predictions (if available)
        if self.X_test is not None and self.y_test is not None:
            y_test_pred = self.model.predict(self.X_test)
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)
            
            print("\nTest Set Performance:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            return train_metrics, test_metrics
        
        return train_metrics, None
    
    def plot_predictions(self):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training predictions
        y_train_pred = self.model.predict(self.X_train)
        
        axes[0].scatter(self.y_train, y_train_pred, alpha=0.6)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                    [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Training Set: Actual vs Predicted')
        
        # Test predictions (if available)
        if self.X_test is not None and self.y_test is not None:
            y_test_pred = self.model.predict(self.X_test)
            
            axes[1].scatter(self.y_test, y_test_pred, alpha=0.6)
            axes[1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1].set_xlabel('Actual Price')
            axes[1].set_ylabel('Predicted Price')
            axes[1].set_title('Test Set: Actual vs Predicted')
        else:
            axes[1].text(0.5, 0.5, 'No Test Data Available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Test Set: Not Available')
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self):
        """Plot residual analysis"""
        y_train_pred = self.model.predict(self.X_train)
        residuals = self.y_train - y_train_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0,0].scatter(y_train_pred, residuals, alpha=0.6)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted')
        
        # Residuals distribution
        axes[0,1].hist(residuals, bins=30, alpha=0.7)
        axes[0,1].set_xlabel('Residuals')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Residuals Distribution')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot of Residuals')
        
        # Residuals vs Actual
        axes[1,1].scatter(self.y_train, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Actual Values')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residuals vs Actual')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance (if supported by model)"""
        if hasattr(self.model, 'feature_importances_'):
            print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
            print("="*50)
            
            importances = self.model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else [f'Feature_{i}' for i in range(len(importances))]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print(importance_df.head(15).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(15)
            sns.barplot(data=top_features, y='Feature', x='Importance')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 STARTING MODEL EVALUATION")
        print("="*60)
        
        # Metrics evaluation
        metrics = self.evaluate_predictions()
        
        # Prediction plots
        self.plot_predictions()
        
        # Residual analysis
        self.plot_residuals()
        
        # Feature importance
        importance_df = self.feature_importance_analysis()
        
        print("\n✅ MODEL EVALUATION COMPLETE!")
        return metrics, importance_df