"""
Model Development Pipeline for Housing Price Prediction
Author: Your Name
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class HousingModelDevelopment:
    """
    Comprehensive model development and comparison class
    """
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        """Initialize with preprocessed data"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def create_baseline_model(self):
        """Create simple baseline model"""
        print("🎯 CREATING BASELINE MODEL")
        print("="*50)
        
        # Simple Linear Regression baseline
        baseline = LinearRegression()
        baseline.fit(self.X_train, self.y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(baseline, self.X_train, self.y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        self.models['Baseline_LinearRegression'] = baseline
        self.results['Baseline_LinearRegression'] = {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'model': baseline
        }
        
        print(f"  Baseline RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std()*2:.4f})")
        return baseline
    
    def setup_model_candidates(self):
        """Setup all model candidates for comparison"""
        print("\n🏗️ SETTING UP MODEL CANDIDATES")
        print("="*50)
        
        models = {
            # Linear Models
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            
            # Tree-based Models
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        }
        
        print(f"  Setup {len(models)} model candidates")
        return models
    
    def compare_models(self, cv_folds=5):
        """Compare all models using cross-validation"""
        print("\n🔄 MODEL COMPARISON")
        print("="*50)
        
        models = self.setup_model_candidates()
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                       cv=cv_folds, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'model': model
            }
            
            print(f"    RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std()*2:.4f})")
        
        # Display results summary
        self.display_model_comparison()
        
    def display_model_comparison(self):
        """Display model comparison results"""
        print("\n📊 MODEL COMPARISON RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_RMSE_Mean': [self.results[model]['cv_rmse_mean'] for model in self.results.keys()],
            'CV_RMSE_Std': [self.results[model]['cv_rmse_std'] for model in self.results.keys()]
        })
        
        results_df = results_df.sort_values('CV_RMSE_Mean')
        print(results_df.to_string(index=False))
        
        best_model = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        
        return results_df
    
    def optimize_best_models(self, top_n=3):
        """Hyperparameter optimization for top models"""
        print(f"\n⚡ HYPERPARAMETER OPTIMIZATION (Top {top_n})")
        print("="*60)
        
        # Get top models
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_RMSE_Mean': [self.results[model]['cv_rmse_mean'] for model in self.results.keys()]
        }).sort_values('CV_RMSE_Mean')
        
        top_models = results_df.head(top_n)['Model'].tolist()
        
        # Parameter grids
        param_grids = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        optimized_results = {}
        
        for model_name in top_models:
            if model_name in param_grids:
                print(f"  Optimizing {model_name}...")
                
                model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                optimized_results[f"{model_name}_Optimized"] = {
                    'cv_rmse_mean': np.sqrt(-grid_search.best_score_),
                    'best_params': grid_search.best_params_,
                    'model': grid_search.best_estimator_
                }
                
                print(f"    Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
                print(f"    Best params: {grid_search.best_params_}")
        
        self.results.update(optimized_results)
        return optimized_results
    
    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = min(self.results.keys(), 
                             key=lambda x: self.results[x]['cv_rmse_mean'])
        best_model = self.results[best_model_name]['model']
        
        print(f"\n🏆 FINAL BEST MODEL: {best_model_name}")
        print(f"   RMSE: {self.results[best_model_name]['cv_rmse_mean']:.4f}")
        
        return best_model, best_model_name
    
    def run_complete_pipeline(self):
        """Run complete model development pipeline"""
        print("🚀 STARTING MODEL DEVELOPMENT PIPELINE")
        print("="*70)
        
        # Step 1: Baseline
        self.create_baseline_model()
        
        # Step 2: Model comparison
        self.compare_models()
        
        # Step 3: Optimization
        self.optimize_best_models()
        
        # Step 4: Final selection
        best_model, best_name = self.get_best_model()
        
        print("\n✅ MODEL DEVELOPMENT COMPLETE!")
        return best_model, best_name, self.results

if __name__ == "__main__":
    # Example usage with preprocessed data
    from data_preprocessing import HousingPreprocessor
    
    # Load and preprocess data
    train_df = pd.read_csv('../home-data-for-ml-course/train.csv')
    preprocessor = HousingPreprocessor()
    X_train, y_train = preprocessor.preprocess_pipeline(train_df)
    
    # Run model development
    model_dev = HousingModelDevelopment(X_train, y_train)
    best_model, best_name, results = model_dev.run_complete_pipeline()