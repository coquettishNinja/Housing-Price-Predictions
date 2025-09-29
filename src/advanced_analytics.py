"""
Advanced Analytics and Model Interpretability for Housing Price Prediction
Author: Yanni Qu
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Handle optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available. Install with: pip install shap>=0.41.0")
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly>=5.10.0")
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None

import warnings
warnings.filterwarnings('ignore')

class AdvancedHousingAnalytics:
    """
    Advanced analytics and interpretability for housing price models
    """
    
    def __init__(self, model, X_train, y_train, X_test=None, y_test=None, feature_names=None):
        """Initialize with trained model and data"""
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None)
        self.shap_explainer = None
        self.shap_values = None
        
    def feature_importance_analysis(self):
        """Comprehensive feature importance analysis"""
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        importance_results = {}
        
        # 1. Built-in feature importance (for tree models)
        if hasattr(self.model, 'feature_importances_'):
            importance_results['builtin'] = self.model.feature_importances_
            print("✅ Built-in feature importance extracted")
        
        # 2. Permutation importance
        print("🔄 Computing permutation importance...")
        perm_importance = permutation_importance(
            self.model, self.X_train, self.y_train, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        importance_results['permutation'] = perm_importance.importances_mean
        print("✅ Permutation importance computed")
        
        # 3. Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Permutation_Importance': importance_results['permutation']
        })
        
        if 'builtin' in importance_results:
            importance_df['Builtin_Importance'] = importance_results['builtin']
        
        importance_df = importance_df.sort_values('Permutation_Importance', ascending=False)
        
        # 4. Visualize top features
        self._plot_feature_importance(importance_df)
        
        return importance_df
    
    def _plot_feature_importance(self, importance_df, top_n=20):
        """Plot feature importance"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Permutation importance
        top_features = importance_df.head(top_n)
        sns.barplot(data=top_features, y='Feature', x='Permutation_Importance', ax=axes[0])
        axes[0].set_title(f'Top {top_n} Features - Permutation Importance')
        axes[0].set_xlabel('Importance Score')
        
        # Built-in importance (if available)
        if 'Builtin_Importance' in importance_df.columns:
            sns.barplot(data=top_features, y='Feature', x='Builtin_Importance', ax=axes[1])
            axes[1].set_title(f'Top {top_n} Features - Built-in Importance')
            axes[1].set_xlabel('Importance Score')
        else:
            axes[1].text(0.5, 0.5, 'Built-in importance\nnot available for this model', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Built-in Importance - Not Available')
        
        plt.tight_layout()
        plt.show()
    
    def shap_analysis(self, sample_size=100):
        """SHAP (SHapley Additive exPlanations) analysis"""
        print("\n🔍 SHAP ANALYSIS")
        print("="*60)
        
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available. Install with: pip install shap>=0.41.0")
            return None
        
        # Sample data for SHAP (can be computationally expensive)
        if len(self.X_train) > sample_size:
            sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train.iloc[sample_idx] if hasattr(self.X_train, 'iloc') else self.X_train[sample_idx]
        else:
            X_sample = self.X_train
        
        print(f"🔄 Computing SHAP values for {len(X_sample)} samples...")
        
        # Create SHAP explainer
        try:
            # Try TreeExplainer first (faster for tree models)
            if hasattr(self.model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("✅ Using TreeExplainer")
            else:
                # Fallback to general explainer
                self.shap_explainer = shap.Explainer(self.model, X_sample)
                print("✅ Using general Explainer")
            
            # Calculate SHAP values
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            print("✅ SHAP values computed")
            
            # Generate SHAP plots
            self._create_shap_plots(X_sample)
            
        except Exception as e:
            print(f"❌ SHAP analysis failed: {str(e)}")
            print("💡 Continuing with other analyses...")
        
        return self.shap_values
    
    def _create_shap_plots(self, X_sample):
        """Create SHAP visualization plots"""
        print("📊 Creating SHAP visualizations...")
        
        # 1. Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Predictions')
        plt.tight_layout()
        plt.show()
        
        # 2. Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def market_segmentation_analysis(self):
        """Analyze market segments based on predictions and features"""
        print("\n🏘️ MARKET SEGMENTATION ANALYSIS")
        print("="*60)
        
        # Get predictions
        y_pred = self.model.predict(self.X_train)
        
        # Create price segments
        price_segments = pd.cut(y_pred, bins=5, labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury'])
        
        # Analyze segments
        segment_analysis = pd.DataFrame({
            'Predicted_Price': y_pred,
            'Actual_Price': self.y_train,
            'Segment': price_segments
        })
        
        # Add key features for analysis
        key_features = ['OverallQual', 'GrLivArea', 'GarageCars'] if hasattr(self.X_train, 'columns') else [0, 1, 2]
        for i, feature in enumerate(key_features[:3]):  # Top 3 features
            if hasattr(self.X_train, 'columns') and feature in self.X_train.columns:
                segment_analysis[feature] = self.X_train[feature]
            elif not hasattr(self.X_train, 'columns') and i < self.X_train.shape[1]:
                segment_analysis[f'Feature_{i}'] = self.X_train[:, i]
        
        # Segment statistics
        segment_stats = segment_analysis.groupby('Segment').agg({
            'Predicted_Price': ['mean', 'std', 'count'],
            'Actual_Price': ['mean', 'std']
        }).round(2)
        
        print("📊 Market Segment Statistics:")
        print(segment_stats)
        
        # Visualize segments
        self._plot_market_segments(segment_analysis)
        
        return segment_analysis, segment_stats
    
    def _plot_market_segments(self, segment_analysis):
        """Plot market segmentation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Price distribution by segment
        sns.boxplot(data=segment_analysis, x='Segment', y='Predicted_Price', ax=axes[0,0])
        axes[0,0].set_title('Price Distribution by Market Segment')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Actual vs Predicted by segment
        sns.scatterplot(data=segment_analysis, x='Actual_Price', y='Predicted_Price', 
                       hue='Segment', ax=axes[0,1])
        axes[0,1].plot([segment_analysis['Actual_Price'].min(), segment_analysis['Actual_Price'].max()],
                      [segment_analysis['Actual_Price'].min(), segment_analysis['Actual_Price'].max()], 
                      'r--', alpha=0.8)
        axes[0,1].set_title('Actual vs Predicted Price by Segment')
        
        # 3. Segment counts
        segment_counts = segment_analysis['Segment'].value_counts()
        axes[1,0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Market Segment Distribution')
        
        # 4. Feature analysis by segment (if available)
        numeric_cols = segment_analysis.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:  # Has features beyond price columns
            feature_col = numeric_cols[3]  # First feature column
            sns.boxplot(data=segment_analysis, x='Segment', y=feature_col, ax=axes[1,1])
            axes[1,1].set_title(f'{feature_col} by Market Segment')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def business_recommendations(self, importance_df, segment_stats):
        """Generate business recommendations"""
        print("\n💼 BUSINESS RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # 1. Top value drivers
        top_features = importance_df.head(5)['Feature'].tolist()
        recommendations.append(f"🎯 Focus on top value drivers: {', '.join(top_features[:3])}")
        
        # 2. Market segment insights
        if segment_stats is not None:
            luxury_count = segment_stats.loc['Luxury', ('Predicted_Price', 'count')] if 'Luxury' in segment_stats.index else 0
            total_count = segment_stats[('Predicted_Price', 'count')].sum()
            luxury_pct = (luxury_count / total_count) * 100
            
            recommendations.append(f"🏘️ Luxury market represents {luxury_pct:.1f}% of properties")
            
            # Price ranges
            budget_avg = segment_stats.loc['Budget', ('Predicted_Price', 'mean')] if 'Budget' in segment_stats.index else 0
            luxury_avg = segment_stats.loc['Luxury', ('Predicted_Price', 'mean')] if 'Luxury' in segment_stats.index else 0
            
            if luxury_avg > 0 and budget_avg > 0:
                price_multiplier = luxury_avg / budget_avg
                recommendations.append(f"💰 Luxury properties command {price_multiplier:.1f}x premium over budget homes")
        
        # 3. Feature-based recommendations
        if 'OverallQual' in top_features:
            recommendations.append("🏗️ Overall quality is crucial - invest in high-quality materials and finishes")
        
        if 'GrLivArea' in top_features:
            recommendations.append("📐 Living area size strongly impacts value - maximize usable space")
        
        if any('Garage' in feature for feature in top_features):
            recommendations.append("🚗 Garage features are important - ensure adequate parking")
        
        # Display recommendations
        print("\n📋 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return recommendations
    
    def run_complete_analytics(self):
        """Run complete advanced analytics pipeline"""
        print("🚀 STARTING ADVANCED ANALYTICS PIPELINE")
        print("="*70)
        
        results = {}
        
        # 1. Feature importance
        importance_df = self.feature_importance_analysis()
        results['feature_importance'] = importance_df
        
        # 2. SHAP analysis
        shap_values = self.shap_analysis()
        results['shap_values'] = shap_values
        
        # 3. Market segmentation
        segment_analysis, segment_stats = self.market_segmentation_analysis()
        results['market_segments'] = {'analysis': segment_analysis, 'stats': segment_stats}
        
        # 4. Business recommendations
        recommendations = self.business_recommendations(importance_df, segment_stats)
        results['recommendations'] = recommendations
        
        print("\n✅ ADVANCED ANALYTICS COMPLETE!")
        print("="*70)
        
        return results

if __name__ == "__main__":
    # Example usage - commented out since data files may not be available
    print("AdvancedHousingAnalytics class is ready to use!")
    print("Import this class in your main script with your trained model and data.")
    
    # Uncomment and modify the following when you have your data and trained model:
    # from model_development import HousingModelDevelopment
    # from data_preprocessing import HousingPreprocessor
    # 
    # # Load and preprocess data
    # train_df = pd.read_csv('../home-data-for-ml-course/train.csv')
    # preprocessor = HousingPreprocessor()
    # X_train, y_train = preprocessor.preprocess_pipeline(train_df)
    # 
    # # Train best model
    # model_dev = HousingModelDevelopment(X_train, y_train)
    # best_model, best_name, _ = model_dev.run_complete_pipeline()
    # 
    # # Run advanced analytics
    # analytics = AdvancedHousingAnalytics(best_model, X_train, y_train, feature_names=X_train.columns.tolist())
    # results = analytics.run_complete_analytics()
