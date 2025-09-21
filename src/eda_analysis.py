"""
Exploratory Data Analysis for Housing Price Prediction
Author: Your Name
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HousingEDA:
    """
    Comprehensive EDA class for housing price prediction project
    """
    
    def __init__(self, train_path, test_path=None):
        """Initialize with data paths"""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path) if test_path else None
        self.setup_plotting()
        
    def setup_plotting(self):
        """Set up plotting parameters"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("="*60)
        print("🏠 HOUSING PRICE PREDICTION - DATA OVERVIEW")
        print("="*60)
        
        print(f"Training data shape: {self.train_df.shape}")
        if self.test_df is not None:
            print(f"Test data shape: {self.test_df.shape}")
        print(f"Total features: {self.train_df.shape[1] - 1}")
        
        print("\n📊 DATA TYPES DISTRIBUTION:")
        print(self.train_df.dtypes.value_counts())
        
        print("\n💰 TARGET VARIABLE (SalePrice) STATISTICS:")
        target_stats = self.train_df['SalePrice'].describe()
        print(target_stats)
        
        return target_stats
    
    def missing_values_analysis(self):
        """Comprehensive missing values analysis"""
        print("\n" + "="*60)
        print("🔍 MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing = self.train_df.isnull().sum()
        missing_pct = (missing / len(self.train_df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        print(f"Total columns with missing values: {len(missing_df)}")
        print(f"Total missing values: {missing_df['Missing_Count'].sum()}")
        
        if len(missing_df) > 0:
            print("\nTop 10 columns with missing values:")
            print(missing_df.head(10).to_string(index=False))
            
            # Visualize missing values
            plt.figure(figsize=(12, 8))
            top_missing = missing_df.head(15)
            sns.barplot(data=top_missing, y='Column', x='Missing_Percentage')
            plt.title('Top 15 Features with Missing Values')
            plt.xlabel('Missing Percentage (%)')
            plt.tight_layout()
            plt.show()
        
        return missing_df
    
    def target_analysis(self):
        """Analyze the target variable (SalePrice)"""
        print("\n" + "="*60)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target = self.train_df['SalePrice']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution plot
        sns.histplot(target, kde=True, ax=axes[0,0])
        axes[0,0].set_title('SalePrice Distribution')
        axes[0,0].axvline(target.mean(), color='red', linestyle='--', label=f'Mean: ${target.mean():,.0f}')
        axes[0,0].axvline(target.median(), color='green', linestyle='--', label=f'Median: ${target.median():,.0f}')
        axes[0,0].legend()
        
        # Log distribution
        sns.histplot(np.log1p(target), kde=True, ax=axes[0,1])
        axes[0,1].set_title('Log(SalePrice) Distribution')
        
        # Box plot
        sns.boxplot(y=target, ax=axes[1,0])
        axes[1,0].set_title('SalePrice Box Plot')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot (Normal Distribution)')
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print(f"Skewness: {target.skew():.3f}")
        print(f"Kurtosis: {target.kurtosis():.3f}")
        
        # Price ranges
        print("\n💰 PRICE RANGES:")
        price_ranges = pd.cut(target, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        print(price_ranges.value_counts().sort_index())
        
        return target
    
    def correlation_analysis(self):
        """Analyze correlations with target variable"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Select only numeric columns
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        correlations = self.train_df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
        
        print("Top 15 features most correlated with SalePrice:")
        top_corr = correlations.head(16)[1:]  # Exclude SalePrice itself
        print(top_corr.to_string())
        
        # Visualize top correlations
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title('Top 15 Features Correlated with SalePrice')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def feature_analysis(self):
        """Analyze key features"""
        print("\n" + "="*60)
        print("🏗️ KEY FEATURES ANALYSIS")
        print("="*60)
        
        # Analyze categorical features
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns
        print(f"Number of categorical features: {len(categorical_cols)}")
        
        # High cardinality features
        high_cardinality = []
        for col in categorical_cols:
            unique_count = self.train_df[col].nunique()
            if unique_count > 10:
                high_cardinality.append((col, unique_count))
        
        if high_cardinality:
            print("\nHigh cardinality categorical features (>10 unique values):")
            for col, count in sorted(high_cardinality, key=lambda x: x[1], reverse=True):
                print(f"  {col}: {count} unique values")
        
        # Analyze some key features
        key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            if feature in self.train_df.columns:
                sns.scatterplot(data=self.train_df, x=feature, y='SalePrice', ax=axes[i])
                axes[i].set_title(f'SalePrice vs {feature}')
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_eda(self):
        """Run complete EDA analysis"""
        print("🚀 Starting Comprehensive EDA Analysis...")
        
        # Basic information
        self.basic_info()
        
        # Missing values
        missing_df = self.missing_values_analysis()
        
        # Target analysis
        target = self.target_analysis()
        
        # Correlation analysis
        correlations = self.correlation_analysis()
        
        # Feature analysis
        self.feature_analysis()
        
        print("\n" + "="*60)
        print("✅ EDA ANALYSIS COMPLETE!")
        print("="*60)
        
        return {
            'missing_values': missing_df,
            'target_stats': target.describe(),
            'correlations': correlations
        }

if __name__ == "__main__":
    # Run EDA
    eda = HousingEDA('../home-data-for-ml-course/train.csv', '../home-data-for-ml-course/test.csv')
    results = eda.run_complete_eda()
