"""
Data Preprocessing and Feature Engineering for Housing Price Prediction
Author: Your Name
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class HousingPreprocessor:
    """
    Comprehensive preprocessing class for housing price prediction
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def missing_value_strategy(self, df):
        """Implement missing value treatment strategy"""
        print("🔧 MISSING VALUE TREATMENT")
        print("="*50)
        
        df_processed = df.copy()
        
        # Strategy 1: Features where NA means "None"
        none_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
        
        for feature in none_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna('None')
                print(f"  {feature}: Filled with 'None'")
        
        # Strategy 2: Numeric features - fill with 0 where logical
        zero_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
        
        for feature in zero_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(0)
                print(f"  {feature}: Filled with 0")
        
        # Strategy 3: Mode for categorical
        categorical_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 
                           'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 
                           'SaleType']
        
        for feature in categorical_mode:
            if feature in df_processed.columns:
                mode_val = df_processed[feature].mode()[0] if not df_processed[feature].mode().empty else 'Unknown'
                df_processed[feature] = df_processed[feature].fillna(mode_val)
                print(f"  {feature}: Filled with mode '{mode_val}'")
        
        # Strategy 4: Median for numeric
        numeric_median = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        
        for feature in numeric_median:
            if feature in df_processed.columns:
                median_val = df_processed[feature].median()
                df_processed[feature] = df_processed[feature].fillna(median_val)
                print(f"  {feature}: Filled with median {median_val}")
        
        return df_processed
    
    def outlier_treatment(self, df, target_col='SalePrice'):
        """Handle outliers using IQR method"""
        print("\n🎯 OUTLIER TREATMENT")
        print("="*50)
        
        df_processed = df.copy()
        
        if target_col in df_processed.columns:
            # Remove extreme price outliers
            Q1 = df_processed[target_col].quantile(0.25)
            Q3 = df_processed[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df_processed[target_col] < lower_bound) | (df_processed[target_col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            print(f"  Found {outliers_count} price outliers")
            print(f"  Keeping outliers for model robustness")
        
        # Handle feature outliers (cap extreme values)
        numeric_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'LotFrontage']
        
        for feature in numeric_features:
            if feature in df_processed.columns:
                Q1 = df_processed[feature].quantile(0.01)
                Q99 = df_processed[feature].quantile(0.99)
                df_processed[feature] = df_processed[feature].clip(lower=Q1, upper=Q99)
                print(f"  {feature}: Capped to 1st-99th percentile")
        
        return df_processed
    
    def feature_engineering(self, df):
        """Create new features"""
        print("\n🏗️ FEATURE ENGINEERING")
        print("="*50)
        
        df_processed = df.copy()
        
        # Total area features
        df_processed['TotalSF'] = (df_processed['TotalBsmtSF'] + 
                                  df_processed['1stFlrSF'] + 
                                  df_processed['2ndFlrSF'])
        print("  Created: TotalSF")
        
        # Total bathrooms
        df_processed['TotalBath'] = (df_processed['FullBath'] + 
                                    df_processed['HalfBath'] * 0.5 + 
                                    df_processed['BsmtFullBath'] + 
                                    df_processed['BsmtHalfBath'] * 0.5)
        print("  Created: TotalBath")
        
        # Age features
        df_processed['HouseAge'] = df_processed['YrSold'] - df_processed['YearBuilt']
        df_processed['RemodAge'] = df_processed['YrSold'] - df_processed['YearRemodAdd']
        print("  Created: HouseAge, RemodAge")
        
        # Quality scores
        df_processed['OverallScore'] = df_processed['OverallQual'] * df_processed['OverallCond']
        print("  Created: OverallScore")
        
        # Porch area
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        df_processed['TotalPorchSF'] = df_processed[porch_cols].sum(axis=1)
        print("  Created: TotalPorchSF")
        
        # Has feature flags
        df_processed['HasBasement'] = (df_processed['TotalBsmtSF'] > 0).astype(int)
        df_processed['HasGarage'] = (df_processed['GarageArea'] > 0).astype(int)
        df_processed['HasFireplace'] = (df_processed['Fireplaces'] > 0).astype(int)
        df_processed['HasPool'] = (df_processed['PoolArea'] > 0).astype(int)
        print("  Created: HasBasement, HasGarage, HasFireplace, HasPool")
        
        return df_processed
    
    def encode_features(self, df, target_col='SalePrice'):
        """Encode categorical features"""
        print("\n🔤 FEATURE ENCODING")
        print("="*50)
        
        df_processed = df.copy()
        
        # Ordinal encoding for quality features
        quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                           'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 
                           'GarageCond', 'PoolQC']
        
        for feature in quality_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].map(quality_map)
                print(f"  {feature}: Ordinal encoded")
        
        # Binary encoding for some features
        binary_map = {'N': 0, 'Y': 1}
        if 'CentralAir' in df_processed.columns:
            df_processed['CentralAir'] = df_processed['CentralAir'].map(binary_map)
            print("  CentralAir: Binary encoded")
        
        # One-hot encoding for remaining categorical features
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        if len(categorical_cols) > 0:
            df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
            print(f"  One-hot encoded {len(categorical_cols)} categorical features")
            return df_encoded
        
        return df_processed
    
    def scale_features(self, df, target_col='SalePrice'):
        """Scale numerical features"""
        print("\n📏 FEATURE SCALING")
        print("="*50)
        
        df_processed = df.copy()
        
        # Separate features and target
        if target_col in df_processed.columns:
            X = df_processed.drop([target_col, 'Id'], axis=1, errors='ignore')
            y = df_processed[target_col]
        else:
            X = df_processed.drop(['Id'], axis=1, errors='ignore')
            y = None
        
        # Scale numerical features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        self.scalers['standard'] = scaler
        self.feature_names = X_scaled.columns.tolist()
        
        print(f"  Scaled {len(numeric_features)} numerical features")
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def preprocess_pipeline(self, df, is_training=True):
        """Complete preprocessing pipeline"""
        print("🚀 STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Missing values
        df_processed = self.missing_value_strategy(df)
        
        # Step 2: Feature engineering
        df_processed = self.feature_engineering(df_processed)
        
        # Step 3: Outlier treatment
        df_processed = self.outlier_treatment(df_processed)
        
        # Step 4: Encoding
        df_processed = self.encode_features(df_processed)
        
        # Step 5: Scaling
        if is_training:
            X_scaled, y = self.scale_features(df_processed)
            print("\n✅ PREPROCESSING COMPLETE!")
            return X_scaled, y
        else:
            X_scaled = self.scale_features(df_processed, target_col=None)
            print("\n✅ PREPROCESSING COMPLETE!")
            return X_scaled

if __name__ == "__main__":
    # Test preprocessing
    train_df = pd.read_csv('../home-data-for-ml-course/train.csv')
    
    preprocessor = HousingPreprocessor()
    X_train, y_train = preprocessor.preprocess_pipeline(train_df, is_training=True)
    
    print(f"\nFinal training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")