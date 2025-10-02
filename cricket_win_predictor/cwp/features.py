"""
Feature engineering module for Cricket Win Predictor.
Handles advanced feature creation and feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


class FeatureEngineer:
    """Handles feature engineering for cricket match prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from the basic dataset.
        
        Args:
            data: Raw match data DataFrame
            
        Returns:
            DataFrame with advanced features
        """
        df = data.copy()
        
        # Team strength difference
        df['team_strength_diff'] = df['team1_strength'] - df['team2_strength']
        
        # Venue advantage (how much above/below average the venue is)
        df['venue_advantage_1st'] = df['venue_avg_1st'] - 250  # 250 is global average
        df['venue_advantage_2nd'] = df['venue_avg_2nd'] - 220  # 220 is global average
        
        # Combined venue factor
        df['venue_factor'] = (df['venue_avg_1st'] + df['venue_avg_2nd']) / 2
        
        # Team form (simplified - could be enhanced with recent matches)
        df['team1_form'] = df['team1_strength'] / 100
        df['team2_form'] = df['team2_strength'] / 100
        
        # Match pressure indicators
        df['wicket_pressure'] = df['team2_wickets'] / 10  # Higher wickets = more pressure
        df['run_rate_pressure'] = df['required_run_rate'] - df['current_run_rate']
        
        # Venue-specific team performance (simplified)
        df['venue_team1_factor'] = df['team1_strength'] * (df['venue_avg_1st'] / 250)
        df['venue_team2_factor'] = df['team2_strength'] * (df['venue_avg_2nd'] / 220)
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            data: DataFrame with basic features
            
        Returns:
            DataFrame with interaction features
        """
        df = data.copy()
        
        # Team strength interactions
        df['strength_product'] = df['team1_strength'] * df['team2_strength']
        df['strength_ratio'] = df['team1_strength'] / (df['team2_strength'] + 1e-8)
        
        # Venue-team interactions
        df['team1_venue_advantage'] = df['team1_strength'] * df['venue_avg_1st']
        df['team2_venue_advantage'] = df['team2_strength'] * df['venue_avg_2nd']
        
        # Pressure-performance interactions
        df['pressure_performance'] = df['run_rate_pressure'] * df['wicket_pressure']
        
        return df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from the data.
        
        Args:
            data: DataFrame with basic features
            
        Returns:
            DataFrame with statistical features
        """
        df = data.copy()
        
        # Rolling averages (if we had time series data)
        # For now, we'll create some statistical measures
        
        # Team performance consistency (standard deviation of strength)
        df['team1_consistency'] = 1.0  # Placeholder - would need historical data
        df['team2_consistency'] = 1.0  # Placeholder - would need historical data
        
        # Venue difficulty (how much the venue affects scoring)
        df['venue_difficulty'] = abs(df['venue_avg_1st'] - df['venue_avg_2nd'])
        
        # Match competitiveness
        df['competitiveness'] = 1 - abs(df['team_strength_diff']) / 100
        
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            data: DataFrame with categorical features
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = data.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str], fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data: DataFrame with features
            feature_columns: List of feature column names to scale
            fit_scaler: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        df = data.copy()
        
        # Select only the feature columns that exist
        existing_features = [col for col in feature_columns if col in df.columns]
        
        if fit_scaler:
            df[existing_features] = self.scaler.fit_transform(df[existing_features])
        else:
            df[existing_features] = self.scaler.transform(df[existing_features])
        
        return df
    
    def select_features(self, data: pd.DataFrame, feature_importance: Dict[str, float], 
                       threshold: float = 0.01) -> List[str]:
        """
        Select features based on importance scores.
        
        Args:
            data: DataFrame with features
            feature_importance: Dictionary mapping feature names to importance scores
            threshold: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        selected_features = [
            feature for feature, importance in feature_importance.items()
            if importance >= threshold and feature in data.columns
        ]
        
        return selected_features
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in the pipeline.
        
        Args:
            data: Raw match data DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        # Create advanced features
        df = self.create_advanced_features(data)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create statistical features
        df = self.create_statistical_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def get_feature_importance_mapping(self) -> Dict[str, float]:
        """
        Get feature importance mapping based on domain knowledge.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {
            'team1_strength': 0.25,
            'team2_strength': 0.25,
            'team_strength_diff': 0.20,
            'venue_avg_1st': 0.15,
            'venue_avg_2nd': 0.15,
            'current_run_rate': 0.10,
            'required_run_rate': 0.10,
            'wicket_pressure': 0.08,
            'run_rate_pressure': 0.08,
            'venue_factor': 0.12,
            'competitiveness': 0.06,
            'strength_ratio': 0.05
        }
    
    def prepare_training_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            data: DataFrame with all features
            target_column: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        # Create all features
        df = self.create_all_features(data)
        
        # Get feature importance mapping
        feature_importance = self.get_feature_importance_mapping()
        
        # Select important features
        feature_columns = self.select_features(df, feature_importance)
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scale_features(X, feature_columns, fit_scaler=True)
        
        self.feature_columns = feature_columns
        
        return X_scaled, y
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction (without target).
        
        Args:
            data: DataFrame with features
            
        Returns:
            Prepared DataFrame for prediction
        """
        # Create all features
        df = self.create_all_features(data)
        
        # Select the same features used in training
        if not self.feature_columns:
            feature_importance = self.get_feature_importance_mapping()
            self.feature_columns = self.select_features(df, feature_importance)
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Scale features using existing scaler
        X_scaled = self.scale_features(X, self.feature_columns, fit_scaler=False)
        
        return X_scaled

