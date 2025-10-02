"""
Machine learning models module for Cricket Win Predictor.
Handles model training, prediction, and model management.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


class CricketWinPredictor:
    """Main class for cricket match win prediction using machine learning."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
    def _create_model(self) -> Any:
        """
        Create the machine learning model based on the specified type.
        
        Returns:
            Initialized model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            return SVC(
                probability=True,
                random_state=42
            )
        elif self.model_type == 'ensemble':
            # Create ensemble of multiple models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              feature_columns: List[str] = None,
              hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the machine learning model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_columns: List of feature column names
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        if feature_columns:
            self.feature_columns = feature_columns
        
        # Create model
        self.model = self._create_model()
        
        # Hyperparameter tuning
        if hyperparameter_tuning and self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print("Performing hyperparameter tuning...")
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train the model
        print(f"Training {self.model_type} model...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_score = self.model.score(X, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        results = {
            'model_type': self.model_type,
            'training_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns)
        }
        
        print(f"Training completed!")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            print("Feature importance not available for this model type.")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def predict_match(self, team1_strength: float, team2_strength: float,
                     venue_avg_1st: float, venue_avg_2nd: float,
                     team1_wickets: int, team2_wickets: int,
                     current_run_rate: float, required_run_rate: float) -> Dict[str, float]:
        """
        Predict match outcome for specific inputs.
        
        Args:
            team1_strength: Team 1 strength score
            team2_strength: Team 2 strength score
            venue_avg_1st: Venue average for 1st innings
            venue_avg_2nd: Venue average for 2nd innings
            team1_wickets: Team 1 wickets lost
            team2_wickets: Team 2 wickets lost
            current_run_rate: Current run rate
            required_run_rate: Required run rate
            
        Returns:
            Dictionary with prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'team1_strength': team1_strength,
            'team2_strength': team2_strength,
            'venue_avg_1st': venue_avg_1st,
            'venue_avg_2nd': venue_avg_2nd,
            'team1_wickets': team1_wickets,
            'team2_wickets': team2_wickets,
            'current_run_rate': current_run_rate,
            'required_run_rate': required_run_rate
        }])
        
        # Add derived features
        input_data['team_strength_diff'] = team1_strength - team2_strength
        input_data['venue_factor'] = (venue_avg_1st + venue_avg_2nd) / 2
        input_data['wicket_pressure'] = team2_wickets / 10
        input_data['run_rate_pressure'] = required_run_rate - current_run_rate
        input_data['strength_ratio'] = team1_strength / (team2_strength + 1e-8)
        input_data['competitiveness'] = 1 - abs(input_data['team_strength_diff']) / 100
        
        # Ensure we have all required features
        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0.0
        
        # Select only the features used in training
        X = input_data[self.feature_columns]
        
        # Make prediction
        probabilities = self.predict_proba(X)
        
        return {
            'team1_win_probability': probabilities[0][1],
            'team2_win_probability': probabilities[0][0],
            'prediction': 1 if probabilities[0][1] > 0.5 else 0
        }


class ModelManager:
    """Manages multiple models and model comparison."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model: CricketWinPredictor) -> None:
        """
        Add a model to the manager.
        
        Args:
            name: Name for the model
            model: Trained model instance
        """
        self.models[name] = model
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare performance of all models.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for name, model in self.models.items():
            if model.is_trained:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                comparison_results.append({
                    'model': name,
                    'accuracy': accuracy,
                    'model_type': model.model_type
                })
        
        return pd.DataFrame(comparison_results).sort_values('accuracy', ascending=False)
    
    def get_best_model(self) -> Tuple[str, CricketWinPredictor]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.results:
            raise ValueError("No model results available. Run compare_models() first.")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.models[best_model_name]

