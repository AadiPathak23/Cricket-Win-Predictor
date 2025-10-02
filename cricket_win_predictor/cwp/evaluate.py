"""
Model evaluation module for Cricket Win Predictor.
Handles model evaluation, metrics calculation, and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, learning_curve
import joblib


class ModelEvaluator:
    """Handles comprehensive model evaluation and analysis."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a trained model comprehensively.
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            model_name: Name of the model for identification
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model_name': model_name
        }
        
        # ROC AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                metrics['roc_auc'] = roc_auc
            except ValueError:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        return cv_results
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                             model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Team 2 Win', 'Team 1 Win'],
                   yticklabels=['Team 2 Win', 'Team 1 Win'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series,
                          model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot learning curve.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            save_path: Path to save the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models_results: Dictionary with model results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'ROC-AUC': results.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('Accuracy', ascending=False)
    
    def generate_evaluation_report(self, model_name: str, save_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model: {model_name}")
        
        results = self.evaluation_results[model_name]
        
        report = f"""
# Model Evaluation Report: {model_name}

## Performance Metrics
- **Accuracy**: {results['accuracy']:.4f}
- **Precision**: {results['precision']:.4f}
- **Recall**: {results['recall']:.4f}
- **F1-Score**: {results['f1_score']:.4f}
"""
        
        if results.get('roc_auc') is not None:
            report += f"- **ROC-AUC**: {results['roc_auc']:.4f}\n"
        
        report += f"""
## Confusion Matrix
{results['confusion_matrix']}

## Classification Report
{results['classification_report']}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save the results
        """
        joblib.dump(self.evaluation_results, filepath)
        print(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath: str) -> None:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to load the results from
        """
        self.evaluation_results = joblib.load(filepath)
        print(f"Evaluation results loaded from {filepath}")


class PredictionAnalyzer:
    """Analyzes prediction patterns and model behavior."""
    
    def __init__(self):
        """Initialize the prediction analyzer."""
        pass
    
    def analyze_prediction_confidence(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze prediction confidence distribution.
        
        Args:
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with confidence analysis
        """
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        analysis = {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'high_confidence_predictions': np.sum(confidence_scores > 0.8),
            'low_confidence_predictions': np.sum(confidence_scores < 0.6)
        }
        
        return analysis
    
    def analyze_prediction_errors(self, y_true: pd.Series, y_pred: np.ndarray,
                                 X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze prediction errors and identify patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X_test: Test feature matrix
            
        Returns:
            DataFrame with error analysis
        """
        # Identify misclassified samples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            return pd.DataFrame()
        
        # Create error analysis DataFrame
        error_analysis = X_test.iloc[error_indices].copy()
        error_analysis['true_label'] = y_true.iloc[error_indices]
        error_analysis['predicted_label'] = y_pred[error_indices]
        error_analysis['error_type'] = 'False Positive' if y_true.iloc[error_indices].iloc[0] == 0 else 'False Negative'
        
        return error_analysis

