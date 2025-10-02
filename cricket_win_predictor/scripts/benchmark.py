"""
Benchmarking script for Cricket Win Predictor.
Compares different models and evaluates their performance.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwp.data_loader import CricketDataLoader
from cwp.features import FeatureEngineer
from cwp.models import CricketWinPredictor, ModelManager
from cwp.evaluate import ModelEvaluator


class CricketWinPredictorBenchmark:
    """Benchmarking class for Cricket Win Predictor models."""
    
    def __init__(self, data_dir: str = 'datasets'):
        """Initialize the benchmark."""
        self.data_dir = data_dir
        self.data_loader = CricketDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        self.evaluator = ModelEvaluator()
        self.results = {}
        
    def load_data(self):
        """Load and preprocess data."""
        print("📊 Loading and preprocessing data...")
        
        # Load raw data
        match_data, match_info = self.data_loader.load_raw_data()
        processed_data = self.data_loader.create_processed_dataset()
        
        # Feature engineering
        X, y = self.feature_engineer.prepare_training_data(processed_data, 'match_result')
        
        print(f"✅ Loaded {len(processed_data)} matches with {X.shape[1]} features")
        
        return X, y
    
    def benchmark_models(self, X: pd.DataFrame, y: pd.Series, 
                        model_types: list = None) -> pd.DataFrame:
        """
        Benchmark multiple models.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_types: List of model types to benchmark
            
        Returns:
            DataFrame with benchmark results
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'ensemble']
        
        print(f"\n🏁 Benchmarking {len(model_types)} models...")
        
        benchmark_results = []
        
        for model_type in model_types:
            print(f"\n🤖 Training {model_type}...")
            
            # Record training time
            start_time = time.time()
            
            # Create and train model
            predictor = CricketWinPredictor(model_type=model_type)
            training_results = predictor.train(X, y, self.feature_engineer.feature_columns, 
                                             hyperparameter_tuning=False)
            
            training_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(predictor.model, X, y, cv=5, scoring='accuracy')
            
            # Record results
            result = {
                'model_type': model_type,
                'training_time': training_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_accuracy': training_results['training_accuracy'],
                'n_features': training_results['n_features']
            }
            
            benchmark_results.append(result)
            
            # Add to model manager
            self.model_manager.add_model(model_type, predictor)
            
            print(f"✅ {model_type} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"   Training Time: {training_time:.2f}s")
        
        return pd.DataFrame(benchmark_results)
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate models on test set.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n📊 Evaluating models on test set...")
        
        evaluation_results = []
        
        for model_name, model in self.model_manager.models.items():
            if model.is_trained:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Record results
                result = {
                    'model_type': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                evaluation_results.append(result)
                
                print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return pd.DataFrame(evaluation_results)
    
    def plot_benchmark_results(self, benchmark_results: pd.DataFrame, 
                             save_path: str = None) -> None:
        """
        Plot benchmark results.
        
        Args:
            benchmark_results: DataFrame with benchmark results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cross-validation scores
        axes[0, 0].bar(benchmark_results['model_type'], benchmark_results['cv_mean'])
        axes[0, 0].set_title('Cross-Validation Scores')
        axes[0, 0].set_ylabel('CV Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Training time
        axes[0, 1].bar(benchmark_results['model_type'], benchmark_results['training_time'])
        axes[0, 1].set_title('Training Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training accuracy
        axes[1, 0].bar(benchmark_results['model_type'], benchmark_results['training_accuracy'])
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Number of features
        axes[1, 1].bar(benchmark_results['model_type'], benchmark_results['n_features'])
        axes[1, 1].set_title('Number of Features')
        axes[1, 1].set_ylabel('Features')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, X: pd.DataFrame, y: pd.Series, 
                            save_path: str = None) -> None:
        """
        Plot learning curves for all models.
        
        Args:
            X: Feature matrix
            y: Target vector
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(self.model_manager.models.items()):
            if i >= len(axes):
                break
                
            if model.is_trained:
                # Calculate learning curve
                train_sizes, train_scores, val_scores = learning_curve(
                    model.model, X, y, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )
                
                # Plot learning curve
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                axes[i].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
                axes[i].fill_between(train_sizes, train_mean - train_std, 
                                   train_mean + train_std, alpha=0.1, color='blue')
                
                axes[i].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
                axes[i].fill_between(train_sizes, val_mean - val_std, 
                                   val_mean + val_std, alpha=0.1, color='red')
                
                axes[i].set_title(f'Learning Curve - {model_name}')
                axes[i].set_xlabel('Training Set Size')
                axes[i].set_ylabel('Score')
                axes[i].legend()
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(len(self.model_manager.models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_benchmark_report(self, benchmark_results: pd.DataFrame,
                                evaluation_results: pd.DataFrame,
                                save_path: str = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            benchmark_results: Benchmark results DataFrame
            evaluation_results: Evaluation results DataFrame
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        report = f"""
# Cricket Win Predictor - Benchmark Report

## Model Performance Comparison

### Cross-Validation Results
{benchmark_results[['model_type', 'cv_mean', 'cv_std', 'training_time']].to_string(index=False)}

### Test Set Evaluation
{evaluation_results[['model_type', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False)}

## Best Performing Models

### By Cross-Validation Score
{benchmark_results.nlargest(3, 'cv_mean')[['model_type', 'cv_mean', 'cv_std']].to_string(index=False)}

### By Test Accuracy
{evaluation_results.nlargest(3, 'accuracy')[['model_type', 'accuracy', 'f1_score']].to_string(index=False)}

### By Training Speed
{benchmark_results.nsmallest(3, 'training_time')[['model_type', 'training_time', 'cv_mean']].to_string(index=False)}

## Summary

The benchmark evaluated {len(benchmark_results)} different model types:
- Random Forest: {benchmark_results[benchmark_results['model_type'] == 'random_forest']['cv_mean'].iloc[0]:.4f} CV score
- Gradient Boosting: {benchmark_results[benchmark_results['model_type'] == 'gradient_boosting']['cv_mean'].iloc[0]:.4f} CV score
- Logistic Regression: {benchmark_results[benchmark_results['model_type'] == 'logistic_regression']['cv_mean'].iloc[0]:.4f} CV score
- SVM: {benchmark_results[benchmark_results['model_type'] == 'svm']['cv_mean'].iloc[0]:.4f} CV score
- Ensemble: {benchmark_results[benchmark_results['model_type'] == 'ensemble']['cv_mean'].iloc[0]:.4f} CV score

## Recommendations

1. **Best Overall Performance**: {evaluation_results.loc[evaluation_results['accuracy'].idxmax(), 'model_type']}
2. **Fastest Training**: {benchmark_results.loc[benchmark_results['training_time'].idxmin(), 'model_type']}
3. **Most Balanced**: {evaluation_results.loc[evaluation_results['f1_score'].idxmax(), 'model_type']}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def run_full_benchmark(self, save_results: bool = True) -> dict:
        """
        Run a complete benchmark evaluation.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all benchmark results
        """
        print("🏏 Cricket Win Predictor - Full Benchmark")
        print("=" * 50)
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Benchmark models
        benchmark_results = self.benchmark_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # Generate plots
        if save_results:
            self.plot_benchmark_results(benchmark_results, 'models/benchmark_results.png')
            self.plot_learning_curves(X_train, y_train, 'models/learning_curves.png')
        
        # Generate report
        report = self.generate_benchmark_report(benchmark_results, evaluation_results, 
                                             'models/benchmark_report.md' if save_results else None)
        
        # Store results
        self.results = {
            'benchmark_results': benchmark_results,
            'evaluation_results': evaluation_results,
            'report': report
        }
        
        print("\n📊 Benchmark Results:")
        print(benchmark_results[['model_type', 'cv_mean', 'training_time']].to_string(index=False))
        
        print("\n📈 Evaluation Results:")
        print(evaluation_results[['model_type', 'accuracy', 'f1_score']].to_string(index=False))
        
        if save_results:
            print(f"\n💾 Results saved to models/ directory")
        
        return self.results


def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Cricket Win Predictor Models')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing the dataset files')
    parser.add_argument('--save_results', action='store_true',
                       help='Save benchmark results to files')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'ensemble'],
                       help='Specific models to benchmark')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = CricketWinPredictorBenchmark(args.data_dir)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(save_results=args.save_results)
    
    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()

