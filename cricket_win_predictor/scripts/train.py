"""
Training script for Cricket Win Predictor.
Handles model training, validation, and saving.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwp.data_loader import CricketDataLoader
from cwp.features import FeatureEngineer
from cwp.models import CricketWinPredictor, ModelManager
from cwp.evaluate import ModelEvaluator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Cricket Win Predictor Model')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing the dataset files')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'ensemble'],
                       help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--save_model', type=str, default='models/cricket_win_predictor_model.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--save_results', type=str, default='models/training_results.pkl',
                       help='Path to save training results')
    
    args = parser.parse_args()
    
    print("🏏 Cricket Win Predictor - Model Training")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    
    try:
        # Step 1: Load and preprocess data
        print("\n📊 Loading and preprocessing data...")
        data_loader = CricketDataLoader(args.data_dir)
        match_data, match_info = data_loader.load_raw_data()
        processed_data = data_loader.create_processed_dataset()
        
        print(f"✅ Loaded {len(processed_data)} matches")
        
        # Step 2: Feature engineering
        print("\n🔧 Engineering features...")
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.prepare_training_data(processed_data, 'match_result')
        
        print(f"✅ Created {X.shape[1]} features")
        
        # Step 3: Split data
        print(f"\n📈 Splitting data (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Step 4: Train model
        print(f"\n🤖 Training {args.model_type} model...")
        predictor = CricketWinPredictor(model_type=args.model_type)
        
        training_results = predictor.train(
            X_train, y_train,
            feature_columns=feature_engineer.feature_columns,
            hyperparameter_tuning=args.hyperparameter_tuning
        )
        
        # Step 5: Evaluate model
        print("\n📊 Evaluating model...")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(predictor.model, X_test, y_test, args.model_type)
        
        # Step 6: Save model and results
        print(f"\n💾 Saving model to {args.save_model}...")
        predictor.save_model(args.save_model)
        
        # Save training results
        results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'feature_columns': feature_engineer.feature_columns,
            'model_type': args.model_type,
            'test_size': args.test_size,
            'random_state': args.random_state
        }
        
        import joblib
        joblib.dump(results, args.save_results)
        
        # Step 7: Display results
        print("\n📈 Training Results:")
        print(f"Model Type: {args.model_type}")
        print(f"Training Accuracy: {training_results['training_accuracy']:.4f}")
        print(f"Cross-validation Score: {training_results['cv_mean']:.4f} (+/- {training_results['cv_std']*2:.4f})")
        print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Test Precision: {evaluation_results['precision']:.4f}")
        print(f"Test Recall: {evaluation_results['recall']:.4f}")
        print(f"Test F1-Score: {evaluation_results['f1_score']:.4f}")
        
        if evaluation_results.get('roc_auc'):
            print(f"Test ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        
        # Feature importance
        if hasattr(predictor.model, 'feature_importances_'):
            print("\n🔍 Top 5 Most Important Features:")
            feature_importance = predictor.get_feature_importance()
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                print(f"{i+1}. {feature}: {importance:.4f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"Model saved to: {args.save_model}")
        print(f"Results saved to: {args.save_results}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)


def train_multiple_models():
    """Train multiple models and compare their performance."""
    print("🏏 Cricket Win Predictor - Multi-Model Training")
    print("=" * 50)
    
    # Load data
    data_loader = CricketDataLoader('datasets')
    match_data, match_info = data_loader.load_raw_data()
    processed_data = data_loader.create_processed_dataset()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_training_data(processed_data, 'match_result')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train multiple models
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression', 'ensemble']
    model_manager = ModelManager()
    evaluator = ModelEvaluator()
    
    for model_type in model_types:
        print(f"\n🤖 Training {model_type} model...")
        
        predictor = CricketWinPredictor(model_type=model_type)
        predictor.train(X_train, y_train, feature_engineer.feature_columns, hyperparameter_tuning=False)
        
        # Evaluate model
        evaluation_results = evaluator.evaluate_model(predictor.model, X_test, y_test, model_type)
        
        # Add to model manager
        model_manager.add_model(model_type, predictor)
        
        print(f"✅ {model_type} - Accuracy: {evaluation_results['accuracy']:.4f}")
    
    # Compare models
    print("\n📊 Model Comparison:")
    comparison_results = model_manager.compare_models(X_test, y_test)
    print(comparison_results)
    
    # Get best model
    best_model_name, best_model = model_manager.get_best_model()
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Save best model
    best_model.save_model(f"models/best_model_{best_model_name}.pkl")
    print(f"💾 Best model saved to: models/best_model_{best_model_name}.pkl")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--multi':
        train_multiple_models()
    else:
        main()

