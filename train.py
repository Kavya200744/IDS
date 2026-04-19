"""
Training Script
Run this once to train and save models
"""

import os
import sys
import joblib
from config import Config
from preprocessing import DataPreprocessor
from models import IDSModel

def main():
    print("\n" + "="*70)
    print(" "*15 + "INTRUSION DETECTION SYSTEM")
    print(" "*20 + "Model Training")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Initialize preprocessor
    print("Step 1: Data Preprocessing")
    print("-" * 70)
    preprocessor = DataPreprocessor()
    
    # Load/Generate data
    df = preprocessor.load_sample_data(n_samples=Config.SAMPLE_SIZE)
    
    # Split data
    train_df, test_df = preprocessor.prepare_train_test_split(df, Config.TEST_SIZE)
    
    # Preprocess training data
    X_train, y_train = preprocessor.preprocess_pipeline(
        train_df, 
        fit=True, 
        apply_smote=True,
        feature_selection_method='hybrid'  # Use hybrid method (Correlation + Information Gain)
    )
    
    # Preprocess test data (no SMOTE)
    X_test, y_test = preprocessor.preprocess_pipeline(
        test_df, 
        fit=False, 
        apply_smote=False,
        feature_selection_method='hybrid'
    )
    
    print(f"\nTraining samples: {X_train.shape}")
    print(f"Testing samples:  {X_test.shape}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(Config.MODELS_DIR, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Initialize models
    print("\n" + "="*70)
    print("Step 2: Model Training")
    print("-" * 70)
    
    ids_model = IDSModel(Config)
    ids_model.initialize_models()
    
    # Train individual models
    ids_model.train_individual_models(X_train, y_train, X_test, y_test)
    
    # Create and train ensemble
    ids_model.create_ensemble_model()
    ids_model.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "="*70)
    print("Step 3: Results Summary")
    print("-" * 70)
    results_df = ids_model.get_results_dataframe()
    print(results_df.to_string(index=False))
    
    # Save models
    print("\n" + "="*70)
    print("Step 4: Saving Models")
    print("-" * 70)
    model_prefix = os.path.join(Config.MODELS_DIR, 'ids_model')
    ids_model.save_models(model_prefix)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python app.py")
    print("  2. Open browser: http://127.0.0.1:5000")
    print("  3. Use the web interface for detection\n")

if __name__ == "__main__":
    main()
