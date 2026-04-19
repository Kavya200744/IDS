"""
Machine Learning Models Module
Ensemble IDS using Random Forest, XGBoost, and SVM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from xgboost import XGBClassifier
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class IDSModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.training_time = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize all ML models"""
        print("Initializing ML models...")
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            **self.config.RANDOM_FOREST_PARAMS
        )
        
        # XGBoost
        self.models['XGBoost'] = XGBClassifier(
            **self.config.XGBOOST_PARAMS,
            eval_metric='logloss'
        )
        
        # SVM (with probability for ensemble)
        self.models['SVM'] = SVC(
            **self.config.SVM_PARAMS,
            probability=True
        )
        
        # Decision Tree
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # KNN
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=2
        )
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=200,
            n_jobs=2,
            random_state=42
        )
        
        # Naive Bayes
        self.models['Naive Bayes'] = GaussianNB()
        
        print(f"Initialized {len(self.models)} models")
        
    def train_individual_models(self, X_train, y_train, X_test, y_test):
        """Train all individual models and evaluate"""
        print("\n" + "="*60)
        print("TRAINING INDIVIDUAL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Training
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time,
                'test_time': test_time,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
            
            self.training_time[name] = train_time
            
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            print(f"  Train Time: {train_time:.2f}s")
            print(f"  Test Time:  {test_time:.4f}s")
        
        print("\n" + "="*60)
        
    def create_ensemble_model(self):
        """Create voting ensemble of top 3 models"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Select top 3 models based on F1-score
        top_models = sorted(
            [(name, res['f1_score']) for name, res in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        print(f"\nTop 3 models selected:")
        for name, f1 in top_models:
            print(f"  {name}: F1={f1*100:.2f}%")
        
        # Create ensemble
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=2
        )
        
        print("\nEnsemble model created with soft voting")
        
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train and evaluate ensemble model"""
        print("\nTraining ensemble model...")
        
        # Training
        start_time = time.time()
        self.ensemble_model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = self.ensemble_model.predict(X_test)
        test_time = time.time() - start_time
        
        # Probability predictions
        y_proba = self.ensemble_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False positive/negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Store results
        self.results['Ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'fnr': fnr,
            'train_time': train_time,
            'test_time': test_time,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print("\n" + "="*60)
        print("ENSEMBLE MODEL RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"ROC-AUC:   {roc_auc*100:.2f}%")
        print(f"FPR:       {fpr*100:.2f}%")
        print(f"FNR:       {fnr*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"  TN={tn}, FP={fp}")
        print(f"  FN={fn}, TP={tp}")
        print(f"\nTrain Time: {train_time:.2f}s")
        print(f"Test Time:  {test_time:.4f}s")
        print("="*60 + "\n")
        
    def get_results_dataframe(self):
        """Get results as pandas DataFrame"""
        results_list = []
        
        for name, res in self.results.items():
            results_list.append({
                'Model': name,
                'Accuracy': f"{res['accuracy']*100:.2f}%",
                'Precision': f"{res['precision']*100:.2f}%",
                'Recall': f"{res['recall']*100:.2f}%",
                'F1-Score': f"{res['f1_score']*100:.2f}%",
                'Train Time (s)': f"{res['train_time']:.2f}",
                'Test Time (s)': f"{res['test_time']:.4f}"
            })
        
        return pd.DataFrame(results_list)
    
    def save_models(self, filepath_prefix):
        """Save trained models"""
        print(f"\nSaving models to {filepath_prefix}...")
        
        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f"{filepath_prefix}_ensemble.pkl")
            print("  ✓ Ensemble model saved")
        
        # Save best individual model
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['f1_score'])[0]
        if best_model_name != 'Ensemble':
            joblib.dump(self.models[best_model_name], 
                       f"{filepath_prefix}_best_individual.pkl")
            print(f"  ✓ Best individual model ({best_model_name}) saved")
        
        # Save results
        joblib.dump(self.results, f"{filepath_prefix}_results.pkl")
        print("  ✓ Results saved")
        
    def load_model(self, filepath):
        """Load trained model"""
        print(f"Loading model from {filepath}...")
        self.ensemble_model = joblib.load(filepath)
        print("  ✓ Model loaded successfully")
        
    def predict(self, X):
        """Make predictions using ensemble model"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.ensemble_model.predict(X)
        probabilities = self.ensemble_model.predict_proba(X)
        
        return predictions, probabilities
