"""
Flask Web Application for Intrusion Detection System
Modern UI with real-time detection
"""

from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from config import Config
from preprocessing import DataPreprocessor
import json
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

# Global variables
preprocessor = None
model = None
detection_history = []

def load_models():
    """Load trained models"""
    global preprocessor, model
    
    try:
        preprocessor_path = os.path.join(Config.MODELS_DIR, 'preprocessor.pkl')
        model_path = os.path.join(Config.MODELS_DIR, 'ids_model_ensemble.pkl')
        
        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            return False
        
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Real-time detection endpoint"""
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train models first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Preprocess
        X, _ = preprocessor.preprocess_pipeline(
            df, 
            fit=False, 
            apply_smote=False,
            feature_selection_method='hybrid'
        )
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Extract single prediction
        prediction = predictions[0]
        probability = probabilities[0]
        
        # Get result
        is_attack = bool(prediction)
        confidence = float(probability[1] if is_attack else probability[0])
        
        result = {
            'success': True,
            'prediction': 'Attack' if is_attack else 'Normal',
            'confidence': round(confidence * 100, 2),
            'is_attack': is_attack,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to history
        detection_history.append(result)
        if len(detection_history) > 100:
            detection_history.pop(0)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR in single detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch-detect', methods=['POST'])
def batch_detect():
    """Batch detection for uploaded files"""
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train models first.'
            }), 500
        
        # Get file data
        file = request.files.get('file')
        if not file:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Store original data
        original_df = df.copy()
        
        # Preprocess
        X, _ = preprocessor.preprocess_pipeline(
            df, 
            fit=False, 
            apply_smote=False,
            feature_selection_method='hybrid'
        )
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add results to original dataframe
        original_df['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in predictions]
        original_df['Confidence'] = [round(prob[1]*100, 2) if pred == 1 else round(prob[0]*100, 2) 
                                     for pred, prob in zip(predictions, probabilities)]
        
        # Statistics
        total = len(predictions)
        attacks = int(np.sum(predictions))
        normal = total - attacks
        
        result = {
            'success': True,
            'total': total,
            'attacks': attacks,
            'normal': normal,
            'attack_percentage': round((attacks / total) * 100, 2),
            'results': original_df.to_dict('records')[:100]  # Limit to 100 for display
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/history')
def history():
    """Get detection history"""
    return jsonify({
        'success': True,
        'history': detection_history[-50:]  # Last 50 detections
    })

@app.route('/stats')
def stats():
    """Get system statistics"""
    try:
        results_path = os.path.join(Config.MODELS_DIR, 'ids_model_results.pkl')
        
        if not os.path.exists(results_path):
            return jsonify({
                'success': False,
                'error': 'No training results found'
            })
        
        results = joblib.load(results_path)
        
        # Format results for display
        stats_data = []
        for name, res in results.items():
            stats_data.append({
                'model': name,
                'accuracy': round(res['accuracy'] * 100, 2),
                'precision': round(res['precision'] * 100, 2),
                'recall': round(res['recall'] * 100, 2),
                'f1_score': round(res['f1_score'] * 100, 2),
                'train_time': round(res['train_time'], 2)
            })
        
        return jsonify({
            'success': True,
            'stats': stats_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = model is not None and preprocessor is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    # Create required directories
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.STATIC_DIR, exist_ok=True)
    os.makedirs(Config.TEMPLATES_DIR, exist_ok=True)
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n" + "="*70)
        print("WARNING: Models not found!")
        print("="*70)
        print("Please run training first:")
        print("  python train.py")
        print("="*70 + "\n")
    
    # Run app
    print("\n" + "="*70)
    print(" "*20 + "IDS WEB APPLICATION")
    print("="*70)
    print(f"Server running at: http://{Config.HOST}:{Config.PORT}")
    print("Press CTRL+C to stop")
    print("="*70 + "\n")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
