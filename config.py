# Configuration for IDS System
# Optimized for Intel i3 processors

import os

class Config:
    # Project paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    
    # Model parameters (optimized for i3)
    # Reduced complexity for faster execution
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 50,  # Reduced from 100 for faster training
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': 2,  # i3 has 2 cores (4 threads)
        'random_state': 42
    }
    
    XGBOOST_PARAMS = {
        'n_estimators': 50,
        'max_depth': 6,
        'learning_rate': 0.1,
        'nthread': 2,  # Use 2 threads for i3
        'random_state': 42
    }
    
    SVM_PARAMS = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'cache_size': 500,  # Reduced cache for low memory
        'random_state': 42
    }
    
    # Dataset configuration
    SAMPLE_SIZE = 10000  # Use 10k samples for faster training on i3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Feature selection
    FEATURE_SELECTION_METHOD = 'hybrid'  # Options: 'correlation', 'information_gain', 'hybrid', 'none'
    CORRELATION_THRESHOLD = 0.3
    TOP_FEATURES = 22  # Reduced from 49 features
    
    # HYBRID METHOD (Novel Approach):
    # Step 1: Correlation-based selection (threshold > 0.3)
    # Step 2: Information Gain on correlation-selected features
    # Result: 49 → 22 features (54% reduction)
    
    # Flask configuration
    SECRET_KEY = 'your-secret-key-here-change-in-production'
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000
    
    # Performance settings
    MAX_WORKERS = 2  # Limit concurrent operations
    CACHE_ENABLED = True
    
    # Alert settings
    ALERT_THRESHOLD = 0.8  # Confidence threshold for alerts
