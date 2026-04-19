"""
Data Preprocessing Module
Optimized for Intel i3 processors
With Hybrid Feature Selection (Correlation + Information Gain)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.feature_names = None
        
    def load_sample_data(self, n_samples=10000):
        """
        Generate sample UNSW-NB15 like data for demonstration
        In production, replace with actual dataset loading
        """
        print(f"Generating {n_samples} sample records...")
        
        np.random.seed(42)
        
        # Simulate network traffic features
        data = {
            'dur': np.random.exponential(2, n_samples),
            'spkts': np.random.poisson(10, n_samples),
            'dpkts': np.random.poisson(8, n_samples),
            'sbytes': np.random.exponential(500, n_samples),
            'dbytes': np.random.exponential(400, n_samples),
            'rate': np.random.uniform(0, 1000, n_samples),
            'sttl': np.random.randint(0, 255, n_samples),
            'dttl': np.random.randint(0, 255, n_samples),
            'sload': np.random.exponential(100, n_samples),
            'dload': np.random.exponential(80, n_samples),
            'sloss': np.random.poisson(2, n_samples),
            'dloss': np.random.poisson(1, n_samples),
            'sinpkt': np.random.exponential(0.5, n_samples),
            'dinpkt': np.random.exponential(0.4, n_samples),
            'sjit': np.random.exponential(10, n_samples),
            'djit': np.random.exponential(8, n_samples),
            'swin': np.random.randint(0, 65535, n_samples),
            'stcpb': np.random.randint(0, 4294967295, n_samples, dtype=np.int64),
            'dtcpb': np.random.randint(0, 4294967295, n_samples, dtype=np.int64),
            'dwin': np.random.randint(0, 65535, n_samples),
            'tcprtt': np.random.exponential(50, n_samples),
            'synack': np.random.exponential(20, n_samples),
            'ackdat': np.random.exponential(30, n_samples),
        }
        
        # Categorical features
        data['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
        data['service'] = np.random.choice(['http', 'ftp', 'ssh', 'dns', '-'], n_samples)
        data['state'] = np.random.choice(['FIN', 'INT', 'CON', 'REQ', 'RST'], n_samples)
        
        # Create labels (20% attacks, 80% normal)
        data['label'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Make attacks have distinctive patterns
        attack_mask = data['label'] == 1
        data['rate'][attack_mask] *= 2  # Higher rate for attacks
        data['spkts'][attack_mask] = (data['spkts'][attack_mask] * 1.5).astype(int)  # More packets
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records with {df['label'].sum()} attacks")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        print("Handling missing values...")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'label' and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = ['proto', 'service', 'state']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        df[col] = df[col].map(lambda x: le.transform([x])[0] 
                                              if x in le.classes_ else -1)
        
        return df
    
    def normalize_features(self, X, fit=True):
        """Normalize numerical features to [0,1]"""
        print("Normalizing features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def feature_selection_correlation(self, X, y, threshold=0.3):
        """Select features based on Pearson correlation with target"""
        print(f"\n--- Correlation-Based Feature Selection ---")
        print(f"Threshold: |correlation| > {threshold}")
        
        df_temp = pd.DataFrame(X, columns=self.feature_names)
        df_temp['target'] = y
        
        # Calculate Pearson correlation with target
        correlations = df_temp.corr()['target'].abs().sort_values(ascending=False)
        
        # Display top correlations
        print("\nTop 10 Correlated Features:")
        for i, (feature, corr) in enumerate(correlations[1:11].items(), 1):
            print(f"  {i}. {feature}: {corr:.4f}")
        
        # Select features above threshold
        selected_features = correlations[correlations > threshold].index.tolist()
        if 'target' in selected_features:
            selected_features.remove('target')
        
        print(f"\nCorrelation-based selection: {len(self.feature_names)} → {len(selected_features)} features")
        
        return selected_features
    
    def feature_selection_information_gain(self, X, y, top_k=22):
        """Select features based on Information Gain (Mutual Information)"""
        from sklearn.feature_selection import mutual_info_classif
        
        print(f"\n--- Information Gain Feature Selection ---")
        print(f"Selecting top {top_k} features")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dataframe
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': mi_scores
        }).sort_values('score', ascending=False)
        
        # Display top features
        print("\nTop 10 Features by Information Gain:")
        for i, row in feature_scores.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['score']:.4f}")
        
        # Select top k features
        selected_features = feature_scores.head(top_k)['feature'].tolist()
        
        print(f"\nInformation Gain selection: {len(self.feature_names)} → {len(selected_features)} features")
        
        return selected_features, feature_scores
    
    def feature_selection_variance_threshold(self, X, threshold=0.01):
        """Remove low variance features"""
        from sklearn.feature_selection import VarianceThreshold
        
        print(f"\n--- Variance Threshold Feature Selection ---")
        print(f"Removing features with variance < {threshold}")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Variance threshold selection: {len(self.feature_names)} → {len(selected_features)} features")
        
        return selected_features
    
    def feature_selection_hybrid(self, X, y, correlation_threshold=0.3, top_k=22):
        """
        Hybrid Feature Selection: Correlation + Information Gain
        This is the NOVEL approach combining two methods
        """
        print("\n" + "="*60)
        print("HYBRID FEATURE SELECTION (Novel Approach)")
        print("Step 1: Correlation-based → Step 2: Information Gain")
        print("="*60)
        
        # Step 1: Correlation-based selection
        corr_features = self.feature_selection_correlation(X, y, correlation_threshold)
        
        # Get indices of correlation-selected features
        corr_indices = [self.feature_names.index(f) for f in corr_features]
        X_corr = X[:, corr_indices]
        
        # Step 2: Information Gain on correlation-selected features
        print(f"\nApplying Information Gain on {len(corr_features)} correlation-selected features...")
        
        # Update feature names temporarily for IG
        original_feature_names = self.feature_names
        self.feature_names = corr_features
        
        # Apply Information Gain
        ig_features, ig_scores = self.feature_selection_information_gain(X_corr, y, min(top_k, len(corr_features)))
        
        # Restore original feature names
        self.feature_names = original_feature_names
        
        print("\n" + "="*60)
        print(f"FINAL HYBRID SELECTION: {len(original_feature_names)} → {len(ig_features)} features")
        print("="*60)
        
        # Display final selected features
        print("\nFinal Selected Features:")
        for i, feature in enumerate(ig_features, 1):
            print(f"  {i}. {feature}")
        
        return ig_features, ig_scores
    
    def balance_classes(self, X, y):
        """Balance classes using SMOTE"""
        print("Balancing classes using SMOTE...")
        
        original_counts = np.bincount(y)
        print(f"Original distribution: Normal={original_counts[0]}, Attack={original_counts[1]}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        balanced_counts = np.bincount(y_balanced)
        print(f"Balanced distribution: Normal={balanced_counts[0]}, Attack={balanced_counts[1]}")
        
        return X_balanced, y_balanced
    
    def preprocess_pipeline(self, df, fit=True, apply_smote=True, feature_selection_method='hybrid'):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        fit : bool
            Whether to fit preprocessors (True for training, False for testing)
        apply_smote : bool
            Whether to apply SMOTE balancing
        feature_selection_method : str
            Method for feature selection: 'correlation', 'information_gain', 'hybrid', or 'none'
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # Separate features and labels
        if 'label' in df.columns:
            y = df['label'].values
            X = df.drop('label', axis=1)
        else:
            y = None
            X = df.copy()
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
            print(f"Original features: {len(self.feature_names)}")
        
        # Step 1: Handle missing values
        X = self.handle_missing_values(X)
        
        # Step 2: Encode categorical features
        X = self.encode_categorical(X, fit=fit)
        
        # Step 3: Normalize features
        X_scaled = self.normalize_features(X.values, fit=fit)
        
        # Step 4: Feature selection (only during training)
        if fit and y is not None and feature_selection_method != 'none':
            print(f"\nFeature Selection Method: {feature_selection_method.upper()}")
            
            if feature_selection_method == 'correlation':
                selected_features = self.feature_selection_correlation(X_scaled, y, threshold=0.3)
            elif feature_selection_method == 'information_gain':
                selected_features, _ = self.feature_selection_information_gain(X_scaled, y, top_k=22)
            elif feature_selection_method == 'hybrid':
                # NOVEL APPROACH: Correlation + Information Gain
                selected_features, _ = self.feature_selection_hybrid(X_scaled, y, 
                                                                     correlation_threshold=0.3, 
                                                                     top_k=22)
            else:
                selected_features = self.feature_names
            
            # Store selected feature indices
            self.selected_feature_indices = [self.feature_names.index(f) for f in selected_features]
            self.selected_features = selected_features
            
            # Apply feature selection
            X_scaled = X_scaled[:, self.selected_feature_indices]
            
            print(f"\n✓ Feature selection complete: {len(self.feature_names)} → {len(self.selected_features)} features")
            
        elif hasattr(self, 'selected_feature_indices'):
            # Use previously selected features for testing
            X_scaled = X_scaled[:, self.selected_feature_indices]
        
        # Step 5: Balance classes (only during training)
        if fit and apply_smote and y is not None:
            X_scaled, y = self.balance_classes(X_scaled, y)
        
        print("\n" + "="*60)
        print(f"Final shape: {X_scaled.shape}")
        if y is not None:
            print(f"Class distribution: Normal={np.sum(y==0)}, Attack={np.sum(y==1)}")
        print("="*60 + "\n")
        
        return X_scaled, y
    
    def prepare_train_test_split(self, df, test_size=0.2):
        """Split data into train and test sets"""
        print("Splitting data into train and test sets...")
        
        # Split before preprocessing
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
