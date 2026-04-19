"""
Feature Selection Demonstration Script
Shows all three methods: Correlation, Information Gain, and Hybrid
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import DataPreprocessor
from config import Config

def visualize_feature_selection():
    """Demonstrate and visualize feature selection methods"""
    
    print("\n" + "="*80)
    print(" "*20 + "FEATURE SELECTION DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Generate sample data
    print("Step 1: Generating sample data...")
    df = preprocessor.load_sample_data(n_samples=5000)
    
    # Separate features and labels
    y = df['label'].values
    X_df = df.drop('label', axis=1)
    
    # Store feature names
    preprocessor.feature_names = X_df.columns.tolist()
    original_feature_count = len(preprocessor.feature_names)
    print(f"✓ Original features: {original_feature_count}\n")
    
    # Basic preprocessing (no feature selection yet)
    print("Step 2: Basic preprocessing...")
    X_df = preprocessor.handle_missing_values(X_df)
    X_df = preprocessor.encode_categorical(X_df, fit=True)
    X_scaled = preprocessor.normalize_features(X_df.values, fit=True)
    print("✓ Preprocessing complete\n")
    
    # Create comparison results
    results = {}
    
    # Method 1: Correlation-based selection
    print("\n" + "="*80)
    print("METHOD 1: CORRELATION-BASED FEATURE SELECTION")
    print("="*80)
    corr_features = preprocessor.feature_selection_correlation(X_scaled, y, threshold=0.3)
    results['Correlation'] = {
        'features': corr_features,
        'count': len(corr_features)
    }
    
    # Method 2: Information Gain selection
    print("\n" + "="*80)
    print("METHOD 2: INFORMATION GAIN FEATURE SELECTION")
    print("="*80)
    ig_features, ig_scores = preprocessor.feature_selection_information_gain(X_scaled, y, top_k=22)
    results['Information Gain'] = {
        'features': ig_features,
        'count': len(ig_features),
        'scores': ig_scores
    }
    
    # Method 3: Hybrid selection (Novel Approach)
    print("\n" + "="*80)
    print("METHOD 3: HYBRID FEATURE SELECTION (NOVEL APPROACH)")
    print("="*80)
    hybrid_features, hybrid_scores = preprocessor.feature_selection_hybrid(
        X_scaled, y, 
        correlation_threshold=0.3, 
        top_k=22
    )
    results['Hybrid'] = {
        'features': hybrid_features,
        'count': len(hybrid_features),
        'scores': hybrid_scores
    }
    
    # Print comparison summary
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPARISON SUMMARY")
    print("="*80)
    print(f"\nOriginal features: {original_feature_count}")
    print("-" * 80)
    
    for method, data in results.items():
        reduction = ((original_feature_count - data['count']) / original_feature_count) * 100
        print(f"{method:20s}: {original_feature_count} → {data['count']} features ({reduction:.1f}% reduction)")
    
    # Find common features across all methods
    corr_set = set(results['Correlation']['features'])
    ig_set = set(results['Information Gain']['features'])
    hybrid_set = set(results['Hybrid']['features'])
    
    common_all = corr_set & ig_set & hybrid_set
    common_corr_ig = corr_set & ig_set
    common_corr_hybrid = corr_set & hybrid_set
    common_ig_hybrid = ig_set & hybrid_set
    
    print("\n" + "="*80)
    print("FEATURE OVERLAP ANALYSIS")
    print("="*80)
    print(f"Common to all 3 methods: {len(common_all)} features")
    print(f"Common to Correlation & IG: {len(common_corr_ig)} features")
    print(f"Common to Correlation & Hybrid: {len(common_corr_hybrid)} features")
    print(f"Common to IG & Hybrid: {len(common_ig_hybrid)} features")
    
    if common_all:
        print(f"\nFeatures selected by ALL methods:")
        for i, feature in enumerate(sorted(common_all), 1):
            print(f"  {i}. {feature}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Figure 1: Feature count comparison
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    counts = [results[m]['count'] for m in methods]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = plt.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=original_feature_count, color='gray', linestyle='--', 
                label=f'Original ({original_feature_count})')
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Feature Selection Methods Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_selection_comparison.png', dpi=300)
    print("✓ Saved: visualizations/feature_selection_comparison.png")
    
    # Figure 2: Information Gain scores
    if 'scores' in results['Information Gain']:
        plt.figure(figsize=(12, 6))
        scores_df = results['Information Gain']['scores'].head(22)
        
        plt.barh(range(len(scores_df)), scores_df['score'].values, color='#2ecc71', alpha=0.7)
        plt.yticks(range(len(scores_df)), scores_df['feature'].values)
        plt.xlabel('Information Gain Score', fontsize=12)
        plt.title('Top 22 Features by Information Gain', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/information_gain_scores.png', dpi=300)
        print("✓ Saved: visualizations/information_gain_scores.png")
    
    # Figure 3: Venn diagram (simplified bar chart showing overlaps)
    plt.figure(figsize=(10, 6))
    overlap_data = {
        'All 3 methods': len(common_all),
        'Corr & IG': len(common_corr_ig - common_all),
        'Corr & Hybrid': len(common_corr_hybrid - common_all),
        'IG & Hybrid': len(common_ig_hybrid - common_all),
        'Corr only': len(corr_set - ig_set - hybrid_set),
        'IG only': len(ig_set - corr_set - hybrid_set),
        'Hybrid only': len(hybrid_set - corr_set - ig_set)
    }
    
    labels = list(overlap_data.keys())
    values = list(overlap_data.values())
    colors_overlap = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    plt.bar(labels, values, color=colors_overlap, alpha=0.7, edgecolor='black')
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Feature Selection Overlap Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        if v > 0:
            plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_overlap_analysis.png', dpi=300)
    print("✓ Saved: visualizations/feature_overlap_analysis.png")
    
    # Save detailed results to CSV
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)
    
    # Create comparison CSV
    max_features = max(len(results[m]['features']) for m in results.keys())
    comparison_df = pd.DataFrame()
    
    for method in results.keys():
        features = results[method]['features'] + [''] * (max_features - len(results[method]['features']))
        comparison_df[method] = features[:max_features]
    
    comparison_df.to_csv('visualizations/feature_selection_comparison.csv', index=False)
    print("✓ Saved: visualizations/feature_selection_comparison.csv")
    
    # Save Information Gain scores
    if 'scores' in results['Information Gain']:
        results['Information Gain']['scores'].to_csv(
            'visualizations/information_gain_scores.csv', 
            index=False
        )
        print("✓ Saved: visualizations/information_gain_scores.csv")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\n🎯 HYBRID METHOD (Correlation + Information Gain) is RECOMMENDED because:")
    print("   ✓ Combines strengths of both methods")
    print("   ✓ Two-stage filtering ensures robust feature selection")
    print("   ✓ Reduces features effectively while maintaining information")
    print("   ✓ Novel approach for your project (not commonly used)")
    print(f"   ✓ Selected {results['Hybrid']['count']} most important features\n")
    
    print("="*80)
    print("✓ FEATURE SELECTION DEMONSTRATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        visualize_feature_selection()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
