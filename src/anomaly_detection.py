
"""
Anomaly Detection for Network Traffic
Using unsupervised learning approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Unsupervised anomaly detection for network intrusion
    """
    
    def __init__(self, method='isolation_forest', contamination=0.1):
        """
        Initialize anomaly detector
        
        Args:
            method: 'isolation_forest' or 'one_class_svm'
            contamination: Expected proportion of anomalies
        """
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X):
        """
        Fit the anomaly detection model
        
        Args:
            X: Training data (normal traffic only ideally)
        """
        print(f"\nTraining {self.method.upper().replace('_', ' ')} model...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        
        print(f"Training completed. Contamination rate: {self.contamination}")
    
    def predict(self, X):
        """
        Predict anomalies
        
        Args:
            X: Data to predict on
            
        Returns:
            Predictions: 1 for normal, -1 for anomaly
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def decision_function(self, X):
        """
        Get anomaly scores
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        return scores
    
    def evaluate(self, X_test, y_test_binary):
        """
        Evaluate anomaly detection performance
        
        Args:
            X_test: Test features
            y_test_binary: True labels (0=normal, 1=anomaly)
            
        Returns:
            Metrics dictionary
        """
        # Predict
        predictions = self.predict(X_test)
        
        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = np.where(predictions == -1, 1, 0)
        
        # Get anomaly scores
        scores = self.decision_function(X_test)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC-AUC (use negative scores for ROC curve calculation)
        try:
            auc = roc_auc_score(y_test_binary, -scores)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        return metrics, y_pred
    
    def print_evaluation_report(self, metrics):
        """
        Print evaluation report
        """
        print(f"\n{'='*60}")
        print(f"ANOMALY DETECTION RESULTS - {self.method.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:        {metrics['auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
    
    def plot_roc_curve(self, X_test, y_test_binary, save_path=None):
        """
        Plot ROC curve
        """
        scores = self.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test_binary, -scores)
        auc = roc_auc_score(y_test_binary, -scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.method.upper().replace("_", " ")}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return plt


def compare_anomaly_methods(X_train, X_test, y_test):
    """
    Compare different anomaly detection methods
    """
    methods = ['isolation_forest', 'one_class_svm']
    results = {}
    
    # Convert labels to binary (0=normal, 1=attack)
    y_test_binary = np.where(y_test == 'attack', 1, 0)
    
    print("\n" + "="*70)
    print("COMPARING ANOMALY DETECTION METHODS")
    print("="*70)
    
    for method in methods:
        print(f"\n{'#'*70}")
        print(f"Method: {method.upper().replace('_', ' ')}")
        print(f"{'#'*70}")
        
        # Create and train detector
        detector = AnomalyDetector(method=method, contamination=0.3)
        detector.fit(X_train)
        
        # Evaluate
        metrics, y_pred = detector.evaluate(X_test, y_test_binary)
        detector.print_evaluation_report(metrics)
        
        # Store results
        results[method] = {
            'metrics': metrics,
            'detector': detector
        }
    
    return results


def plot_anomaly_comparison(results, save_path=None):
    """
    Plot comparison of anomaly detection methods
    """
    methods = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Anomaly Detection Methods Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 3, idx % 3]
        values = [results[method]['metrics'][metric] for method in methods]
        colors = ['#9b59b6', '#f39c12']
        
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAnomaly detection comparison saved to {save_path}")
    
    return plt


def main():
    """
    Main execution for anomaly detection demo
    """
    print("\n" + "="*70)
    print("ANOMALY DETECTION FOR NETWORK INTRUSION")
    print("="*70)
    
    # Import data generation function
    from intrusion_detection import generate_synthetic_network_traffic
    
    # Generate data
    df = generate_synthetic_network_traffic(n_samples=8000, attack_ratio=0.3)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Compare methods
    results = compare_anomaly_methods(X_train, X_test, y_test)
    
    # Plot comparison
    plot_anomaly_comparison(results, save_path='../docs/anomaly_comparison.png')
    
    # Plot ROC curves
    y_test_binary = np.where(y_test == 'attack', 1, 0)
    results['isolation_forest']['detector'].plot_roc_curve(
        X_test, y_test_binary, save_path='../docs/roc_curve_isolation_forest.png'
    )
    
    print("\n" + "="*70)
    print("Anomaly detection analysis completed!")
    print("="*70)


if __name__ == "__main__":
    main()
