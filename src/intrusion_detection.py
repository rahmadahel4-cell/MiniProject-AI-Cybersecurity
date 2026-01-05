AI-Based Intrusion Detection System
Project 16: Implementation of an AI-Based Cybersecurity Use Case
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionSystem:
    """
    AI-based Intrusion Detection System with multiple ML algorithms
    """
    
    def __init__(self, algorithm='random_forest'):
        """
        Initialize IDS with specified algorithm
        
        Args:
            algorithm: 'random_forest', 'decision_tree', or 'neural_network'
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_time = 0
        self.prediction_time = 0
        
        # Initialize model based on algorithm choice
        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif algorithm == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif algorithm == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features and labels
        
        Args:
            X: Feature matrix
            y: Labels (optional, for training)
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed X and y
        """
        # Handle categorical features if present
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            # Convert categorical columns to numeric
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                if fit:
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    X[col] = le.transform(X[col].astype(str))
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def train(self, X_train, y_train):
        """
        Train the IDS model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm.upper()} model...")
        print(f"{'='*60}")
        
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train model and measure time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_encoded)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.4f} seconds")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        # Preprocess data
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        
        # Predict and measure time
        start_time = time.time()
        y_pred_encoded = self.model.predict(X_test_scaled)
        self.prediction_time = time.time() - start_time
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary and multi-class classification
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # For binary classification, use 'attack' as positive label
            pos_label = 'attack' if 'attack' in unique_labels else unique_labels[1]
            precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            # For multi-class, use weighted average
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
        
        return metrics, y_pred
    
    def print_evaluation_report(self, y_test, y_pred, metrics):
        """
        Print detailed evaluation report
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print(f"Training Time:  {metrics['training_time']:.4f} seconds")
        print(f"Prediction Time: {metrics['prediction_time']:.6f} seconds")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.algorithm.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt


def generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3):
    """
    Generate synthetic network traffic data for testing
    
    Args:
        n_samples: Number of samples to generate
        attack_ratio: Proportion of attack samples
        
    Returns:
        DataFrame with features and labels
    """
    print(f"\nGenerating {n_samples} synthetic network traffic samples...")
    print(f"Attack ratio: {attack_ratio*100:.1f}%")
    
    np.random.seed(42)
    
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    # Normal traffic features
    normal_data = {
        'duration': np.random.exponential(scale=10, size=n_normal),
        'src_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'wrong_fragment': np.random.poisson(lam=0.1, size=n_normal),
        'urgent': np.random.poisson(lam=0.05, size=n_normal),
        'hot': np.random.poisson(lam=0.5, size=n_normal),
        'num_failed_logins': np.zeros(n_normal),
        'logged_in': np.ones(n_normal),
        'num_compromised': np.zeros(n_normal),
        'root_shell': np.zeros(n_normal),
        'su_attempted': np.zeros(n_normal),
        'num_root': np.random.poisson(lam=0.2, size=n_normal),
        'num_file_creations': np.random.poisson(lam=1, size=n_normal),
        'num_shells': np.random.poisson(lam=0.1, size=n_normal),
        'num_access_files': np.random.poisson(lam=0.5, size=n_normal),
        'count': np.random.poisson(lam=5, size=n_normal),
        'srv_count': np.random.poisson(lam=5, size=n_normal),
        'label': ['normal'] * n_normal
    }
    
    # Attack traffic features (anomalous patterns)
    attack_data = {
        'duration': np.random.exponential(scale=50, size=n_attacks),  # Longer duration
        'src_bytes': np.random.lognormal(mean=10, sigma=3, size=n_attacks),  # More bytes
        'dst_bytes': np.random.lognormal(mean=5, sigma=3, size=n_attacks),
        'wrong_fragment': np.random.poisson(lam=2, size=n_attacks),  # More errors
        'urgent': np.random.poisson(lam=1, size=n_attacks),
        'hot': np.random.poisson(lam=3, size=n_attacks),  # More hot indicators
        'num_failed_logins': np.random.poisson(lam=2, size=n_attacks),  # Failed logins
        'logged_in': np.random.binomial(1, 0.3, size=n_attacks),  # Less likely logged in
        'num_compromised': np.random.poisson(lam=1, size=n_attacks),  # Compromise indicators
        'root_shell': np.random.binomial(1, 0.2, size=n_attacks),
        'su_attempted': np.random.binomial(1, 0.3, size=n_attacks),
        'num_root': np.random.poisson(lam=2, size=n_attacks),
        'num_file_creations': np.random.poisson(lam=5, size=n_attacks),  # More file activity
        'num_shells': np.random.poisson(lam=1, size=n_attacks),
        'num_access_files': np.random.poisson(lam=3, size=n_attacks),
        'count': np.random.poisson(lam=20, size=n_attacks),  # More connections
        'srv_count': np.random.poisson(lam=15, size=n_attacks),
        'label': ['attack'] * n_attacks
    }
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"  - Normal: {len(df[df['label']=='normal'])} ({len(df[df['label']=='normal'])/len(df)*100:.1f}%)")
    print(f"  - Attack: {len(df[df['label']=='attack'])} ({len(df[df['label']=='attack'])/len(df)*100:.1f}%)")
    
    return df


def compare_algorithms(X_train, y_train, X_test, y_test):
    """
    Compare different ML algorithms for intrusion detection
    """
    algorithms = ['random_forest', 'decision_tree', 'neural_network']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DIFFERENT ML ALGORITHMS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f"Algorithm: {algo.upper().replace('_', ' ')}")
        print(f"{'#'*70}")
        
        # Create and train model
        ids = IntrusionDetectionSystem(algorithm=algo)
        ids.train(X_train, y_train)
        
        # Evaluate
        metrics, y_pred = ids.evaluate(X_test, y_test)
        ids.print_evaluation_report(y_test, y_pred, metrics)
        
        # Save results
        results[algo] = {
            'metrics': metrics,
            'model': ids
        }
    
    return results


def plot_algorithm_comparison(results, save_path=None):
    """
    Plot comparison of different algorithms
    """
    algorithms = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Comparison - Performance Metrics', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        values = [results[algo]['metrics'][metric] for algo in algorithms]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    return plt


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("AI-BASED INTRUSION DETECTION SYSTEM")
    print("Project 16: Implementation of an AI-Based Cybersecurity Use Case")
    print("="*70)
    
    # Generate synthetic data
    df = generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3)
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Compare algorithms
    results = compare_algorithms(X_train, y_train, X_test, y_test)
    
    # Plot comparison
    plot_algorithm_comparison(results, save_path='../docs/algorithm_comparison.png')
    
    # Plot confusion matrix for best model (Random Forest)
    best_model = results['random_forest']['model']
    _, y_pred = results['random_forest']['model'].evaluate(X_test, y_test)
    best_model.plot_confusion_matrix(y_test, y_pred, save_path='../docs/confusion_matrix_rf.png')
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nPerformance Ranking:")
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'], 
                          reverse=True)
    
    for rank, (algo, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"{rank}. {algo.upper().replace('_', ' ')}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Training Time: {metrics['training_time']:.4f}s")
    
    print("\n" + "="*70)
    print("Project completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
AI-Based Intrusion Detection System
Project 16: Implementation of an AI-Based Cybersecurity Use Case
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionSystem:
    """
    AI-based Intrusion Detection System with multiple ML algorithms
    """
    
    def __init__(self, algorithm='random_forest'):
        """
        Initialize IDS with specified algorithm
        
        Args:
            algorithm: 'random_forest', 'decision_tree', or 'neural_network'
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_time = 0
        self.prediction_time = 0
        
        # Initialize model based on algorithm choice
        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif algorithm == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif algorithm == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features and labels
        
        Args:
            X: Feature matrix
            y: Labels (optional, for training)
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed X and y
        """
        # Handle categorical features if present
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            # Convert categorical columns to numeric
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                if fit:
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    X[col] = le.transform(X[col].astype(str))
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def train(self, X_train, y_train):
        """
        Train the IDS model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm.upper()} model...")
        print(f"{'='*60}")
        
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train model and measure time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_encoded)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.4f} seconds")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        # Preprocess data
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        
        # Predict and measure time
        start_time = time.time()
        y_pred_encoded = self.model.predict(X_test_scaled)
        self.prediction_time = time.time() - start_time
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary and multi-class classification
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # For binary classification, use 'attack' as positive label
            pos_label = 'attack' if 'attack' in unique_labels else unique_labels[1]
            precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            # For multi-class, use weighted average
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
        
        return metrics, y_pred
    
    def print_evaluation_report(self, y_test, y_pred, metrics):
        """
        Print detailed evaluation report
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print(f"Training Time:  {metrics['training_time']:.4f} seconds")
        print(f"Prediction Time: {metrics['prediction_time']:.6f} seconds")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.algorithm.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt


def generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3):
    """
    Generate synthetic network traffic data for testing
    
    Args:
        n_samples: Number of samples to generate
        attack_ratio: Proportion of attack samples
        
    Returns:
        DataFrame with features and labels
    """
    print(f"\nGenerating {n_samples} synthetic network traffic samples...")
    print(f"Attack ratio: {attack_ratio*100:.1f}%")
    
    np.random.seed(42)
    
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    # Normal traffic features
    normal_data = {
        'duration': np.random.exponential(scale=10, size=n_normal),
        'src_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'wrong_fragment': np.random.poisson(lam=0.1, size=n_normal),
        'urgent': np.random.poisson(lam=0.05, size=n_normal),
        'hot': np.random.poisson(lam=0.5, size=n_normal),
        'num_failed_logins': np.zeros(n_normal),
        'logged_in': np.ones(n_normal),
        'num_compromised': np.zeros(n_normal),
        'root_shell': np.zeros(n_normal),
        'su_attempted': np.zeros(n_normal),
        'num_root': np.random.poisson(lam=0.2, size=n_normal),
        'num_file_creations': np.random.poisson(lam=1, size=n_normal),
        'num_shells': np.random.poisson(lam=0.1, size=n_normal),
        'num_access_files': np.random.poisson(lam=0.5, size=n_normal),
        'count': np.random.poisson(lam=5, size=n_normal),
        'srv_count': np.random.poisson(lam=5, size=n_normal),
        'label': ['normal'] * n_normal
    }
    
    # Attack traffic features (anomalous patterns)
    attack_data = {
        'duration': np.random.exponential(scale=50, size=n_attacks),  # Longer duration
        'src_bytes': np.random.lognormal(mean=10, sigma=3, size=n_attacks),  # More bytes
        'dst_bytes': np.random.lognormal(mean=5, sigma=3, size=n_attacks),
        'wrong_fragment': np.random.poisson(lam=2, size=n_attacks),  # More errors
        'urgent': np.random.poisson(lam=1, size=n_attacks),
        'hot': np.random.poisson(lam=3, size=n_attacks),  # More hot indicators
        'num_failed_logins': np.random.poisson(lam=2, size=n_attacks),  # Failed logins
        'logged_in': np.random.binomial(1, 0.3, size=n_attacks),  # Less likely logged in
        'num_compromised': np.random.poisson(lam=1, size=n_attacks),  # Compromise indicators
        'root_shell': np.random.binomial(1, 0.2, size=n_attacks),
        'su_attempted': np.random.binomial(1, 0.3, size=n_attacks),
        'num_root': np.random.poisson(lam=2, size=n_attacks),
        'num_file_creations': np.random.poisson(lam=5, size=n_attacks),  # More file activity
        'num_shells': np.random.poisson(lam=1, size=n_attacks),
        'num_access_files': np.random.poisson(lam=3, size=n_attacks),
        'count': np.random.poisson(lam=20, size=n_attacks),  # More connections
        'srv_count': np.random.poisson(lam=15, size=n_attacks),
        'label': ['attack'] * n_attacks
    }
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"  - Normal: {len(df[df['label']=='normal'])} ({len(df[df['label']=='normal'])/len(df)*100:.1f}%)")
    print(f"  - Attack: {len(df[df['label']=='attack'])} ({len(df[df['label']=='attack'])/len(df)*100:.1f}%)")
    
    return df


def compare_algorithms(X_train, y_train, X_test, y_test):
    """
    Compare different ML algorithms for intrusion detection
    """
    algorithms = ['random_forest', 'decision_tree', 'neural_network']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DIFFERENT ML ALGORITHMS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f"Algorithm: {algo.upper().replace('_', ' ')}")
        print(f"{'#'*70}")
        
        # Create and train model
        ids = IntrusionDetectionSystem(algorithm=algo)
        ids.train(X_train, y_train)
        
        # Evaluate
        metrics, y_pred = ids.evaluate(X_test, y_test)
        ids.print_evaluation_report(y_test, y_pred, metrics)
        
        # Save results
        results[algo] = {
            'metrics': metrics,
            'model': ids
        }
    
    return results


def plot_algorithm_comparison(results, save_path=None):
    """
    Plot comparison of different algorithms
    """
    algorithms = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Comparison - Performance Metrics', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        values = [results[algo]['metrics'][metric] for algo in algorithms]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    return plt


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("AI-BASED INTRUSION DETECTION SYSTEM")
    print("Project 16: Implementation of an AI-Based Cybersecurity Use Case")
    print("="*70)
    
    # Generate synthetic data
    df = generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3)
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Compare algorithms
    results = compare_algorithms(X_train, y_train, X_test, y_test)
    
    # Plot comparison
    plot_algorithm_comparison(results, save_path='../docs/algorithm_comparison.png')
    
    # Plot confusion matrix for best model (Random Forest)
    best_model = results['random_forest']['model']
    _, y_pred = results['random_forest']['model'].evaluate(X_test, y_test)
    best_model.plot_confusion_matrix(y_test, y_pred, save_path='../docs/confusion_matrix_rf.png')
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nPerformance Ranking:")
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'], 
                          reverse=True)
    
    for rank, (algo, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"{rank}. {algo.upper().replace('_', ' ')}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Training Time: {metrics['training_time']:.4f}s")
    
    print("\n" + "="*70)
    print("Project completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
AI-Based Intrusion Detection System
Project 16: Implementation of an AI-Based Cybersecurity Use Case
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionSystem:
    """
    AI-based Intrusion Detection System with multiple ML algorithms
    """
    
    def __init__(self, algorithm='random_forest'):
        """
        Initialize IDS with specified algorithm
        
        Args:
            algorithm: 'random_forest', 'decision_tree', or 'neural_network'
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_time = 0
        self.prediction_time = 0
        
        # Initialize model based on algorithm choice
        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif algorithm == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif algorithm == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features and labels
        
        Args:
            X: Feature matrix
            y: Labels (optional, for training)
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed X and y
        """
        # Handle categorical features if present
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            # Convert categorical columns to numeric
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                if fit:
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    X[col] = le.transform(X[col].astype(str))
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def train(self, X_train, y_train):
        """
        Train the IDS model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm.upper()} model...")
        print(f"{'='*60}")
        
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train model and measure time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_encoded)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.4f} seconds")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        # Preprocess data
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        
        # Predict and measure time
        start_time = time.time()
        y_pred_encoded = self.model.predict(X_test_scaled)
        self.prediction_time = time.time() - start_time
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary and multi-class classification
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # For binary classification, use 'attack' as positive label
            pos_label = 'attack' if 'attack' in unique_labels else unique_labels[1]
            precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            # For multi-class, use weighted average
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
        
        return metrics, y_pred
    
    def print_evaluation_report(self, y_test, y_pred, metrics):
        """
        Print detailed evaluation report
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print(f"Training Time:  {metrics['training_time']:.4f} seconds")
        print(f"Prediction Time: {metrics['prediction_time']:.6f} seconds")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.algorithm.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt


def generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3):
    """
    Generate synthetic network traffic data for testing
    
    Args:
        n_samples: Number of samples to generate
        attack_ratio: Proportion of attack samples
        
    Returns:
        DataFrame with features and labels
    """
    print(f"\nGenerating {n_samples} synthetic network traffic samples...")
    print(f"Attack ratio: {attack_ratio*100:.1f}%")
    
    np.random.seed(42)
    
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    # Normal traffic features
    normal_data = {
        'duration': np.random.exponential(scale=10, size=n_normal),
        'src_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'wrong_fragment': np.random.poisson(lam=0.1, size=n_normal),
        'urgent': np.random.poisson(lam=0.05, size=n_normal),
        'hot': np.random.poisson(lam=0.5, size=n_normal),
        'num_failed_logins': np.zeros(n_normal),
        'logged_in': np.ones(n_normal),
        'num_compromised': np.zeros(n_normal),
        'root_shell': np.zeros(n_normal),
        'su_attempted': np.zeros(n_normal),
        'num_root': np.random.poisson(lam=0.2, size=n_normal),
        'num_file_creations': np.random.poisson(lam=1, size=n_normal),
        'num_shells': np.random.poisson(lam=0.1, size=n_normal),
        'num_access_files': np.random.poisson(lam=0.5, size=n_normal),
        'count': np.random.poisson(lam=5, size=n_normal),
        'srv_count': np.random.poisson(lam=5, size=n_normal),
        'label': ['normal'] * n_normal
    }
    
    # Attack traffic features (anomalous patterns)
    attack_data = {
        'duration': np.random.exponential(scale=50, size=n_attacks),  # Longer duration
        'src_bytes': np.random.lognormal(mean=10, sigma=3, size=n_attacks),  # More bytes
        'dst_bytes': np.random.lognormal(mean=5, sigma=3, size=n_attacks),
        'wrong_fragment': np.random.poisson(lam=2, size=n_attacks),  # More errors
        'urgent': np.random.poisson(lam=1, size=n_attacks),
        'hot': np.random.poisson(lam=3, size=n_attacks),  # More hot indicators
        'num_failed_logins': np.random.poisson(lam=2, size=n_attacks),  # Failed logins
        'logged_in': np.random.binomial(1, 0.3, size=n_attacks),  # Less likely logged in
        'num_compromised': np.random.poisson(lam=1, size=n_attacks),  # Compromise indicators
        'root_shell': np.random.binomial(1, 0.2, size=n_attacks),
        'su_attempted': np.random.binomial(1, 0.3, size=n_attacks),
        'num_root': np.random.poisson(lam=2, size=n_attacks),
        'num_file_creations': np.random.poisson(lam=5, size=n_attacks),  # More file activity
        'num_shells': np.random.poisson(lam=1, size=n_attacks),
        'num_access_files': np.random.poisson(lam=3, size=n_attacks),
        'count': np.random.poisson(lam=20, size=n_attacks),  # More connections
        'srv_count': np.random.poisson(lam=15, size=n_attacks),
        'label': ['attack'] * n_attacks
    }
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"  - Normal: {len(df[df['label']=='normal'])} ({len(df[df['label']=='normal'])/len(df)*100:.1f}%)")
    print(f"  - Attack: {len(df[df['label']=='attack'])} ({len(df[df['label']=='attack'])/len(df)*100:.1f}%)")
    
    return df


def compare_algorithms(X_train, y_train, X_test, y_test):
    """
    Compare different ML algorithms for intrusion detection
    """
    algorithms = ['random_forest', 'decision_tree', 'neural_network']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DIFFERENT ML ALGORITHMS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f"Algorithm: {algo.upper().replace('_', ' ')}")
        print(f"{'#'*70}")
        
        # Create and train model
        ids = IntrusionDetectionSystem(algorithm=algo)
        ids.train(X_train, y_train)
        
        # Evaluate
        metrics, y_pred = ids.evaluate(X_test, y_test)
        ids.print_evaluation_report(y_test, y_pred, metrics)
        
        # Save results
        results[algo] = {
            'metrics': metrics,
            'model': ids
        }
    
    return results


def plot_algorithm_comparison(results, save_path=None):
    """
    Plot comparison of different algorithms
    """
    algorithms = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Comparison - Performance Metrics', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        values = [results[algo]['metrics'][metric] for algo in algorithms]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    return plt


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("AI-BASED INTRUSION DETECTION SYSTEM")
    print("Project 16: Implementation of an AI-Based Cybersecurity Use Case")
    print("="*70)
    
    # Generate synthetic data
    df = generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3)
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Compare algorithms
    results = compare_algorithms(X_train, y_train, X_test, y_test)
    
    # Plot comparison
    plot_algorithm_comparison(results, save_path='../docs/algorithm_comparison.png')
    
    # Plot confusion matrix for best model (Random Forest)
    best_model = results['random_forest']['model']
    _, y_pred = results['random_forest']['model'].evaluate(X_test, y_test)
    best_model.plot_confusion_matrix(y_test, y_pred, save_path='../docs/confusion_matrix_rf.png')
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nPerformance Ranking:")
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'], 
                          reverse=True)
    
    for rank, (algo, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"{rank}. {algo.upper().replace('_', ' ')}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Training Time: {metrics['training_time']:.4f}s")
    
    print("\n" + "="*70)
    print("Project completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
AI-Based Intrusion Detection System
Project 16: Implementation of an AI-Based Cybersecurity Use Case
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionSystem:
    """
    AI-based Intrusion Detection System with multiple ML algorithms
    """
    
    def __init__(self, algorithm='random_forest'):
        """
        Initialize IDS with specified algorithm
        
        Args:
            algorithm: 'random_forest', 'decision_tree', or 'neural_network'
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_time = 0
        self.prediction_time = 0
        
        # Initialize model based on algorithm choice
        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif algorithm == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif algorithm == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features and labels
        
        Args:
            X: Feature matrix
            y: Labels (optional, for training)
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed X and y
        """
        # Handle categorical features if present
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            # Convert categorical columns to numeric
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                if fit:
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    X[col] = le.transform(X[col].astype(str))
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def train(self, X_train, y_train):
        """
        Train the IDS model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm.upper()} model...")
        print(f"{'='*60}")
        
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train model and measure time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_encoded)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.4f} seconds")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        # Preprocess data
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        
        # Predict and measure time
        start_time = time.time()
        y_pred_encoded = self.model.predict(X_test_scaled)
        self.prediction_time = time.time() - start_time
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary and multi-class classification
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # For binary classification, use 'attack' as positive label
            pos_label = 'attack' if 'attack' in unique_labels else unique_labels[1]
            precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            # For multi-class, use weighted average
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
        
        return metrics, y_pred
    
    def print_evaluation_report(self, y_test, y_pred, metrics):
        """
        Print detailed evaluation report
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print(f"Training Time:  {metrics['training_time']:.4f} seconds")
        print(f"Prediction Time: {metrics['prediction_time']:.6f} seconds")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.algorithm.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt


def generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3):
    """
    Generate synthetic network traffic data for testing
    
    Args:
        n_samples: Number of samples to generate
        attack_ratio: Proportion of attack samples
        
    Returns:
        DataFrame with features and labels
    """
    print(f"\nGenerating {n_samples} synthetic network traffic samples...")
    print(f"Attack ratio: {attack_ratio*100:.1f}%")
    
    np.random.seed(42)
    
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    # Normal traffic features
    normal_data = {
        'duration': np.random.exponential(scale=10, size=n_normal),
        'src_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n_normal),
        'wrong_fragment': np.random.poisson(lam=0.1, size=n_normal),
        'urgent': np.random.poisson(lam=0.05, size=n_normal),
        'hot': np.random.poisson(lam=0.5, size=n_normal),
        'num_failed_logins': np.zeros(n_normal),
        'logged_in': np.ones(n_normal),
        'num_compromised': np.zeros(n_normal),
        'root_shell': np.zeros(n_normal),
        'su_attempted': np.zeros(n_normal),
        'num_root': np.random.poisson(lam=0.2, size=n_normal),
        'num_file_creations': np.random.poisson(lam=1, size=n_normal),
        'num_shells': np.random.poisson(lam=0.1, size=n_normal),
        'num_access_files': np.random.poisson(lam=0.5, size=n_normal),
        'count': np.random.poisson(lam=5, size=n_normal),
        'srv_count': np.random.poisson(lam=5, size=n_normal),
        'label': ['normal'] * n_normal
    }
    
    # Attack traffic features (anomalous patterns)
    attack_data = {
        'duration': np.random.exponential(scale=50, size=n_attacks),  # Longer duration
        'src_bytes': np.random.lognormal(mean=10, sigma=3, size=n_attacks),  # More bytes
        'dst_bytes': np.random.lognormal(mean=5, sigma=3, size=n_attacks),
        'wrong_fragment': np.random.poisson(lam=2, size=n_attacks),  # More errors
        'urgent': np.random.poisson(lam=1, size=n_attacks),
        'hot': np.random.poisson(lam=3, size=n_attacks),  # More hot indicators
        'num_failed_logins': np.random.poisson(lam=2, size=n_attacks),  # Failed logins
        'logged_in': np.random.binomial(1, 0.3, size=n_attacks),  # Less likely logged in
        'num_compromised': np.random.poisson(lam=1, size=n_attacks),  # Compromise indicators
        'root_shell': np.random.binomial(1, 0.2, size=n_attacks),
        'su_attempted': np.random.binomial(1, 0.3, size=n_attacks),
        'num_root': np.random.poisson(lam=2, size=n_attacks),
        'num_file_creations': np.random.poisson(lam=5, size=n_attacks),  # More file activity
        'num_shells': np.random.poisson(lam=1, size=n_attacks),
        'num_access_files': np.random.poisson(lam=3, size=n_attacks),
        'count': np.random.poisson(lam=20, size=n_attacks),  # More connections
        'srv_count': np.random.poisson(lam=15, size=n_attacks),
        'label': ['attack'] * n_attacks
    }
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"  - Normal: {len(df[df['label']=='normal'])} ({len(df[df['label']=='normal'])/len(df)*100:.1f}%)")
    print(f"  - Attack: {len(df[df['label']=='attack'])} ({len(df[df['label']=='attack'])/len(df)*100:.1f}%)")
    
    return df


def compare_algorithms(X_train, y_train, X_test, y_test):
    """
    Compare different ML algorithms for intrusion detection
    """
    algorithms = ['random_forest', 'decision_tree', 'neural_network']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DIFFERENT ML ALGORITHMS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f"Algorithm: {algo.upper().replace('_', ' ')}")
        print(f"{'#'*70}")
        
        # Create and train model
        ids = IntrusionDetectionSystem(algorithm=algo)
        ids.train(X_train, y_train)
        
        # Evaluate
        metrics, y_pred = ids.evaluate(X_test, y_test)
        ids.print_evaluation_report(y_test, y_pred, metrics)
        
        # Save results
        results[algo] = {
            'metrics': metrics,
            'model': ids
        }
    
    return results


def plot_algorithm_comparison(results, save_path=None):
    """
    Plot comparison of different algorithms
    """
    algorithms = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Comparison - Performance Metrics', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        values = [results[algo]['metrics'][metric] for algo in algorithms]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    return plt


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("AI-BASED INTRUSION DETECTION SYSTEM")
    print("Project 16: Implementation of an AI-Based Cybersecurity Use Case")
    print("="*70)
    
    # Generate synthetic data
    df = generate_synthetic_network_traffic(n_samples=10000, attack_ratio=0.3)
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Compare algorithms
    results = compare_algorithms(X_train, y_train, X_test, y_test)
    
    # Plot comparison
    plot_algorithm_comparison(results, save_path='../docs/algorithm_comparison.png')
    
    # Plot confusion matrix for best model (Random Forest)
    best_model = results['random_forest']['model']
    _, y_pred = results['random_forest']['model'].evaluate(X_test, y_test)
    best_model.plot_confusion_matrix(y_test, y_pred, save_path='../docs/confusion_matrix_rf.png')
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nPerformance Ranking:")
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'], 
                          reverse=True)
    
    for rank, (algo, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"{rank}. {algo.upper().replace('_', ' ')}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Training Time: {metrics['training_time']:.4f}s")
    
    print("\n" + "="*70)
    print("Project completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
