
"""
Network Traffic Classification
Multi-class classification of different attack types
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class TrafficClassifier:
    """
    Multi-class network traffic classifier
    """
    
    def __init__(self):
        """
        Initialize traffic classifier
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features and labels
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
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
        Train the classifier
        """
        print(f"\nTraining Multi-Class Traffic Classifier...")
        
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        self.model.fit(X_train_scaled, y_train_encoded)
        
        print(f"Training completed.")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {', '.join(self.label_encoder.classes_)}")
    
    def predict(self, X_test):
        """
        Predict traffic classes
        """
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        y_pred_encoded = self.model.predict(X_test_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate classifier performance
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"TRAFFIC CLASSIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot multi-class confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    cbar_kws={'label': 'Number of Samples'})
        plt.title('Traffic Classification - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to {save_path}")
        
        return plt
    
    def plot_class_distribution(self, y_train, y_test, save_path=None):
        """
        Plot class distribution in training and test sets
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training set distribution
        train_counts = pd.Series(y_train).value_counts().sort_index()
        axes[0].bar(range(len(train_counts)), train_counts.values, 
                   color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Traffic Class', fontsize=11)
        axes[0].set_ylabel('Number of Samples', fontsize=11)
        axes[0].set_title('Training Set - Class Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(train_counts)))
        axes[0].set_xticklabels(train_counts.index, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(train_counts.values):
            axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Test set distribution
        test_counts = pd.Series(y_test).value_counts().sort_index()
        axes[1].bar(range(len(test_counts)), test_counts.values,
                   color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Traffic Class', fontsize=11)
        axes[1].set_ylabel('Number of Samples', fontsize=11)
        axes[1].set_title('Test Set - Class Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(test_counts)))
        axes[1].set_xticklabels(test_counts.index, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(test_counts.values):
            axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        return plt


def generate_multiclass_traffic(n_samples=10000):
    """
    Generate synthetic multi-class network traffic data
    """
    print(f"\nGenerating {n_samples} multi-class traffic samples...")
    
    np.random.seed(42)
    
    # Define attack types with different proportions
    attack_types = {
        'normal': 0.5,      # 50%
        'dos': 0.15,        # 15% Denial of Service
        'probe': 0.15,      # 15% Probing/Scanning
        'r2l': 0.10,        # 10% Remote to Local
        'u2r': 0.10         # 10% User to Root
    }
    
    data_list = []
    
    for attack_type, proportion in attack_types.items():
        n = int(n_samples * proportion)
        
        if attack_type == 'normal':
            # Normal traffic
            data = {
                'duration': np.random.exponential(scale=10, size=n),
                'src_bytes': np.random.lognormal(mean=7, sigma=2, size=n),
                'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n),
                'wrong_fragment': np.random.poisson(lam=0.1, size=n),
                'urgent': np.random.poisson(lam=0.05, size=n),
                'hot': np.random.poisson(lam=0.5, size=n),
                'num_failed_logins': np.zeros(n),
                'logged_in': np.ones(n),
                'num_compromised': np.zeros(n),
                'count': np.random.poisson(lam=5, size=n),
                'srv_count': np.random.poisson(lam=5, size=n),
                'serror_rate': np.random.uniform(0, 0.1, size=n),
                'srv_serror_rate': np.random.uniform(0, 0.1, size=n),
                'rerror_rate': np.random.uniform(0, 0.1, size=n),
                'same_srv_rate': np.random.uniform(0.8, 1.0, size=n),
                'label': [attack_type] * n
            }
        
        elif attack_type == 'dos':
            # Denial of Service - high connection rates
            data = {
                'duration': np.random.exponential(scale=2, size=n),  # Short duration
                'src_bytes': np.random.lognormal(mean=5, sigma=1, size=n),  # Small packets
                'dst_bytes': np.random.lognormal(mean=4, sigma=1, size=n),
                'wrong_fragment': np.random.poisson(lam=3, size=n),  # More errors
                'urgent': np.random.poisson(lam=0.5, size=n),
                'hot': np.random.poisson(lam=1, size=n),
                'num_failed_logins': np.zeros(n),
                'logged_in': np.zeros(n),
                'num_compromised': np.zeros(n),
                'count': np.random.poisson(lam=50, size=n),  # Very high count
                'srv_count': np.random.poisson(lam=45, size=n),
                'serror_rate': np.random.uniform(0.5, 1.0, size=n),  # High error rate
                'srv_serror_rate': np.random.uniform(0.5, 1.0, size=n),
                'rerror_rate': np.random.uniform(0.3, 0.8, size=n),
                'same_srv_rate': np.random.uniform(0.9, 1.0, size=n),  # Same service
                'label': [attack_type] * n
            }
        
        elif attack_type == 'probe':
            # Probing/Scanning attacks
            data = {
                'duration': np.random.exponential(scale=5, size=n),
                'src_bytes': np.random.lognormal(mean=5, sigma=1.5, size=n),
                'dst_bytes': np.random.lognormal(mean=4, sigma=1.5, size=n),
                'wrong_fragment': np.random.poisson(lam=0.5, size=n),
                'urgent': np.random.poisson(lam=0.1, size=n),
                'hot': np.random.poisson(lam=0.3, size=n),
                'num_failed_logins': np.random.poisson(lam=1, size=n),
                'logged_in': np.random.binomial(1, 0.2, size=n),
                'num_compromised': np.zeros(n),
                'count': np.random.poisson(lam=25, size=n),  # Moderate count
                'srv_count': np.random.poisson(lam=3, size=n),  # Low srv_count
                'serror_rate': np.random.uniform(0.2, 0.6, size=n),
                'srv_serror_rate': np.random.uniform(0.1, 0.4, size=n),
                'rerror_rate': np.random.uniform(0.4, 0.8, size=n),  # High rejection
                'same_srv_rate': np.random.uniform(0.1, 0.3, size=n),  # Different services
                'label': [attack_type] * n
            }
        
        elif attack_type == 'r2l':
            # Remote to Local attacks
            data = {
                'duration': np.random.exponential(scale=30, size=n),  # Longer duration
                'src_bytes': np.random.lognormal(mean=8, sigma=2, size=n),
                'dst_bytes': np.random.lognormal(mean=6, sigma=2, size=n),
                'wrong_fragment': np.random.poisson(lam=0.3, size=n),
                'urgent': np.random.poisson(lam=0.2, size=n),
                'hot': np.random.poisson(lam=2, size=n),
                'num_failed_logins': np.random.poisson(lam=3, size=n),  # Many failed logins
                'logged_in': np.random.binomial(1, 0.4, size=n),
                'num_compromised': np.random.poisson(lam=1, size=n),
                'count': np.random.poisson(lam=8, size=n),
                'srv_count': np.random.poisson(lam=7, size=n),
                'serror_rate': np.random.uniform(0.1, 0.3, size=n),
                'srv_serror_rate': np.random.uniform(0.1, 0.3, size=n),
                'rerror_rate': np.random.uniform(0.2, 0.5, size=n),
                'same_srv_rate': np.random.uniform(0.6, 0.9, size=n),
                'label': [attack_type] * n
            }
        
        else:  # u2r
            # User to Root attacks
            data = {
                'duration': np.random.exponential(scale=40, size=n),  # Long duration
                'src_bytes': np.random.lognormal(mean=9, sigma=2, size=n),
                'dst_bytes': np.random.lognormal(mean=7, sigma=2, size=n),
                'wrong_fragment': np.random.poisson(lam=0.2, size=n),
                'urgent': np.random.poisson(lam=0.3, size=n),
                'hot': np.random.poisson(lam=4, size=n),  # High hot indicators
                'num_failed_logins': np.random.poisson(lam=2, size=n),
                'logged_in': np.ones(n),  # Usually logged in
                'num_compromised': np.random.poisson(lam=2, size=n),  # Compromised
                'count': np.random.poisson(lam=6, size=n),
                'srv_count': np.random.poisson(lam=5, size=n),
                'serror_rate': np.random.uniform(0.0, 0.2, size=n),
                'srv_serror_rate': np.random.uniform(0.0, 0.2, size=n),
                'rerror_rate': np.random.uniform(0.0, 0.2, size=n),
                'same_srv_rate': np.random.uniform(0.7, 1.0, size=n),
                'label': [attack_type] * n
            }
        
        df_temp = pd.DataFrame(data)
        data_list.append(df_temp)
    
    # Combine all data
    df = pd.concat(data_list, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Multi-class dataset generated: {len(df)} samples")
    for attack_type in attack_types.keys():
        count = len(df[df['label'] == attack_type])
        print(f"  - {attack_type}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def main():
    """
    Main execution for traffic classification
    """
    print("\n" + "="*70)
    print("NETWORK TRAFFIC CLASSIFICATION (Multi-Class)")
    print("="*70)
    
    # Generate multi-class data
    df = generate_multiclass_traffic(n_samples=10000)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Train classifier
    classifier = TrafficClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy, y_pred = classifier.evaluate(X_test, y_test)
    
    # Plot results
    classifier.plot_class_distribution(y_train, y_test, 
                                      save_path='../docs/class_distribution.png')
    classifier.plot_confusion_matrix(y_test, y_pred,
                                    save_path='../docs/confusion_matrix_multiclass.png')
    
    print("\n" + "="*70)
    print("Traffic classification completed!")
    print("="*70)


if __name__ == "__main__":
    main()
