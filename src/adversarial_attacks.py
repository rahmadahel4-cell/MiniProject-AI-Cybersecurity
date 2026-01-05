"""
Adversarial Attacks on AI-Based Intrusion Detection
Demonstrating vulnerabilities of ML models to adversarial examples
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AdversarialAttackSimulator:
    """
    Simulate adversarial attacks on IDS models
    """
    
    def __init__(self, model, scaler):
        """
        Initialize adversarial attack simulator
        
        Args:
            model: Trained IDS model
            scaler: Fitted StandardScaler
        """
        self.model = model
        self.scaler = scaler
    
    def fgsm_attack(self, X, epsilon=0.1):
        """
        Fast Gradient Sign Method (FGSM) attack simulation
        
        Args:
            X: Original features
            epsilon: Perturbation magnitude
            
        Returns:
            Perturbed features
        """
        # Add small random perturbations
        perturbation = np.random.randn(*X.shape) * epsilon
        X_adversarial = X + perturbation
        
        # Ensure non-negative values for meaningful features
        X_adversarial = np.maximum(X_adversarial, 0)
        
        return X_adversarial
    
    def random_noise_attack(self, X, noise_level=0.05):
        """
        Add random noise to features
        
        Args:
            X: Original features
            noise_level: Proportion of noise to add
            
        Returns:
            Noisy features
        """
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + X * noise
        X_noisy = np.maximum(X_noisy, 0)
        
        return X_noisy
    
    def feature_manipulation_attack(self, X, feature_indices, manipulation_factor=0.5):
        """
        Manipulate specific features to evade detection
        
        Args:
            X: Original features
            feature_indices: Indices of features to manipulate
            manipulation_factor: How much to change features
            
        Returns:
            Manipulated features
        """
        X_manipulated = X.copy()
        
        for idx in feature_indices:
            # Reduce suspicious feature values
            X_manipulated[:, idx] = X_manipulated[:, idx] * manipulation_factor
        
        return X_manipulated
    
    def evaluate_robustness(self, X_test, y_test, attack_type='fgsm', **kwargs):
        """
        Evaluate model robustness against adversarial attacks
        
        Args:
            X_test: Test features
            y_test: True labels
            attack_type: Type of attack ('fgsm', 'random_noise', 'feature_manipulation')
            **kwargs: Additional parameters for specific attacks
            
        Returns:
            Results dictionary
        """
        # Convert to numpy if pandas DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values
        else:
            X_test_np = X_test
        
        # Original predictions
        X_test_scaled = self.scaler.transform(X_test_np)
        y_pred_original = self.model.predict(X_test_scaled)
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            epsilon = kwargs.get('epsilon', 0.1)
            X_adversarial = self.fgsm_attack(X_test_np, epsilon)
        elif attack_type == 'random_noise':
            noise_level = kwargs.get('noise_level', 0.05)
            X_adversarial = self.random_noise_attack(X_test_np, noise_level)
        elif attack_type == 'feature_manipulation':
            feature_indices = kwargs.get('feature_indices', [0, 1, 2])
            manipulation_factor = kwargs.get('manipulation_factor', 0.5)
            X_adversarial = self.feature_manipulation_attack(
                X_test_np, feature_indices, manipulation_factor
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Predictions on adversarial examples
        X_adversarial_scaled = self.scaler.transform(X_adversarial)
        y_pred_adversarial = self.model.predict(X_adversarial_scaled)
        
        # Calculate metrics
        if hasattr(self.model, 'classes_'):
            # For sklearn models
            original_accuracy = np.mean(y_pred_original == y_test)
            adversarial_accuracy = np.mean(y_pred_adversarial == y_test)
        else:
            original_accuracy = np.mean(y_pred_original == y_test)
            adversarial_accuracy = np.mean(y_pred_adversarial == y_test)
        
        # Calculate evasion rate (attacks that changed predictions)
        prediction_changes = np.sum(y_pred_original != y_pred_adversarial)
        evasion_rate = prediction_changes / len(y_test)
        
        # Calculate successful evasions (attacks predicted as normal that were originally attacks)
        if isinstance(y_test, pd.Series):
            y_test_binary = (y_test == 'attack').values
        else:
            y_test_binary = y_test
        
        successful_evasions = np.sum(
            (y_pred_original == y_test_binary) & 
            (y_pred_adversarial != y_test_binary)
        )
        
        results = {
            'attack_type': attack_type,
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'accuracy_drop': original_accuracy - adversarial_accuracy,
            'evasion_rate': evasion_rate,
            'successful_evasions': successful_evasions,
            'total_samples': len(y_test)
        }
        
        return results
    
    def print_robustness_report(self, results):
        """
        Print robustness evaluation report
        """
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL ROBUSTNESS EVALUATION")
        print(f"Attack Type: {results['attack_type'].upper()}")
        print(f"{'='*60}")
        print(f"Original Accuracy:      {results['original_accuracy']:.4f} ({results['original_accuracy']*100:.2f}%)")
        print(f"Adversarial Accuracy:   {results['adversarial_accuracy']:.4f} ({results['adversarial_accuracy']*100:.2f}%)")
        print(f"Accuracy Drop:          {results['accuracy_drop']:.4f} ({results['accuracy_drop']*100:.2f}%)")
        print(f"Evasion Rate:           {results['evasion_rate']:.4f} ({results['evasion_rate']*100:.2f}%)")
        print(f"Successful Evasions:    {results['successful_evasions']} / {results['total_samples']}")


def demonstrate_adversarial_attacks():
    """
    Demonstrate adversarial attacks on IDS
    """
    print("\n" + "="*70)
    print("ADVERSARIAL ATTACKS DEMONSTRATION")
    print("="*70)
    
    # Import and generate data
    from intrusion_detection import IntrusionDetectionSystem, generate_synthetic_network_traffic
    
    df = generate_synthetic_network_traffic(n_samples=5000, attack_ratio=0.3)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining baseline model...")
    ids = IntrusionDetectionSystem(algorithm='random_forest')
    ids.train(X_train, y_train)
    
    # Evaluate original performance
    metrics, y_pred = ids.evaluate(X_test, y_test)
    print(f"\nBaseline Accuracy: {metrics['accuracy']:.4f}")
    
    # Initialize adversarial attack simulator
    simulator = AdversarialAttackSimulator(ids.model, ids.scaler)
    
    # Test different attacks
    attack_types = [
        ('fgsm', {'epsilon': 0.1}),
        ('random_noise', {'noise_level': 0.05}),
        ('feature_manipulation', {'feature_indices': [0, 1, 2, 5], 'manipulation_factor': 0.5})
    ]
    
    all_results = []
    
    for attack_type, params in attack_types:
        print(f"\n{'#'*70}")
        print(f"Testing {attack_type.upper().replace('_', ' ')} Attack")
        print(f"{'#'*70}")
        
        # Convert labels for binary evaluation
        y_test_binary = ids.label_encoder.transform(y_test)
        
        results = simulator.evaluate_robustness(
            X_test, y_test_binary, 
            attack_type=attack_type, 
            **params
        )
        
        simulator.print_robustness_report(results)
        all_results.append(results)
    
    # Plot comparison
    plot_adversarial_comparison(all_results, save_path='../docs/adversarial_attacks_comparison.png')
    
    print("\n" + "="*70)
    print("Adversarial attacks demonstration completed!")
    print("="*70)


def plot_adversarial_comparison(results, save_path=None):
    """
    Plot comparison of different adversarial attacks
    """
    attack_types = [r['attack_type'].replace('_', '\n') for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    original_acc = [r['original_accuracy'] for r in results]
    adversarial_acc = [r['adversarial_accuracy'] for r in results]
    
    x = np.arange(len(attack_types))
    width = 0.35
    
    axes[0].bar(x - width/2, original_acc, width, label='Original', 
               color='green', alpha=0.7, edgecolor='black')
    axes[0].bar(x + width/2, adversarial_acc, width, label='After Attack',
               color='red', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy: Original vs After Attack', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(attack_types, fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Evasion rate
    evasion_rates = [r['evasion_rate'] for r in results]
    colors = ['#e74c3c', '#e67e22', '#f39c12']
    
    bars = axes[1].bar(attack_types, evasion_rates, color=colors, 
                      alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Evasion Rate', fontsize=12)
    axes[1].set_title('Attack Evasion Rate', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAdversarial attacks comparison saved to {save_path}")
    
    return plt


if __name__ == "__main__":
    demonstrate_adversarial_attacks()
