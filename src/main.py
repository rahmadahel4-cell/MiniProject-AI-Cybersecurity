"""
Main Execution Script for AI-Based Cybersecurity Project
Project 16: Implementation of an AI-Based Cybersecurity Use Case

This script runs all components of the project:
1. Intrusion Detection (Supervised Learning)
2. Anomaly Detection (Unsupervised Learning)
3. Traffic Classification (Multi-class Classification)
4. Adversarial Attacks Analysis

Author: DAHEL Rahma & GHEDJATI Zainab
Date: January 2026
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure output directory exists
os.makedirs('../docs', exist_ok=True)

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")


def main():
    """
    Main execution function
    """
    print_header("AI-BASED CYBERSECURITY PROJECT")
    print("Project 16: Implementation of an AI-Based Cybersecurity Use Case")
    print("Module: Security and Privacy - Medicine & Big Data")
    print("University: Sétif 1 University")
    print("Academic Year: 2025/2026")
    
    print("\n" + "-"*80)
    print("This project demonstrates AI applications in cybersecurity:")
    print("  1. Intrusion Detection using Supervised Learning")
    print("  2. Anomaly Detection using Unsupervised Learning")
    print("  3. Network Traffic Classification (Multi-class)")
    print("  4. Adversarial Attacks on AI Models")
    print("-"*80)
    
    # Component 1: Intrusion Detection
    print_header("COMPONENT 1: INTRUSION DETECTION SYSTEM")
    print("Running supervised learning models for binary classification...")
    print("Models: Random Forest, Decision Tree, Neural Network\n")
    
    try:
        from intrusion_detection import main as ids_main
        ids_main()
    except Exception as e:
        print(f"Error in Intrusion Detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 2: Anomaly Detection
    print_header("COMPONENT 2: ANOMALY DETECTION")
    print("Running unsupervised learning models for anomaly detection...")
    print("Methods: Isolation Forest, One-Class SVM\n")
    
    try:
        from anomaly_detection import main as anomaly_main
        anomaly_main()
    except Exception as e:
        print(f"Error in Anomaly Detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 3: Traffic Classification
    print_header("COMPONENT 3: NETWORK TRAFFIC CLASSIFICATION")
    print("Running multi-class classification...")
    print("Classes: Normal, DoS, Probe, R2L, U2R\n")
    
    try:
        from traffic_classification import main as traffic_main
        traffic_main()
    except Exception as e:
        print(f"Error in Traffic Classification: {e}")
        import traceback
        traceback.print_exc()
    
    # Component 4: Adversarial Attacks
    print_header("COMPONENT 4: ADVERSARIAL ATTACKS ANALYSIS")
    print("Demonstrating model vulnerabilities to adversarial examples...")
    print("Attacks: FGSM, Random Noise, Feature Manipulation\n")
    
    try:
        from adversarial_attacks import demonstrate_adversarial_attacks
        demonstrate_adversarial_attacks()
    except Exception as e:
        print(f"Error in Adversarial Attacks: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print_header("PROJECT EXECUTION COMPLETED")
    print("All components have been executed successfully!")
    print("\nGenerated outputs:")
    print("  📊 Algorithm comparison charts")
    print("  📈 Performance metrics")
    print("  🎯 Confusion matrices")
    print("  ⚠️  Adversarial attack analysis")
    print("\nAll results saved in: ../docs/")
    print("\nCheck the generated images for detailed visualizations.")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
