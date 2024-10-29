---
# Urysohn's Binary Classifier
---

## Overview
This repository presents an innovative approach to binary classification using **Urysohn’s Lemma of Topology**. By leveraging this topological principle, the Urysohn Binary Classifier constructs continuous separating functions to distinguish between classes with high precision. The classifier is robust and has demonstrated superior performance against popular methods like CatBoost and K-Nearest Neighbors (KNN), especially in complex, high-dimensional datasets.

## Key Features
- **Topology-Based Classifier**: Uses Urysohn’s Lemma to create continuous separating functions for binary classification.
- **High Performance**: Achieves an accuracy range between 95% to 100% on benchmark datasets.
- **Interpretable Model**: Provides a mathematically principled approach to separating data classes with clarity.
- **Versatile Applications**: Effective in domains like medical diagnostics, fraud detection, and cyber security.

## Structure
The repository includes:
- **Classifier Code**: Implementation of Urysohn's Binary Classifier.
- **Jupyter Notebook**: Documentation of experiments, including comparisons with other classifiers.
- **Datasets**: Links and setup instructions for the datasets used in benchmarking, such as the Breast Cancer Wisconsin, Chronic Kidney Disease, and Banknotes datasets.
  
## Installation
To use the Urysohn Binary Classifier, clone the repository and install the required dependencies:
```bash
git clone https://github.com/username/Urysohns-Binary-Classifier.git
cd Urysohns-Binary-Classifier
pip install -r requirements.txt
```

## Usage
1. **Train the Model**:
    ```python
    from urysohn_classifier import UrysohnClassifier
    classifier = UrysohnClassifier(metric='manhattan', p=1, epsilon=0)
    classifier.fit(X_train, y_train)
    ```

2. **Make Predictions**:
    ```python
    predictions = classifier.predict(X_test)
    ```

3. **Evaluate the Model**:
    ```python
    from sklearn.metrics import accuracy_score, roc_auc_score
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("ROC AUC:", roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1]))
    ```

## Experimental Results
The classifier was tested on three datasets, achieving high performance across metrics such as accuracy, ROC-AUC, precision, and F1 score. Notably:
- **Breast Cancer Wisconsin Dataset**: Achieved 95% accuracy.
- **Chronic Kidney Disease Dataset**: Achieved 99% accuracy, comparable to CatBoost.
- **Banknotes Dataset**: Perfect classification with 100% accuracy.

The classifier also demonstrated stability and robustness to noise, varying class imbalance, and sensitivity to different metrics (p-metric parameter tuning).

## Performance Comparison
Urysohn's classifier performed comparably to CatBoost while remaining simpler and more interpretable. It outperformed KNN under conditions of class imbalance, showing promise in high-stakes applications.

## Future Directions
- **Parameter Tuning**: Further research into p-metric parameter sensitivity.
- **Multiclass Extension**: Generalizing to multiclass classification and regression.
- **Hybrid Approaches**: Combining Urysohn’s classifier with ensemble methods.

## Citation
If you use this work, please cite:
```
@article{lopez2023urysohn,
  title={Leveraging the Urysohn Lemma of Topology for an Enhanced Binary Classifier},
  author={E. López Fune},
  journal={arXiv preprint arXiv:2312.11948},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
