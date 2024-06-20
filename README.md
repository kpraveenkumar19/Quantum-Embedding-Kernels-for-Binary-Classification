# Quantum Embedding Kernels for Binary Classification

This project demonstrates the use of Quantum Embedding Kernels (QEKs) for binary classification on the make_moons dataset using parameterized quantum circuits to embed data into quantum states and train a Support Vector Machine (SVM) classifier on the resulting kernel matrix.

## Problem Statement

Develop a quantum machine learning model using QEKs for a binary classification task on a chosen dataset.

### Expected Outcome:
1. Implement a parameterized quantum circuit for data embedding.
2. Compute the QEK matrix for the dataset.
3. Optimize the variational parameters using kernel-target alignment.
4. Apply noise mitigation techniques.
5. Train an SVM classifier using the optimized QEK matrix.
6. Evaluate the classifier's accuracy on a test set.

# Conclusion

This project successfully demonstrates the application of Quantum Embedding Kernels for binary classification on the make_moons dataset. The implementation includes data embedding into quantum states, optimization of variational parameters, computation of the QEK matrix, and training of an SVM classifier. The resulting classification accuracy and decision boundary visualization confirm the effectiveness of the quantum machine learning model.

# Installation and Code Overview

```bash
#Install the required packages:
pip install pennylane pennylane-lightning numpy scikit-learn scipy joblib matplotlib

#Step 1: Import Libraries

import time
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#Step 2: Load and Visualize Dataset

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('make_moons Dataset')
plt.show()

#Step 3: Define Quantum Circuit

n_qubits = X.shape[1]
dev = qml.device('default.qubit', wires=n_qubits)

def quantum_feature_map(x, params):
    for i in range(n_qubits):
        qml.RY(x[i % len(x)], wires=i)
    for i in range(n_qubits):
        qml.RZ(params[i], wires=i)
    qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")

@qml.qnode(dev)
def variational_circuit(x, params):
    quantum_feature_map(x, params)
    return qml.state()


#Step 4: Initialize Parameters and Compute QEK Matrix

np.random.seed(42)
params = np.random.uniform(0, np.pi, n_qubits)

def compute_qek_matrix(X, params):
    n_samples = len(X)
    qek_matrix = np.zeros((n_samples, n_samples))

    def compute_element(i, j):
        state_i = variational_circuit(X[i], params)
        state_j = variational_circuit(X[j], params)
        return np.abs(np.dot(np.conj(state_i), state_j))**2

    results = Parallel(n_jobs=-1)(delayed(compute_element)(i, j) for i in range(n_samples) for j in range(n_samples))
    qek_matrix = np.array(results).reshape(n_samples, n_samples)
    return qek_matrix

qek_matrix = compute_qek_matrix(X, params)

#Step 5: Optimize Parameters

def kernel_target_alignment(params, X, y):
    qek_matrix = compute_qek_matrix(X, params)
    y_matrix = np.outer(y, y)
    return -np.sum(qek_matrix * y_matrix)

result = minimize(kernel_target_alignment, params, args=(X, y), method='COBYLA')
optimized_params = result.x

optimized_qek_matrix = compute_qek_matrix(X, optimized_params)

#Step 6: Train and Evaluate SVM Classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_indices = np.array([np.where((X == train_instance).all(axis=1))[0][0] for train_instance in X_train])
test_indices = np.array([np.where((X == test_instance).all(axis=1))[0][0] for test_instance in X_test])

K_train = optimized_qek_matrix[np.ix_(train_indices, train_indices)]
K_test = optimized_qek_matrix[np.ix_(test_indices, train_indices)]

svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

accuracy = svm.score(K_test, y_test)
print(f'Classification accuracy: {accuracy:.2f}')

#Step 7: Plot Decision Boundary

h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def predict_with_quantum_kernel(X_train, X_new, optimized_qek_matrix):
    n_train = len(X_train)
    n_new = len(X_new)
    K_new = np.zeros((n_new, n_train))

    for i in range(n_new):
        for j in range(n_train):
            state_i = variational_circuit(X_new[i], optimized_params)
            state_j = variational_circuit(X_train[j], optimized_params)
            K_new[i, j] = np.abs(np.dot(np.conj(state_i), state_j))**2

    return K_new

Z = predict_with_quantum_kernel(X_train, np.c_[xx.ravel(), yy.ravel()], optimized_qek_matrix)
Z = svm.predict(Z)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Quantum Kernel (make_moons Dataset)')
plt.show()






