import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Generar datos para el problema de prueba
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (Binary)
start_time = time.time()
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_train_time = time.time() - start_time

# Support Vector Machines (SVMs)
start_time = time.time()
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start_time

# Evaluar modelos
logistic_predictions = logistic_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

# Calcular tiempo de predicción
start_time = time.time()
_ = logistic_model.predict(X_test)
logistic_predict_time = time.time() - start_time

start_time = time.time()
_ = svm_model.predict(X_test)
svm_predict_time = time.time() - start_time

# Calcular precisión
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Imprimir resultados
print("Resultados del problema de prueba:")
print("Logistic Regression (Binary) - Precisión:", logistic_accuracy)
print("Support Vector Machines (SVMs) - Precisión:", svm_accuracy)

# Graficar tiempos de entrenamiento y predicción
labels = ['Logistic Regression (Binary)', 'Support Vector Machines (SVMs)']
training_times = [logistic_train_time, svm_train_time]
prediction_times = [logistic_predict_time, svm_predict_time]

plt.figure(figsize=(10, 5))
plt.plot(labels, training_times, marker='o', linestyle='-', color='b', label='Tiempo de Entrenamiento')
plt.plot(labels, prediction_times, marker='o', linestyle='-', color='g', label='Tiempo de Predicción')
plt.xlabel('Algoritmo')
plt.ylabel('Tiempo (segundos)')
plt.title('Tiempo de Entrenamiento y Predicción por Algoritmo')
plt.legend()
plt.show()
