import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

def run_algorithms_with_data_sizes(data_sizes):
    logistic_training_times = []
    svm_training_times = []
    logistic_prediction_times = []
    svm_prediction_times = []
    logistic_accuracies = []
    svm_accuracies = []

    for size in data_sizes:
        # Generar datos para el problema de prueba
        X, y = make_classification(n_samples=size, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Logistic Regression (Binary)
        start_time = time.time()
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        logistic_training_time = time.time() - start_time
        logistic_training_times.append(logistic_training_time)

        # Support Vector Machines (SVMs)
        start_time = time.time()
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        svm_training_time = time.time() - start_time
        svm_training_times.append(svm_training_time)

        # Evaluar modelos
        logistic_predictions = logistic_model.predict(X_test)
        svm_predictions = svm_model.predict(X_test)

        # Calcular tiempo de predicción
        start_time = time.time()
        _ = logistic_model.predict(X_test)
        logistic_predict_time = time.time() - start_time
        logistic_prediction_times.append(logistic_predict_time)

        start_time = time.time()
        _ = svm_model.predict(X_test)
        svm_predict_time = time.time() - start_time
        svm_prediction_times.append(svm_predict_time)

        # Calcular precisión
        logistic_accuracy = accuracy_score(y_test, logistic_predictions)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        logistic_accuracies.append(logistic_accuracy)
        svm_accuracies.append(svm_accuracy)

    return logistic_training_times, svm_training_times, logistic_prediction_times, svm_prediction_times, logistic_accuracies, svm_accuracies


# Definir diferentes tamaños de datos
data_sizes = [100, 500, 1000, 5000, 10000]

logistic_train_times, svm_train_times, logistic_predict_times, svm_predict_times, logistic_accuracies, svm_accuracies = run_algorithms_with_data_sizes(data_sizes)

# Imprimir resultados
for i, size in enumerate(data_sizes):
    print(f"Resultados para {size} muestras:")
    print("Logistic Regression (Binary) - Precisión:", logistic_accuracies[i])
    print("Support Vector Machines (SVMs) - Precisión:", svm_accuracies[i])
    print("Tiempo de entrenamiento:")
    print("Logistic Regression (Binary):", logistic_train_times[i], "segundos")
    print("Support Vector Machines (SVMs):", svm_train_times[i], "segundos")
    print("Tiempo de predicción:")
    print("Logistic Regression (Binary):", logistic_predict_times[i], "segundos")
    print("Support Vector Machines (SVMs):", svm_predict_times[i], "segundos")
    print()

# Graficar tiempos de entrenamiento y predicción
plt.figure(figsize=(10, 5))
plt.plot(data_sizes, logistic_train_times, marker='o', linestyle='-', color='b', label='Logistic Regression (Binary) - Tiempo de Entrenamiento')
plt.plot(data_sizes, svm_train_times, marker='o', linestyle='-', color='g', label='Support Vector Machines (SVMs) - Tiempo de Entrenamiento')
plt.plot(data_sizes, logistic_predict_times, marker='o', linestyle='-', color='r', label='Logistic Regression (Binary) - Tiempo de Predicción')
plt.plot(data_sizes, svm_predict_times, marker='o', linestyle='-', color='y', label='Support Vector Machines (SVMs) - Tiempo de Predicción')
plt.xlabel('Tamaño de los datos')
plt.ylabel('Tiempo (segundos)')
plt.title('Tiempo de Entrenamiento y Predicción por Tamaño de los Datos')
plt.legend()
plt.grid(True)
plt.show()
