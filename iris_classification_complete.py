# TP Classification des Iris - Code Complet
# Étapes 1-7 du TP

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ========== ÉTAPE 1 & 2 : Chargement et Exploration ==========
print("="*50)
print("CHARGEMENT ET EXPLORATION DES DONNÉES")
print("="*50)

df = pd.read_csv("Iris.csv", sep=";")
print("\nPremières lignes :")
print(df.head())

# Normaliser les noms des colonnes
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print("\nColonnes après normalisation :", df.columns.tolist())

print("\nStatistiques descriptives :")
print(df.describe())

# Visualisation de la répartition des classes
plt.figure(figsize=(8, 5))
sns.countplot(x='species', data=df)
plt.title('Distribution des espèces d\'iris')
plt.savefig('distribution_species.png')
plt.close()

# ========== ÉTAPE 3 : Préparation des Données ==========
print("\n" + "="*50)
print("PRÉPARATION DES DONNÉES")
print("="*50)

# Séparer caractéristiques (X) et cible (y)
X = df.drop('species', axis=1)
y = df['species']

print(f"\nDimensions X: {X.shape}")
print(f"Dimensions y: {y.shape}")
print(f"Classes: {y.unique()}")

# Division train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTaille ensemble d'entraînement: {len(X_train)}")
print(f"Taille ensemble de test: {len(X_test)}")

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nNormalisation effectuée ✓")

# ========== ÉTAPE 4 : Création et Entraînement du Modèle KNN ==========
print("\n" + "="*50)
print("ENTRAÎNEMENT DU MODÈLE KNN")
print("="*50)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
print("\nModèle KNN entraîné avec k=3 ✓")

# ========== ÉTAPE 5 : Évaluation du Modèle ==========
print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE KNN")
print("="*50)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Exactitude
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitude du modèle KNN: {accuracy * 100:.2f}%")

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=df['species'].unique(), 
            yticklabels=df['species'].unique())
plt.title('Matrice de confusion - KNN')
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.savefig('confusion_matrix_knn.png')
plt.close()
print("\nMatrice de confusion sauvegardée ✓")

# ========== ÉTAPE 7 : Optimisation et Comparaison ==========
print("\n" + "="*50)
print("OPTIMISATION DES HYPER-PARAMÈTRES KNN")
print("="*50)

# Grid Search pour KNN
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search_knn = GridSearchCV(
    KNeighborsClassifier(), 
    param_grid_knn, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

grid_search_knn.fit(X_train_scaled, y_train)

print(f"\nMeilleurs paramètres KNN: {grid_search_knn.best_params_}")
print(f"Meilleur score (validation croisée): {grid_search_knn.best_score_:.4f}")

# Modèle optimisé
best_knn = grid_search_knn.best_estimator_
y_pred_best = best_knn.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Exactitude sur test (KNN optimisé): {accuracy_best * 100:.2f}%")

# ========== COMPARAISON AVEC D'AUTRES MODÈLES ==========
print("\n" + "="*50)
print("COMPARAISON AVEC D'AUTRES MODÈLES")
print("="*50)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    # Entraînement
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Exactitude
    acc = accuracy_score(y_test, y_pred)
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  Exactitude test: {acc * 100:.2f}%")
    print(f"  CV moyenne: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Visualisation des résultats
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'pink'])
plt.title('Comparaison des performances des modèles')
plt.xlabel('Modèles')
plt.ylabel('Exactitude')
plt.ylim([0.8, 1.0])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('models_comparison.png')
plt.close()

print("\n" + "="*50)
print("MEILLEUR MODÈLE")
print("="*50)
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nLe meilleur modèle est: {best_model_name}")
print(f"Exactitude: {results[best_model_name]['accuracy'] * 100:.2f}%")

# ========== SAUVEGARDE DU MEILLEUR MODÈLE ==========
print("\n" + "="*50)
print("SAUVEGARDE DU MODÈLE")
print("="*50)

# Sauvegarder le meilleur modèle et le scaler
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModèle et scaler sauvegardés ✓")
print("  - best_model.pkl")
print("  - scaler.pkl")

print("\n" + "="*50)
print("TP TERMINÉ AVEC SUCCÈS ! 🎉")
print("="*50)
