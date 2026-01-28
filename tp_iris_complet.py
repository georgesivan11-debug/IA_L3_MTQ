# TP N° 1: Classification des fleurs iris avec un modèle d'apprentissage automatique simple
# Code complet - Toutes les étapes

# ========== IMPORTATION DES BIBLIOTHÈQUES ==========
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

# ========== ÉTAPE 1 & 2 : CHARGEMENT ET EXPLORATION DES DONNÉES ==========
print("="*60)
print("ÉTAPE 1 & 2 : CHARGEMENT ET EXPLORATION DES DONNÉES")
print("="*60)

df = pd.read_csv("Iris.csv", sep=";")

# Afficher les premières lignes du jeu de données
print("\nPremières lignes du dataset :")
print(df.head())
print("\nColonnes originales :")
print(df.columns)

# Normaliser les noms des colonnes en minuscules et sans espaces
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print("\nColonnes après normalisation :")
print(df.columns.tolist())

# Statistiques descriptives pour comprendre la distribution des caractéristiques
print("\nStatistiques descriptives :")
print(df.describe())

# Visualisation de la répartition des classes
plt.figure(figsize=(8, 5))
sns.countplot(x='species', data=df)
plt.title('Distribution des espèces d\'iris')
plt.savefig('distribution_species_initial.png')
plt.show()

# ========== EXERCICE 1 : EFFECTIFS ET REPRÉSENTATIONS GRAPHIQUES ==========
print("\n" + "="*60)
print("EXERCICE 1 : EFFECTIFS ET REPRÉSENTATIONS GRAPHIQUES")
print("="*60)

# 1- Afficher l'effectif de chaque espèce dans le jeu de données
effectifs = df['species'].value_counts()
print("\nEffectifs par espèce :")
print(effectifs)

# 2- Représentations graphiques

# a) Histogramme des espèces
plt.figure(figsize=(8, 5))
effectifs.plot(kind='bar')
plt.title("Histogramme des espèces")
plt.xlabel("Espèces")
plt.ylabel("Effectif")
plt.savefig('ex1_histogramme.png')
plt.show()

# b) Diagramme circulaire des espèces
plt.figure(figsize=(8, 8))
effectifs.plot(kind='pie', autopct='%1.1f%%')
plt.title("Répartition des espèces")
plt.ylabel("")
plt.savefig('ex1_pie.png')
plt.show()

# c) Barres groupées
plt.figure(figsize=(8, 5))
plt.bar(effectifs.index, effectifs.values)
plt.title("Barres groupées des espèces")
plt.xlabel("Espèces")
plt.ylabel("Effectif")
plt.savefig('ex1_barres_groupees.png')
plt.show()

# d) Diagramme en cascades
plt.figure(figsize=(8, 5))
values = effectifs.values
cum = np.cumsum(values)
plt.bar(effectifs.index, values)
plt.plot(effectifs.index, cum, marker='o', color='red', linewidth=2)
plt.title("Diagramme en cascade des espèces")
plt.ylabel("Effectif cumulé")
plt.savefig('ex1_cascade.png')
plt.show()

# 3- Choix des colorations
plt.figure(figsize=(8, 5))
effectifs.plot(
    kind='bar',
    color=['green', 'orange', 'blue']
)
plt.title("Effectifs par espèce (coloré)")
plt.xlabel("Espèce")
plt.ylabel("Nombre d'observations")
plt.savefig('ex1_colore.png')
plt.show()

print("\n✅ Exercice 1 terminé - Meilleure représentation : histogramme ou diagramme circulaire")

# ========== EXERCICE 2 : VARIABLES QUANTITATIVES ==========
print("\n" + "="*60)
print("EXERCICE 2 : ANALYSE DES VARIABLES QUANTITATIVES")
print("="*60)

# 1- Résumé de la variable 'Petal.Length'
print("\nRésumé de 'petallength' :")
print(df['petallength'].describe())

# 2- Visualisation de la variable 'Petal.Length' par un histogramme
plt.figure(figsize=(8, 5))
plt.hist(df['petallength'], bins=10, edgecolor='black')
plt.title("Histogramme de la longueur du pétale")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Effectif")
plt.savefig('ex2_petal_length.png')
plt.show()

# 3- Même analyse pour les autres variables numériques

# a) SepalLength
print("\nRésumé de 'sepallength' :")
print(df['sepallength'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df['sepallength'], bins=10, edgecolor='black')
plt.title("Histogramme de la longueur du sépale")
plt.xlabel("Longueur du sépale (cm)")
plt.ylabel("Effectif")
plt.savefig('ex2_sepal_length.png')
plt.show()

# b) SepalWidth
print("\nRésumé de 'sepalwidth' :")
print(df['sepalwidth'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df['sepalwidth'], bins=10, edgecolor='black')
plt.title("Histogramme de la largeur du sépale")
plt.xlabel("Largeur du sépale (cm)")
plt.ylabel("Effectif")
plt.savefig('ex2_sepal_width.png')
plt.show()

# c) PetalWidth
print("\nRésumé de 'petalwidth' :")
print(df['petalwidth'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df['petalwidth'], bins=10, edgecolor='black')
plt.title("Histogramme de la largeur du pétale")
plt.xlabel("Largeur du pétale (cm)")
plt.ylabel("Effectif")
plt.savefig('ex2_petal_width.png')
plt.show()

print("\n✅ Exercice 2 terminé")

# ========== EXERCICE 3 : NUAGE DE POINTS (ÉTUDE BIVARIÉE) ==========
print("\n" + "="*60)
print("EXERCICE 3 : NUAGE DE POINTS ET CORRÉLATIONS")
print("="*60)

# 1- Représentation graphique des nuages de points pour chaque paire de variables
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species')
plt.suptitle("Nuage de points des paires de variables", y=1.02)
plt.savefig('ex3_pairplot.png')
plt.show()

print("\n✅ Exercice 3 terminé - Les nuages de points montrent une bonne séparation par espèce")

# ========== EXERCICE 4 : BOXPLOT (VARIABLE QUALITATIVE ET QUANTITATIVE) ==========
print("\n" + "="*60)
print("EXERCICE 4 : BOXPLOT PAR ESPÈCE")
print("="*60)

# 1) Longueur du pétale selon l'espèce
plt.figure(figsize=(10, 6))
df.boxplot(column="petallength", by="species")
plt.title("Longueur du pétale selon l'espèce")
plt.suptitle("")
plt.xlabel("Espèce")
plt.ylabel("Longueur du pétale (cm)")
plt.savefig('ex4_boxplot_petal_length.png')
plt.show()

# 2) Autre variable : largeur du pétale selon l'espèce
plt.figure(figsize=(10, 6))
df.boxplot(column="petalwidth", by="species")
plt.title("Largeur du pétale selon l'espèce")
plt.suptitle("")
plt.xlabel("Espèce")
plt.ylabel("Largeur du pétale (cm)")
plt.savefig('ex4_boxplot_petal_width.png')
plt.show()

print("\n✅ Exercice 4 terminé - Les boxplots confirment les différences entre espèces")

# ========== EXERCICE 5 : INTÉGRATION DE L'ESPÈCE DANS L'ANALYSE ==========
print("\n" + "="*60)
print("EXERCICE 5 : CORRÉLATIONS ET VISUALISATIONS AVANCÉES")
print("="*60)

# 2) Corrélations entre variables quantitatives
correlation = df.drop("species", axis=1).corr()
print("\nMatrice de corrélation :")
print(correlation)

# Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title("Matrice de corrélation des variables")
plt.savefig('ex5_correlation_heatmap.png')
plt.show()

# Nuage de points pétales avec distinction par espèce
plt.figure(figsize=(10, 6))
for esp in df['species'].unique():
    sous_df = df[df['species'] == esp]
    plt.scatter(
        sous_df['petallength'],
        sous_df['petalwidth'],
        label=esp,
        s=50,
        alpha=0.7
    )

plt.title("Nuage de points pétales avec distinction par espèce")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Largeur du pétale (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ex5_scatter_petales.png')
plt.show()

print("\n✅ Exercice 5 terminé")

# ========== ÉTAPE 3 : PRÉPARATION DES DONNÉES ==========
print("\n" + "="*60)
print("ÉTAPE 3 : PRÉPARATION DES DONNÉES POUR LE MACHINE LEARNING")
print("="*60)

# Séparer caractéristiques (X) et cible (y)
X = df.drop('species', axis=1)
y = df['species']

print(f"\nDimensions X (caractéristiques) : {X.shape}")
print(f"Dimensions y (cible) : {y.shape}")
print(f"Classes : {y.unique()}")

# Division train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTaille ensemble d'entraînement : {len(X_train)}")
print(f"Taille ensemble de test : {len(X_test)}")

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Normalisation effectuée avec StandardScaler")

# ========== ÉTAPE 4 : CRÉATION ET ENTRAÎNEMENT DU MODÈLE KNN ==========
print("\n" + "="*60)
print("ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE K-NEAREST NEIGHBORS")
print("="*60)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
print("\n✅ Modèle KNN entraîné avec k=3")

# ========== ÉTAPE 5 : ÉVALUATION DU MODÈLE ==========
print("\n" + "="*60)
print("ÉTAPE 5 : ÉVALUATION DU MODÈLE KNN")
print("="*60)

# Prédictions
y_pred = knn.predict(X_test_scaled)

# Exactitude
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Exactitude du modèle KNN : {accuracy * 100:.2f}%")

# Rapport de classification
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=df['species'].unique(), 
            yticklabels=df['species'].unique())
plt.title('Matrice de confusion - KNN (k=3)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.savefig('confusion_matrix_knn.png')
plt.show()

print("\n✅ Évaluation terminée - Matrice de confusion sauvegardée")

# ========== ÉTAPE 6 : INTERPRÉTATION DES RÉSULTATS ==========
print("\n" + "="*60)
print("ÉTAPE 6 : INTERPRÉTATION DES RÉSULTATS")
print("="*60)

print("\n📝 Analyse de la matrice de confusion :")
print(f"   - Nombre total de prédictions : {len(y_test)}")
print(f"   - Nombre de prédictions correctes : {np.sum(y_pred == y_test)}")
print(f"   - Nombre d'erreurs : {np.sum(y_pred != y_test)}")

if np.sum(y_pred != y_test) > 0:
    print("\n   Erreurs détectées dans les prédictions.")
    print("   La normalisation a permis de mettre toutes les variables sur la même échelle,")
    print("   ce qui améliore les performances du modèle KNN basé sur les distances.")
else:
    print("\n   ✅ Aucune erreur - Performance parfaite !")

# ========== ÉTAPE 7 : OPTIMISATION DES HYPER-PARAMÈTRES ==========
print("\n" + "="*60)
print("ÉTAPE 7.1 : OPTIMISATION DES HYPER-PARAMÈTRES KNN")
print("="*60)

# Grid Search pour KNN
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

print("\nRecherche des meilleurs hyper-paramètres...")
print("Paramètres testés :")
print(f"  - Nombre de voisins (k) : {param_grid_knn['n_neighbors']}")
print(f"  - Distances : {param_grid_knn['metric']}")

grid_search_knn = GridSearchCV(
    KNeighborsClassifier(), 
    param_grid_knn, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_knn.fit(X_train_scaled, y_train)

print(f"\n🏆 Meilleurs paramètres KNN : {grid_search_knn.best_params_}")
print(f"📈 Meilleur score (validation croisée) : {grid_search_knn.best_score_:.4f}")

# Modèle optimisé
best_knn = grid_search_knn.best_estimator_
y_pred_best = best_knn.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"🎯 Exactitude sur test (KNN optimisé) : {accuracy_best * 100:.2f}%")

# ========== ÉTAPE 7.2 : COMPARAISON AVEC D'AUTRES MODÈLES ==========
print("\n" + "="*60)
print("ÉTAPE 7.2 : COMPARAISON AVEC D'AUTRES MODÈLES")
print("="*60)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
}

results = {}

print("\nEntraînement et évaluation de 6 modèles différents...\n")

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
    
    print(f"{'='*50}")
    print(f"{name}:")
    print(f"  Exactitude test : {acc * 100:.2f}%")
    print(f"  CV moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Visualisation des résultats
print("\n" + "="*60)
print("VISUALISATION DES PERFORMANCES")
print("="*60)

model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'pink'])
plt.title('Comparaison des performances des modèles', fontsize=14, fontweight='bold')
plt.xlabel('Modèles')
plt.ylabel('Exactitude')
plt.ylim([0.85, 1.0])
plt.xticks(rotation=45, ha='right')

# Ajouter les valeurs sur les barres
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300)
plt.show()

# Meilleur modèle
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"   Exactitude : {results[best_model_name]['accuracy'] * 100:.2f}%")
print(f"   CV moyenne : {results[best_model_name]['cv_mean']:.4f}")

# ========== SAUVEGARDE DU MEILLEUR MODÈLE ==========
print("\n" + "="*60)
print("SAUVEGARDE DU MODÈLE POUR DÉPLOIEMENT")
print("="*60)

# Sauvegarder le meilleur modèle et le scaler
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Fichiers sauvegardés :")
print("   - best_model.pkl (modèle entraîné)")
print("   - scaler.pkl (normalisation)")
print("\n💡 Ces fichiers peuvent être utilisés pour le déploiement avec Flask et Streamlit")

# ========== RÉSUMÉ FINAL ==========
print("\n" + "="*60)
print("🎉 TP TERMINÉ AVEC SUCCÈS !")
print("="*60)

print("\n📊 RÉSUMÉ DES ÉTAPES RÉALISÉES :")
print("   ✅ Exercice 1 : Visualisation des effectifs par espèce")
print("   ✅ Exercice 2 : Analyse des variables quantitatives")
print("   ✅ Exercice 3 : Nuages de points et corrélations")
print("   ✅ Exercice 4 : Boxplots par espèce")
print("   ✅ Exercice 5 : Corrélations et visualisations avancées")
print("   ✅ Étape 3 : Préparation des données")
print("   ✅ Étape 4 : Entraînement du modèle KNN")
print("   ✅ Étape 5 : Évaluation du modèle")
print("   ✅ Étape 6 : Interprétation des résultats")
print("   ✅ Étape 7.1 : Optimisation des hyper-paramètres")
print("   ✅ Étape 7.2 : Comparaison avec 6 modèles différents")
print("   ✅ Sauvegarde des modèles pour déploiement")

print("\n🚀 PROCHAINES ÉTAPES :")
print("   1. Déploiement avec Flask (API REST)")
print("   2. Interface Streamlit (Dashboard interactif)")
print("   3. Publication sur GitHub et Streamlit Cloud")

print("\n" + "="*60)
print("Merci d'avoir suivi ce TP ! 🌸")
print("="*60)
