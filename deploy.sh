#!/bin/bash

# Script de déploiement automatique sur GitHub
# Usage: ./deploy.sh "message de commit"

echo "🚀 Déploiement automatique sur GitHub"
echo "======================================"

# Vérifier si un message de commit a été fourni
if [ -z "$1" ]; then
    echo "❌ Erreur: Veuillez fournir un message de commit"
    echo "Usage: ./deploy.sh \"votre message de commit\""
    exit 1
fi

COMMIT_MESSAGE="$1"

echo ""
echo "📝 Message de commit: $COMMIT_MESSAGE"
echo ""

# Vérifier si Git est initialisé
if [ ! -d ".git" ]; then
    echo "📦 Initialisation de Git..."
    git init
    git branch -M main
    echo "✅ Git initialisé"
else
    echo "✅ Git déjà initialisé"
fi

# Ajouter tous les fichiers
echo ""
echo "📁 Ajout des fichiers..."
git add .

if [ $? -eq 0 ]; then
    echo "✅ Fichiers ajoutés"
else
    echo "❌ Erreur lors de l'ajout des fichiers"
    exit 1
fi

# Créer le commit
echo ""
echo "💾 Création du commit..."
git commit -m "$COMMIT_MESSAGE"

if [ $? -eq 0 ]; then
    echo "✅ Commit créé"
else
    echo "⚠️  Aucun changement à commiter ou erreur"
fi

# Vérifier si le remote existe
if git remote | grep -q "origin"; then
    echo ""
    echo "🔗 Remote 'origin' détecté"
else
    echo ""
    echo "⚠️  Aucun remote 'origin' détecté"
    echo "📝 Configuration du remote..."
    read -p "Entrez l'URL de votre repo GitHub: " REPO_URL
    git remote add origin "$REPO_URL"
    echo "✅ Remote configuré"
fi

# Pousser vers GitHub
echo ""
echo "⬆️  Push vers GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ✅ ✅ DÉPLOIEMENT RÉUSSI ! ✅ ✅ ✅"
    echo ""
    echo "🎉 Votre code est maintenant sur GitHub !"
    echo ""
    echo "Prochaines étapes:"
    echo "1. Allez sur https://streamlit.io/cloud"
    echo "2. Connectez-vous avec GitHub"
    echo "3. Déployez votre app Streamlit"
    echo ""
else
    echo ""
    echo "❌ Erreur lors du push"
    echo ""
    echo "Solutions possibles:"
    echo "1. Vérifiez vos identifiants GitHub"
    echo "2. Vérifiez l'URL du repository"
    echo "3. Utilisez un Personal Access Token si demandé"
    exit 1
fi
