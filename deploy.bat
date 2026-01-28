@echo off
REM Script de déploiement automatique sur GitHub (Windows)
REM Usage: deploy.bat "message de commit"

echo ========================================
echo   Deploiement automatique sur GitHub
echo ========================================
echo.

if "%~1"=="" (
    echo [ERREUR] Veuillez fournir un message de commit
    echo Usage: deploy.bat "votre message de commit"
    exit /b 1
)

set COMMIT_MESSAGE=%~1

echo Message de commit: %COMMIT_MESSAGE%
echo.

REM Vérifier si Git est initialisé
if not exist ".git" (
    echo [INIT] Initialisation de Git...
    git init
    git branch -M main
    echo [OK] Git initialise
) else (
    echo [OK] Git deja initialise
)

echo.
echo [ADD] Ajout des fichiers...
git add .

if %errorlevel% equ 0 (
    echo [OK] Fichiers ajoutes
) else (
    echo [ERREUR] Erreur lors de l'ajout des fichiers
    exit /b 1
)

echo.
echo [COMMIT] Creation du commit...
git commit -m "%COMMIT_MESSAGE%"

if %errorlevel% equ 0 (
    echo [OK] Commit cree
) else (
    echo [WARN] Aucun changement a commiter
)

echo.
echo [PUSH] Push vers GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   DEPLOIEMENT REUSSI !
    echo ========================================
    echo.
    echo Votre code est maintenant sur GitHub !
    echo.
    echo Prochaines etapes:
    echo 1. Allez sur https://streamlit.io/cloud
    echo 2. Connectez-vous avec GitHub
    echo 3. Deployez votre app Streamlit
    echo.
) else (
    echo.
    echo [ERREUR] Erreur lors du push
    echo.
    echo Solutions possibles:
    echo 1. Verifiez vos identifiants GitHub
    echo 2. Verifiez l'URL du repository
    echo 3. Configurez le remote avec: git remote add origin URL
    exit /b 1
)

pause
