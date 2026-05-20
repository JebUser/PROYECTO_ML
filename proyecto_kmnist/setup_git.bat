@echo off
REM ============================================================
REM  setup_git.bat — Inicializa el repo local y sube a GitHub
REM  Ejecutar desde: proyecto_kmnist\
REM  Uso: setup_git.bat TU_USUARIO_GITHUB nombre-del-repo
REM ============================================================

SET GITHUB_USER=%1
SET REPO_NAME=%2

IF "%GITHUB_USER%"=="" (
    echo ERROR: Debes pasar tu usuario de GitHub como primer argumento.
    echo Uso: setup_git.bat TU_USUARIO nombre-del-repo
    exit /b 1
)

IF "%REPO_NAME%"=="" (
    SET REPO_NAME=kmnist-classical-ml
)

echo.
echo ==============================================
echo  Inicializando repositorio Git local
echo ==============================================
git init
git add .
git commit -m "feat: proyecto inicial KMNIST ML clasico

- Notebook principal: notebooks/ProyectoFinal_KMNIST.ipynb
- Funciones reutilizables: funciones.py
- Script de descarga del dataset: notebooks/download_data.py
- Requirements, .gitignore y README completo

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

echo.
echo ==============================================
echo  Conectando con GitHub
echo ==============================================
echo Crea el repositorio en: https://github.com/new
echo Nombre sugerido: %REPO_NAME%
echo (puedes hacerlo publico o privado)
echo.
pause

git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git
git branch -M main
git push -u origin main

echo.
echo ==============================================
echo  LISTO! Repositorio subido a:
echo  https://github.com/%GITHUB_USER%/%REPO_NAME%
echo ==============================================
