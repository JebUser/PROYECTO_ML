"""
Script de configuración inicial del proyecto KMNIST.
Ejecutar una sola vez desde: c:\\Users\\Usuario\\Documents\\ML\\PROYECTO

    python setup_project.py

Crea la estructura de carpetas, el entorno virtual y el requirements.txt.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent / "proyecto_kmnist"

DIRS = [
    BASE,
    BASE / "data",
    BASE / "notebooks",
    BASE / "outputs",
    BASE / "outputs" / "figures",
    BASE / "outputs" / "models",
]

REQUIREMENTS = """\
# =============================================
# KMNIST Classical ML Project - Dependencies
# Python 3.10+
# =============================================

# Core scientific stack
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0

# Machine learning
scikit-learn==1.5.0

# Image processing
scikit-image==0.23.2
opencv-python==4.9.0.80
Pillow==10.3.0

# Visualization
matplotlib==3.9.0
seaborn==0.13.2

# Jupyter
notebook==7.2.0
ipykernel==6.29.4
ipywidgets==8.1.3

# Utilities
tqdm==4.66.4
joblib==1.4.2
"""

GITIGNORE = """\
# Virtual environment
venv/
.venv/
env/

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
*.pyo

# Data files (grandes — no subir a git)
data/*.npz
data/*.csv

# Outputs (modelos pesados)
outputs/models/*.pkl
outputs/models/*.joblib

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
"""

README = """\
# Proyecto Final — Aprendizaje Automático
## Clasificación de Kuzushiji-MNIST con ML Clásico
**PUJC — Pontificia Universidad Javeriana Cali**

---

## 📋 Descripción

Clasificación del dataset **Kuzushiji-MNIST** (70,000 imágenes, 10 clases de
caracteres japoneses cursivos) usando exclusivamente **Machine Learning Clásico**
con extracción manual de características de imagen.

**Modelos implementados:** Perceptron, Adaline, Logistic Regression, Linear SVM,
Poly SVM, RBF SVM, KNN, Decision Tree, Random Forest.

---

## 🚀 Instalación

### Requisitos previos
- Python 3.10 o superior
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/kmnist-classical-ml.git
cd kmnist-classical-ml
```

### 2. Crear y activar el entorno virtual
```bash
# Crear venv
python -m venv venv

# Activar (Windows)
venv\\Scripts\\activate

# Activar (Linux/Mac)
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset
```bash
python notebooks/download_data.py
```
Esto descarga los 4 archivos `.npz` de KMNIST en la carpeta `data/`.

### 5. Lanzar Jupyter
```bash
jupyter notebook notebooks/
```

---

## 📁 Estructura del Proyecto

```
proyecto_kmnist/
├── data/                        <- Archivos .npz del dataset (no subir a git)
├── notebooks/
│   ├── download_data.py         <- Script de descarga del dataset
│   ├── 01_eda.ipynb             <- Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb   <- Preprocesamiento de imágenes
│   ├── 03_feature_extraction.ipynb
│   ├── 04_models.ipynb
│   ├── 05_pca_pipelines.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_evaluation.ipynb
├── outputs/
│   ├── figures/                 <- Gráficas generadas
│   └── models/                  <- Modelos entrenados (.pkl)
├── funciones.py                 <- Funciones reutilizables
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Archivo                    | Descripción            |
|----------------------------|------------------------|
| kmnist-train-imgs.npz      | 60,000 imágenes train  |
| kmnist-train-labels.npz    | Labels de entrenamiento |
| kmnist-test-imgs.npz       | 10,000 imágenes test   |
| kmnist-test-labels.npz     | Labels de prueba       |

Fuente: [KMNIST Dataset](http://codh.rois.ac.jp/kmnist/)

---

## 🔬 Fases del Proyecto

| Fase | Notebook                          | Descripción                          |
|------|-----------------------------------|--------------------------------------|
| 1    | `01_eda.ipynb`                    | EDA y visualización                  |
| 2    | `02_preprocessing.ipynb`          | Normalización, suavizado, umbralizado |
| 3    | `03_feature_extraction.ipynb`     | HOG, LBP, GLCM, Hu Moments, etc.    |
| 4    | `04_models.ipynb`                 | Entrenamiento de 9 modelos           |
| 5    | `05_pca_pipelines.ipynb`          | Reducción de dimensionalidad         |
| 6    | `06_hyperparameter_tuning.ipynb`  | GridSearchCV + StratifiedKFold       |
| 7    | `07_evaluation.ipynb`             | Métricas, matrices, opción de rechazo |

---

## ⚠️ Restricciones

- ❌ Sin redes neuronales (CNN, RNN, Transformers)
- ❌ Sin TensorFlow, PyTorch, Keras
- ✅ Solo ML clásico con scikit-learn
- ✅ Extracción de características manual con skimage/cv2/scipy
"""

DOWNLOAD_SCRIPT = """\
\"\"\"
Script para descargar el dataset Kuzushiji-MNIST.
Ejecutar desde la raíz del proyecto con el venv activo:

    python notebooks/download_data.py
\"\"\"

import urllib.request
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

URLS = {
    "kmnist-train-imgs.npz":   "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz",
    "kmnist-train-labels.npz": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz",
    "kmnist-test-imgs.npz":    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz",
    "kmnist-test-labels.npz":  "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz",
}

def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✅ Ya existe: {dest.name} — omitido")
        return
    print(f"  ⬇️  Descargando {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"  ✅ {dest.name}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    print("\\n📥 Descargando Kuzushiji-MNIST...\\n")
    for filename, url in URLS.items():
        download_file(url, DATA_DIR / filename)
    print("\\n🎉 Dataset listo en:", DATA_DIR.resolve())
"""


def create_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  📄 {path.relative_to(Path(__file__).parent)}")


def create_venv(base: Path) -> None:
    venv_path = base / "venv"
    if venv_path.exists():
        print("\n⚠️  El venv ya existe — omitido.")
        return
    print("\n🔧 Creando entorno virtual...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print(f"  ✅ venv creado en: {venv_path}")


def main():
    print("🚀 Configurando proyecto KMNIST...\n")

    # Create directories
    print("📁 Creando carpetas:")
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  📂 {d.relative_to(Path(__file__).parent)}")

    # Create files
    print("\n📝 Creando archivos:")
    create_file(BASE / "requirements.txt", REQUIREMENTS)
    create_file(BASE / ".gitignore", GITIGNORE)
    create_file(BASE / "README.md", README)
    create_file(BASE / "funciones.py", "# === FUNCIONES REUTILIZABLES ===\n# Poblar en fases posteriores\n")
    create_file(BASE / "notebooks" / "download_data.py", DOWNLOAD_SCRIPT)

    # Create venv
    create_venv(BASE)

    print("\n" + "="*55)
    print("✅ Proyecto configurado exitosamente!")
    print("="*55)
    print("\n📌 Próximos pasos:\n")
    print("  1. cd proyecto_kmnist")
    print("  2. venv\\Scripts\\activate           (Windows)")
    print("     source venv/bin/activate        (Linux/Mac)")
    print("  3. pip install -r requirements.txt")
    print("  4. python notebooks/download_data.py")
    print("  5. jupyter notebook notebooks/")
    print()


if __name__ == "__main__":
    main()
