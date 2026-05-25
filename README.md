# 🎌 Proyecto Final — Aprendizaje Automático y Análisis de Datos

## Clasificación de Kuzushiji-MNIST con Machine Learning Clásico

**Pontificia Universidad Javeriana Cali — PUJC**  
**Asignatura:** Aprendizaje Automático y Análisis de Datos  
**Profesor:** Felipe Palta, M.Sc.

---

## 📋 Descripción

Clasificación del dataset **Kuzushiji-MNIST** (70,000 imágenes de 28×28 px, 10 clases de
caracteres japoneses cursivos históricos) usando exclusivamente **Machine Learning Clásico**
con extracción manual e ingenieril de características visuales.

**❌ Sin redes neuronales — ✅ Solo ML clásico**

### Modelos implementados

Perceptrón · Adaline (SGD) · Regresión Logística · SVM Lineal · SVM Polinómico · SVM RBF · KNN · Árbol de Decisión · Bosque Aleatorio

### Características extraídas

| Familia      | Técnicas                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| Intensidad   | Media, varianza, skewness, kurtosis, histograma 16-bins, ratios oscuro/brillante                         |
| Forma/Bordes | Área, perímetro, circularidad, Hu Moments (7), Canny edge ratio, zonas 4×4                               |
| Textura      | HOG (128 feat), LBP histograma (26 feat), GLCM (contraste, homogeneidad, energía, correlación, entropía) |

---

## 🚀 Instalación rápida

### Requisitos previos

- Python **3.10** o superior
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

# Activar — Windows
venv\Scripts\activate

# Activar — Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Registrar el kernel en Jupyter

```bash
python -m ipykernel install --user --name=kmnist-venv --display-name "Python (kmnist)"
```

### 5. Descargar el dataset KMNIST

```bash
python notebooks/download_data.py
```

Descarga los 4 archivos `.npz` (~50 MB) en `data/`. Los archivos **no se suben a git** (ver `.gitignore`).

### 6. Lanzar el notebook

```bash
jupyter notebook notebooks/ProyectoFinal_KMNIST.ipynb
```

> 💡 Al abrir el notebook, ve a **Kernel → Change Kernel → Python (kmnist)**

---

## ▶️ Ejecución

Una vez en Jupyter con el kernel correcto:

```
Kernel → Restart & Run All
```

O ejecuta celda por celda con `Shift + Enter`.

> ⚠️ La **Sección 4 (extracción de features)** procesa 70,000 imágenes y puede tardar **5–15 minutos** dependiendo de tu CPU.

---

## 📁 Estructura del Proyecto

```
proyecto_kmnist/
├── notebooks/
│   ├── ProyectoFinal_KMNIST.ipynb   ← NOTEBOOK PRINCIPAL (todo el proyecto)
│   └── download_data.py             ← Script de descarga del dataset
├── data/                            ← Archivos .npz (no versionados en git)
│   ├── kmnist-train-imgs.npz
│   ├── kmnist-train-labels.npz
│   ├── kmnist-test-imgs.npz
│   └── kmnist-test-labels.npz
├── outputs/
│   ├── figures/                     ← Gráficas generadas automáticamente
│   └── models/                      ← Modelos entrenados (no versionados)
├── funciones.py                     ← Funciones reutilizables (importadas por el notebook)
├── features.csv                     ← Dataset tabular de características (generado al ejecutar)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📓 Contenido del Notebook

| Sección     | Descripción                                                       |
| ----------- | ----------------------------------------------------------------- |
| **Portada** | Integrantes, institución, título                                  |
| **§1**      | Comprensión del problema — 5 preguntas respondidas                |
| **§2**      | EDA de alta profundidad (10 ítems del enunciado)                  |
| **§3**      | Preprocesamiento: normalización, Gauss, Otsu                      |
| **§4**      | Extracción de características: 3 familias (~224 features)         |
| **§5**      | Dataset tabular, análisis de correlación, guardado CSV            |
| **§6**      | División train/test estratificada + verificación                  |
| **§7**      | 9 modelos obligatorios con tabla comparativa                      |
| **§8**      | PCA: varianza, visualización 2D/3D, comparativa                   |
| **§9**      | Validación cruzada + GridSearchCV (sin data leakage)              |
| **§10**     | Pipelines obligatorios: con y sin PCA                             |
| **§11**     | Evaluación final: matriz de confusión, métricas, errores visuales |
| **§12**     | Opción de Rechazo con análisis crítico                            |

---

## 📊 Dataset KMNIST

| Archivo                   | Descripción                      | Tamaño |
| ------------------------- | -------------------------------- | ------ |
| `kmnist-train-imgs.npz`   | 60,000 imágenes de entrenamiento | ~18 MB |
| `kmnist-train-labels.npz` | Labels train                     | ~60 KB |
| `kmnist-test-imgs.npz`    | 10,000 imágenes de prueba        | ~3 MB  |
| `kmnist-test-labels.npz`  | Labels test                      | ~10 KB |

Fuente oficial: [KMNIST Dataset — CODH](http://codh.rois.ac.jp/kmnist/)

---

## 🔧 Dependencias principales

| Paquete                    | Uso                                  |
| -------------------------- | ------------------------------------ |
| `numpy`, `pandas`, `scipy` | Cálculo numérico y estadístico       |
| `scikit-learn`             | Modelos ML, pipelines, evaluación    |
| `scikit-image`             | HOG, LBP, GLCM, Canny, Otsu          |
| `opencv-python`            | Hu Moments, operaciones morfológicas |
| `matplotlib`, `seaborn`    | Visualización                        |
| `notebook`, `ipykernel`    | Jupyter                              |

Ver versiones exactas en [`requirements.txt`](requirements.txt).
