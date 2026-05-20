"""
funciones.py — Funciones reutilizables del proyecto KMNIST
Importar con: from funciones import <nombre_funcion>

Secciones:
  - CARGA DE DATOS
  - EDA — VISUALIZACIÓN
  - PREPROCESAMIENTO
  - FEATURE EXTRACTION — INTENSIDAD
  - FEATURE EXTRACTION — FORMA
  - FEATURE EXTRACTION — TEXTURA
  - FEATURE EXTRACTION — COMBINADA
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===================================================================
# === CARGA DE DATOS ================================================
# ===================================================================

def cargar_kmnist(data_dir):
    """
    Carga el dataset Kuzushiji-MNIST desde archivos .npz.

    Parameters
    ----------
    data_dir : str or Path
        Ruta a la carpeta que contiene los 4 archivos .npz del dataset.

    Returns
    -------
    X_train : np.ndarray, shape (60000, 28, 28), dtype uint8
    y_train : np.ndarray, shape (60000,), dtype uint8
    X_test  : np.ndarray, shape (10000, 28, 28), dtype uint8
    y_test  : np.ndarray, shape (10000,), dtype uint8
    """
    data_dir = Path(data_dir)

    X_train = np.load(data_dir / 'kmnist-train-imgs.npz')['arr_0']
    y_train = np.load(data_dir / 'kmnist-train-labels.npz')['arr_0']
    X_test  = np.load(data_dir / 'kmnist-test-imgs.npz')['arr_0']
    y_test  = np.load(data_dir / 'kmnist-test-labels.npz')['arr_0']

    return X_train, y_train, X_test, y_test


# ===================================================================
# === EDA — VISUALIZACIÓN ==========================================
# ===================================================================

def plot_distribucion_clases(y_train, y_test, nombres_clases, axes):
    """
    Grafica la distribución de clases para train y test en dos subplots.

    Parameters
    ----------
    y_train : np.ndarray  — Labels de entrenamiento.
    y_test  : np.ndarray  — Labels de prueba.
    nombres_clases : list[str] — Nombre legible de cada clase (len=10).
    axes : array of matplotlib Axes — Exactamente 2 axes (train, test).
    """
    import seaborn as sns

    for ax, y, titulo in zip(axes, [y_train, y_test], ['Train (60,000)', 'Test (10,000)']):
        unique, counts = np.unique(y, return_counts=True)
        colores = sns.color_palette('tab10', len(unique))
        bars = ax.bar(unique, counts, color=colores, edgecolor='black', linewidth=0.5)
        ax.set_xticks(unique)
        ax.set_xticklabels([f'{i}\n{nombres_clases[i]}' for i in unique], fontsize=7.5)
        ax.set_title(f'Distribución de Clases — {titulo}', fontweight='bold')
        ax.set_xlabel('Clase')
        ax.set_ylabel('Cantidad de imágenes')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    str(cnt), ha='center', va='bottom', fontsize=8)


def plot_muestras_por_clase(X, y, nombres_clases, n_muestras=8, seed=42):
    """
    Muestra una grilla de n_muestras imágenes por cada clase.

    Parameters
    ----------
    X : np.ndarray, shape (N, 28, 28)  — Imágenes.
    y : np.ndarray, shape (N,)          — Labels correspondientes.
    nombres_clases : list[str]          — Nombres de las clases.
    n_muestras : int                    — Columnas (muestras por clase).
    seed : int                          — Semilla para reproducibilidad.

    Returns
    -------
    fig : matplotlib Figure
    """
    rng = np.random.default_rng(seed)
    n_clases = len(np.unique(y))
    fig, axes = plt.subplots(n_clases, n_muestras, figsize=(n_muestras * 1.4, n_clases * 1.4))

    for cls in range(n_clases):
        idx = np.where(y == cls)[0]
        seleccionados = rng.choice(idx, size=n_muestras, replace=False)
        for col, img_idx in enumerate(seleccionados):
            ax = axes[cls, col]
            ax.imshow(X[img_idx], cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f'{cls}\n{nombres_clases[cls]}', fontsize=7.5,
                              rotation=0, labelpad=45, va='center')

    fig.suptitle('Muestras por Clase — Kuzushiji-MNIST', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_histogramas_intensidad(X, y, nombres_clases, axes):
    """
    Dibuja el histograma global de intensidad y los histogramas por clase superpuestos.

    Parameters
    ----------
    X : np.ndarray, shape (N, 28, 28)  — Imágenes uint8.
    y : np.ndarray, shape (N,)          — Labels.
    nombres_clases : list[str]          — Nombres de las clases.
    axes : array of matplotlib Axes     — Exactamente 2 axes.
    """
    import seaborn as sns

    # Histograma global
    axes[0].hist(X.flatten(), bins=64, color='steelblue', edgecolor='none', alpha=0.85)
    axes[0].set_title('Histograma Global de Intensidad', fontweight='bold')
    axes[0].set_xlabel('Intensidad de píxel [0-255]')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_yscale('log')

    # Histogramas por clase superpuestos (intensidad media de cada imagen)
    palette = sns.color_palette('tab10', 10)
    for cls in range(10):
        intensidades = X[y == cls].mean(axis=(1, 2))
        axes[1].hist(intensidades, bins=40, alpha=0.5, label=f'{cls}:{nombres_clases[cls]}',
                     color=palette[cls], edgecolor='none')

    axes[1].set_title('Distribución de Intensidad Media por Clase', fontweight='bold')
    axes[1].set_xlabel('Intensidad media por imagen')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend(fontsize=6.5, ncol=2)


# ===================================================================
# === FEATURE EXTRACTION — INTENSIDAD ==============================
# ===================================================================

def extraer_features_intensidad(img):
    """
    Extrae características de intensidad/color de una imagen normalizada [0,1].

    Características (21 total):
      - Media, varianza, skewness, kurtosis (4)
      - Ratio píxeles oscuros (<0.2) y brillantes (>0.8) (2)
      - Histograma de 16 bins (16) [densidad]
      - Rango dinámico: max - min (1)

    Parameters
    ----------
    img : np.ndarray, shape (28, 28), float [0,1]

    Returns
    -------
    features : np.ndarray, shape (23,)
    """
    from scipy.stats import skew, kurtosis

    flat = img.flatten()
    media    = flat.mean()
    varianza = flat.var()
    skewness = skew(flat)
    kurt     = kurtosis(flat)
    ratio_dark   = (flat < 0.2).mean()
    ratio_bright = (flat > 0.8).mean()
    hist, _ = np.histogram(flat, bins=16, range=(0, 1), density=True)
    rango   = flat.max() - flat.min()

    return np.concatenate([[media, varianza, skewness, kurt,
                            ratio_dark, ratio_bright, rango], hist])


# ===================================================================
# === FEATURE EXTRACTION — FORMA ===================================
# ===================================================================

def extraer_features_forma(img_norm, img_bin):
    """
    Extrae características de forma y bordes.

    Características:
      - Área binarizada (1)
      - Perímetro (1)
      - Circularidad = 4π·Area/Perimeter² (1)
      - Hu Moments transformados (log|Hu|) (7)
      - Ratio bordes Canny (1)
      - Descriptores por zonas 4×4 (16)

    Parameters
    ----------
    img_norm : np.ndarray, shape (28,28), float [0,1]  — imagen suavizada
    img_bin  : np.ndarray, shape (28,28), float {0,1}  — imagen binarizada (Otsu)

    Returns
    -------
    features : np.ndarray, shape (27,)
    """
    import cv2
    from skimage.measure import perimeter as sk_perimeter
    from skimage.feature import canny

    # Área
    area = img_bin.sum()

    # Perímetro y circularidad
    try:
        perim = sk_perimeter(img_bin.astype(bool))
        circularidad = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
    except Exception:
        perim, circularidad = 0.0, 0.0

    # Hu Moments (require imagen uint8)
    img_uint8 = (img_bin * 255).astype(np.uint8)
    moments   = cv2.moments(img_uint8)
    hu        = cv2.HuMoments(moments).flatten()
    hu_log    = np.sign(hu) * np.log1p(np.abs(hu))  # transformación estabilizadora

    # Ratio bordes Canny
    edges      = canny(img_norm, sigma=1.0)
    edge_ratio = edges.mean()

    # Descriptores por zonas 4×4
    zonas = img_bin.reshape(4, 7, 4, 7).mean(axis=(1, 3)).flatten()

    return np.concatenate([[area, perim, circularidad], hu_log, [edge_ratio], zonas])


# ===================================================================
# === FEATURE EXTRACTION — TEXTURA =================================
# ===================================================================

def extraer_features_textura(img_norm):
    """
    Extrae características de textura: HOG, LBP y GLCM.

    Características:
      - HOG: 128 valores (orientations=8, pixels_per_cell=(7,7), cells_per_block=(2,2))
      - LBP histograma: 26 bins (P=8, R=1, método='uniform')
      - GLCM: contraste, homogeneidad, energía, correlación, entropía × 4 ángulos = 20

    Parameters
    ----------
    img_norm : np.ndarray, shape (28,28), float [0,1]  — imagen suavizada normalizada

    Returns
    -------
    features : np.ndarray, shape (174,)
    """
    from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

    # HOG
    hog_feat = hog(img_norm, orientations=8, pixels_per_cell=(7, 7),
                   cells_per_block=(2, 2), feature_vector=True)

    # LBP
    lbp      = local_binary_pattern(img_norm, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.flatten(), bins=26, range=(0, 26), density=True)

    # GLCM — requiere uint8
    img_uint8 = (img_norm * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    propiedades = ['contrast', 'homogeneity', 'energy', 'correlation']
    glcm_feat = []
    for prop in propiedades:
        vals = graycoprops(glcm, prop).flatten()  # shape (1,4) → 4 valores
        glcm_feat.extend(vals)

    # Entropía del GLCM
    glcm_flat = glcm[:, :, 0, :].sum(axis=2)
    entropia  = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))

    return np.concatenate([hog_feat, lbp_hist, glcm_feat, [entropia]])


# ===================================================================
# === FEATURE EXTRACTION — COMBINADA ==============================
# ===================================================================

def extraer_todas_features(img_norm, img_bin):
    """
    Extrae y concatena las 3 familias de características.

    Parameters
    ----------
    img_norm : np.ndarray, shape (28,28), float [0,1]
    img_bin  : np.ndarray, shape (28,28), float {0,1}

    Returns
    -------
    features : np.ndarray — concatenación de intensidad + forma + textura
    """
    f_int   = extraer_features_intensidad(img_norm)
    f_forma = extraer_features_forma(img_norm, img_bin)
    f_tex   = extraer_features_textura(img_norm)
    return np.concatenate([f_int, f_forma, f_tex])


def nombres_features(n_int, n_forma, n_tex):
    """
    Genera los nombres de columna para el DataFrame de features.

    Parameters
    ----------
    n_int   : int — número de features de intensidad
    n_forma : int — número de features de forma
    n_tex   : int — número de features de textura

    Returns
    -------
    nombres : list[str]
    """
    cols  = [f'int_{i:02d}' for i in range(n_int)]
    cols += [f'forma_{i:02d}' for i in range(n_forma)]
    cols += [f'tex_{i:02d}' for i in range(n_tex)]
    return cols
