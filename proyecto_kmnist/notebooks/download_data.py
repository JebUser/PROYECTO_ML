"""
Script para descargar el dataset Kuzushiji-MNIST.
Ejecutar desde la raíz del proyecto con el venv activo:

    python notebooks/download_data.py
"""

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
    print("\n📥 Descargando Kuzushiji-MNIST...\n")
    for filename, url in URLS.items():
        download_file(url, DATA_DIR / filename)
    print("\n🎉 Dataset listo en:", DATA_DIR.resolve())
