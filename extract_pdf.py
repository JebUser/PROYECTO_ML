"""
Extrae el texto del PDF del proyecto y lo guarda en un .txt legible.
Ejecutar con el venv activo:
    python extract_pdf.py
"""
from pathlib import Path

PDF_PATH = Path(__file__).parent / "ProyectoFinal_JAVE_11052026.pdf"
OUT_PATH = Path(__file__).parent / "proyecto_preguntas.txt"

def extraer_con_pypdf():
    from pypdf import PdfReader
    reader = PdfReader(PDF_PATH)
    texto = ""
    for i, page in enumerate(reader.pages):
        texto += f"\n{'='*60}\nPÁGINA {i+1}\n{'='*60}\n"
        texto += page.extract_text() or ""
    return texto

def extraer_con_pdfplumber():
    import pdfplumber
    texto = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            texto += f"\n{'='*60}\nPÁGINA {i+1}\n{'='*60}\n"
            texto += page.extract_text() or ""
    return texto

if __name__ == "__main__":
    texto = None
    for fn, nombre in [(extraer_con_pypdf, "pypdf"), (extraer_con_pdfplumber, "pdfplumber")]:
        try:
            texto = fn()
            print(f"✅ Extraído con {nombre}")
            break
        except ImportError:
            print(f"⚠️  {nombre} no instalado, intentando siguiente...")
        except Exception as e:
            print(f"❌ Error con {nombre}: {e}")

    if texto:
        OUT_PATH.write_text(texto, encoding="utf-8")
        print(f"✅ Guardado en: {OUT_PATH}")
        print("\n--- CONTENIDO ---\n")
        print(texto)
    else:
        print("\n❌ Instala pypdf: pip install pypdf")
