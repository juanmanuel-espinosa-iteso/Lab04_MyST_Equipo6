"""Convierte un documento de `docs/` de Markdown a PDF con Chrome sin interfaz (como en el Lab03).

Uso, desde la raíz del repo:  python docs/generar_pdf.py reporte   →  docs/reporte.pdf
                              python docs/generar_pdf.py walkthrough → docs/walkthrough.pdf

No hay pandoc ni LaTeX en la Mac: el Markdown se convierte a HTML en el navegador con `marked` (versión fija, desde
jsDelivr) y Chrome lo imprime a PDF. En `reporte.md`, el comentario `<!-- generar_pdf.py inserta aquí docs/powell.md -->`
se reemplaza por la formulación de Powell como apéndice, para no duplicar ese texto. Las imágenes se resuelven
relativas a `docs/` (p. ej. `../artifacts/equity_curves_test_fold_1.png`).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MARKED = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
POWELL_MARK = "<!-- generar_pdf.py inserta aquí docs/powell.md -->"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.42;
       color: #1d1d1f; background: #fff; }
h1 { font-size: 17pt; margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; padding-bottom: 2pt; border-bottom: 1px solid #d2d2d7;
     break-after: avoid; }
h3 { font-size: 10.8pt; margin: 12pt 0 4pt; break-after: avoid; }
h4 { font-size: 10pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 16pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.1pt; break-inside: avoid; }
th, td { border: 1px solid #d2d2d7; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #f2f2f5; font-weight: 600; }
code { font-family: Menlo, monospace; font-size: 8.2pt; background: #f2f2f5; padding: 0 2pt; border-radius: 2pt; }
blockquote { margin: 6pt 0; padding: 4pt 8pt; border-left: 3px solid #c7522a; background: #fbf3ef; }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; break-inside: avoid; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4pt; break-inside: avoid; }
.grid img { margin: 0; }
em { color: #424245; }
"""


def appendix_from_powell():
    """`docs/powell.md` como apéndice: sin su título y con las secciones numeradas A.1, A.2, …"""
    text = (DOCS / "powell.md").read_text(encoding="utf-8").split("\n", 1)[1]
    text = re.sub(r"^### ", "#### ", text, flags=re.M)
    return re.sub(r"^## (\d+)\. ", r"### A.\1 ", text, flags=re.M)


def build(name):
    """Escribe `docs/{name}.pdf` a partir de `docs/{name}.md`."""
    markdown = (DOCS / f"{name}.md").read_text(encoding="utf-8")
    if POWELL_MARK in markdown:
        markdown = markdown.replace(POWELL_MARK, appendix_from_powell())
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<base href="{DOCS.as_uri()}/"><title>{name}</title><style>{CSS}</style>
<script src="{MARKED}"></script></head><body><main id="doc"></main>
<script>document.getElementById("doc").innerHTML = marked.parse({json.dumps(markdown)});</script>
</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / f"{name}.html"
        page.write_text(html, encoding="utf-8")
        out = DOCS / f"{name}.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", "--virtual-time-budget=20000",
                        f"--print-to-pdf={out}", page.as_uri()], check=True, capture_output=True)
    print(f"{out.relative_to(DOCS.parent)} escrito")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "reporte")
