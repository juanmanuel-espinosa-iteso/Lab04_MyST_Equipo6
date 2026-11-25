"""Configuración de pytest: hace que `import code` resuelva al paquete del proyecto.

La carpeta obligatoria `code/` (§2.2) se llama igual que el módulo `code` de la biblioteca estándar, que pytest ya
importó (vía pdb) antes de recolectar las pruebas. Se quita de `sys.modules` y se pone la raíz del repo al frente de
`sys.path` para que `from code import data` cargue el paquete local. Ver `docs/decisiones.md` (2026-11-23).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not str(getattr(sys.modules.get("code"), "__file__", "")).startswith(str(ROOT)):
    sys.modules.pop("code", None)
