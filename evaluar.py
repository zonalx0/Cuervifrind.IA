# -*- coding: utf-8 -*-
"""
Evalúa el modelo.json ya entrenado con un archivo de pruebas (frases nuevas + intención esperada).
Uso:   py evaluar.py                       (usa pruebas_ciegas.json)
       py evaluar.py pruebas_finales.json
Sirve como evidencia de que la red entiende frases que nunca vio.
"""
import json
import os
import sys

from cerebro import Cerebro

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

archivo = sys.argv[1] if len(sys.argv) > 1 else "pruebas_ciegas.json"
ruta = archivo if os.path.isabs(archivo) else os.path.join(os.path.dirname(os.path.abspath(__file__)), archivo)
with open(ruta, encoding="utf-8") as f:
    pruebas = json.load(f)["pruebas"]

cerebro = Cerebro()
aciertos = 0
for p in pruebas:
    top = cerebro.analizar(p["frase"], k=1)
    pred, prob = top[0] if top else ("(sin señales)", 0.0)
    ok = pred == p["esperado"]
    aciertos += ok
    print(f"{'OK ' if ok else 'MAL'} | «{p['frase']}» -> {pred} ({prob * 100:.1f} %)" + ("" if ok else f"   [esperado: {p['esperado']}]"))
print(f"\n{archivo}: {aciertos}/{len(pruebas)} = {aciertos / len(pruebas) * 100:.1f} %")
