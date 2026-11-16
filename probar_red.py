# -*- coding: utf-8 -*-
"""
Prueba la red neuronal desde la terminal (sin la página web).
Uso:  py probar_red.py
Escribe una frase y verás qué intención detecta la red y con qué probabilidad.
"""
import sys

from cerebro import Cerebro

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

cerebro = Cerebro()
info = cerebro.info()
print(f"Red neuronal cargada desde modelo.json: {info['capas']}  ({info['parametros']:,} parámetros)")
print("Escribe una frase (o 'salir').\n")

contexto = None
while True:
    frase = input("Tú: ").strip()
    if frase.lower() in ("salir", "exit", "q"):
        break
    print("  Análisis de la red neuronal (top 5):")
    for tag, prob in cerebro.analizar(frase, k=5):
        print(f"    {tag:<28} {prob * 100:5.1f} %  " + "█" * int(prob * 30))
    salida = cerebro.responder(frase, contexto)
    contexto = salida["contexto"]
    print("CuerviFriend:", salida["respuesta"], "\n")
