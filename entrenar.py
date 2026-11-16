# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO DE LA RED NEURONAL DE CUERVIFRIEND
------------------------------------------------
Uso:   py entrenar.py

Qué hace:
 1. Lee las frases de ejemplo de intents.json.
 2. Construye el vocabulario (las "señales" que la red podrá ver).
 3. Genera datos extra con faltas de ortografía y palabras de relleno (aumento de datos).
 4. Entrena la red neuronal (red_neuronal.py).
 5. La evalúa con frases NUEVAS (pruebas.json y pruebas_finales.json) que nunca vio al entrenar.
 6. Guarda la red entrenada en modelo.json y las evidencias en la carpeta evidencias/.
"""
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np

from procesamiento import caracteristicas, vectorizar
from red_neuronal import RedNeuronal

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(BASE, "evidencias")

# ---- Hiperparámetros (se pueden cambiar y volver a entrenar) ----
SEMILLA = 7                 # para que el resultado sea repetible
CAPAS_OCULTAS = [64, 32]    # neuronas de las dos capas ocultas
EPOCAS = 80                 # cuántas veces recorre todos los ejemplos
LOTE = 64                   # ejemplos por paso de aprendizaje
TASA = 0.004                # velocidad de aprendizaje (learning rate)
DROPOUT = 0.25              # % de neuronas que se apagan al azar al entrenar
VARIANTES = 12              # versiones de cada frase (original + 11 con cambios)

RELLENO_ANTES = ["oye", "disculpa", "una pregunta", "por favor", "porfa", "wey", "bro", "mi loco", "compa",
                 "oiga", "perdón", "mira", "ey", "cuervifriend", "hola", "buenas", "qué onda", "ocupo saber"]
RELLENO_DESPUES = ["por favor", "porfa", "gracias", "urgente", "mi loco", "wey", "amigo", "compa", "bro"]

lineas_log = []


def log(texto=""):
    print(texto)
    lineas_log.append(texto)


# ----------------------------------------------------------------- aumento de datos
def con_error(palabra, rng):
    """Simula una falta de ortografía: quita, repite o intercambia una letra."""
    if len(palabra) < 4:
        return palabra
    i = int(rng.integers(1, len(palabra) - 1))
    op = int(rng.integers(0, 3))
    if op == 0:
        return palabra[:i] + palabra[i + 1:]
    if op == 1:
        return palabra[:i] + palabra[i] + palabra[i:]
    return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]


def variante(frase, tipo, rng):
    palabras = frase.split()
    if len(palabras) >= 5 and rng.random() < 0.25:            # a veces se olvida una palabra
        palabras.pop(int(rng.integers(len(palabras))))
    for _ in range(int(rng.integers(0, 3))):                  # 0 a 2 faltas de ortografía
        j = int(rng.integers(len(palabras)))
        palabras[j] = con_error(palabras[j], rng)
    texto = " ".join(palabras)
    if tipo != "charla":                                      # relleno solo en trámites y seguimientos
        if rng.random() < 0.25:
            texto = str(rng.choice(RELLENO_ANTES)) + " " + texto
        if rng.random() < 0.20:
            texto = texto + " " + str(rng.choice(RELLENO_DESPUES))
    return texto


# ----------------------------------------------------------------- evidencias
def guardar_curva(historial, ruta):
    """Dibuja la curva de aprendizaje (pérdida y precisión) como imagen SVG, sin librerías extra."""
    W, H, m = 640, 340, 50
    n = len(historial)
    max_perdida = max(h[1] for h in historial)

    def x(i): return m + (W - 2 * m) * i / max(1, n - 1)
    def y_p(v): return H - m - (H - 2 * m) * v / max_perdida
    def y_a(v): return H - m - (H - 2 * m) * v

    perdida = " ".join(f"{x(i):.1f},{y_p(h[1]):.1f}" for i, h in enumerate(historial))
    precision = " ".join(f"{x(i):.1f},{y_a(h[2]):.1f}" for i, h in enumerate(historial))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">Entrenamiento de la red neuronal de CuerviFriend</text>
<line x1="{m}" y1="{H-m}" x2="{W-m}" y2="{H-m}" stroke="#333"/><line x1="{m}" y1="{m}" x2="{m}" y2="{H-m}" stroke="#333"/>
<polyline points="{perdida}" fill="none" stroke="#a31d3f" stroke-width="2.5"/>
<polyline points="{precision}" fill="none" stroke="#1f8a4c" stroke-width="2.5"/>
<text x="{W/2}" y="{H-12}" text-anchor="middle">Épocas (1 a {n})</text>
<text x="{m+10}" y="{m+4}" fill="#a31d3f">■ Pérdida (menor = mejor)</text>
<text x="{m+10}" y="{m+22}" fill="#1f8a4c">■ Precisión (mayor = mejor), tope = 100 %</text>
</svg>"""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEMILLA)
    os.makedirs(EVID, exist_ok=True)

    with open(os.path.join(BASE, "intents.json"), encoding="utf-8") as f:
        intenciones = json.load(f)["intenciones"]
    with open(os.path.join(BASE, "pruebas.json"), encoding="utf-8") as f:
        pruebas = json.load(f)["pruebas"]
    with open(os.path.join(BASE, "pruebas_finales.json"), encoding="utf-8") as f:
        pruebas_finales = json.load(f)["pruebas"]

    clases = [i["tag"] for i in intenciones]
    ejemplos = [(p, i["tag"], i["tipo"]) for i in intenciones for p in i["patrones"]]

    log("=" * 70)
    log("ENTRENAMIENTO DE LA RED NEURONAL · CuerviFriend · Tecnológico de Nuevo León")
    log("Fecha: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 70)
    log(f"Intenciones (clases de salida): {len(clases)}")
    log(f"Frases de ejemplo originales : {len(ejemplos)}")

    # 1) Vocabulario: todas las palabras + pares y fragmentos que aparecen en 2 o más frases
    conteo = Counter()
    for frase, _, _ in ejemplos:
        for s in set(caracteristicas(frase)):
            conteo[s] += 1
    vocabulario = sorted(s for s, c in conteo.items() if s.startswith("w:") or c >= 2)
    indice = {s: j for j, s in enumerate(vocabulario)}
    log(f"Vocabulario (neuronas de entrada): {len(vocabulario)}")

    # 2) Datos de entrenamiento con aumento de datos
    textos, etiquetas = [], []
    for frase, tag, tipo in ejemplos:
        textos.append(frase)
        etiquetas.append(clases.index(tag))
        for _ in range(VARIANTES - 1):
            textos.append(variante(frase, tipo, rng))
            etiquetas.append(clases.index(tag))
    X = np.array([vectorizar(t, indice) for t in textos], dtype=np.float32)
    Y = np.zeros((len(textos), len(clases)), dtype=np.float32)
    Y[np.arange(len(textos)), etiquetas] = 1.0
    log(f"Ejemplos tras aumento de datos : {len(textos)} (cada frase con faltas de ortografía y relleno)")

    # 3) Red neuronal
    arquitectura = [len(vocabulario)] + CAPAS_OCULTAS + [len(clases)]
    red = RedNeuronal(arquitectura, semilla=SEMILLA)
    parametros = sum(w.size for w in red.W) + sum(b.size for b in red.b)
    log(f"Arquitectura: {' -> '.join(map(str, arquitectura))}  (ReLU, ReLU, Softmax)")
    log(f"Parámetros entrenables (pesos y sesgos): {parametros:,}")
    log(f"Hiperparámetros: épocas={EPOCAS}, lote={LOTE}, tasa={TASA}, dropout={DROPOUT}, semilla={SEMILLA}")
    log("-" * 70)
    log("Entrenando...")
    historial = red.entrenar(X, Y, epocas=EPOCAS, lote=LOTE, lr=TASA, dropout=DROPOUT, log=log)

    # 4) Evaluación
    log("-" * 70)
    X_orig = np.array([vectorizar(f, indice) for f, _, _ in ejemplos], dtype=np.float32)
    y_orig = np.array([clases.index(t) for _, t, _ in ejemplos])
    acc_entrenamiento = float(np.mean(red.predecir(X_orig).argmax(1) == y_orig))
    log(f"Precisión en frases de entrenamiento: {acc_entrenamiento * 100:.1f} %")

    def evaluar(titulo, lista):
        """Prueba la red con frases que nunca vio y devuelve (precisión, líneas de detalle)."""
        Xp = np.array([vectorizar(p["frase"], indice) for p in lista], dtype=np.float32)
        P = red.predecir(Xp)
        aciertos, top3, detalle = 0, 0, []
        for p, prob in zip(lista, P):
            orden = np.argsort(-prob)
            pred = clases[orden[0]]
            ok = pred == p["esperado"]
            aciertos += ok
            top3 += p["esperado"] in [clases[k] for k in orden[:3]]
            detalle.append(f"{'OK ' if ok else 'MAL'} | «{p['frase']}» -> {pred} ({prob[orden[0]] * 100:.1f} %)"
                           + ("" if ok else f"   [esperado: {p['esperado']}]"))
        log(f"{titulo}: {aciertos / len(lista) * 100:.1f} %  ({aciertos}/{len(lista)})  |  "
            f"intención correcta entre las 3 primeras: {top3 / len(lista) * 100:.1f} %")
        return aciertos / len(lista), detalle

    acc_ajuste, det_ajuste = evaluar("Precisión en pruebas de ajuste (pruebas.json)", pruebas)
    acc_prueba, det_finales = evaluar("Precisión en pruebas finales (pruebas_finales.json)", pruebas_finales)
    log("-" * 70)
    log("Detalle de las pruebas finales (frases nuevas):")
    for l in det_finales:
        log("  " + l)
    log("Detalle de las pruebas de ajuste:")
    for l in det_ajuste:
        log("  " + l)

    # 5) Guardar modelo y evidencias
    meta = dict(
        version=1, creado=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), vocabulario=vocabulario, clases=clases,
        activaciones=["relu"] * len(CAPAS_OCULTAS) + ["softmax"],
        hiperparametros=dict(epocas=EPOCAS, lote=LOTE, tasa=TASA, dropout=DROPOUT, semilla=SEMILLA),
        metricas=dict(frases_entrenamiento=len(ejemplos), ejemplos_aumentados=len(textos), parametros=parametros,
                      precision_entrenamiento=round(acc_entrenamiento, 4), precision_ajuste=round(acc_ajuste, 4),
                      precision_pruebas=round(acc_prueba, 4), n_pruebas=len(pruebas_finales)),
    )
    ruta_modelo = os.path.join(BASE, "modelo.json")
    red.guardar(ruta_modelo, meta)

    # Comprobar que el JSON guardado funciona igual que la red en memoria
    red2, _ = RedNeuronal.cargar(ruta_modelo)
    Xc = np.array([vectorizar(p["frase"], indice) for p in pruebas_finales], dtype=np.float32)
    yc = np.array([clases.index(p["esperado"]) for p in pruebas_finales])
    acc2 = float(np.mean(red2.predecir(Xc).argmax(1) == yc))
    log("-" * 70)
    log(f"modelo.json guardado ({os.path.getsize(ruta_modelo) / 1024:.0f} KB). "
        f"Al volver a cargarlo desde el JSON, precisión en pruebas finales = {acc2 * 100:.1f} %")
    log(f"Tiempo total: {time.time() - t0:.1f} s")

    guardar_curva(historial, os.path.join(EVID, "curva_entrenamiento.svg"))
    with open(os.path.join(EVID, "entrenamiento.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_log) + "\n")
    log("Evidencias guardadas en la carpeta evidencias/ (entrenamiento.txt y curva_entrenamiento.svg)")


if __name__ == "__main__":
    main()
