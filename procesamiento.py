# -*- coding: utf-8 -*-
"""
PROCESAMIENTO DE TEXTO
----------------------
La red neuronal solo entiende números, así que aquí convertimos una frase
en un vector de ceros y unos (1 = "esta señal aparece en la frase").

Las "señales" que usamos son tres:
  - palabras sueltas           ->  w:beca
  - pares de palabras          ->  b:buenas tardes
  - fragmentos de 3 letras     ->  c:bec   (ayudan cuando hay faltas de ortografía)
"""
import re
import unicodedata

import numpy as np


def normalizar(texto):
    """Minúsculas, sin acentos, sin signos y sin letras repetidas ('ondaaa' -> 'onda')."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")   # quita acentos y la tilde de la ñ
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)     # 3 o más letras iguales seguidas -> 1 letra
    t = re.sub(r"([aiu])\1", r"\1", t)         # 'aa', 'ii', 'uu' no existen en español -> 1 letra
    t = re.sub(r"[^a-z0-9\s]", " ", t)         # quita signos de puntuación y emojis
    return re.sub(r"\s+", " ", t).strip()


def caracteristicas(texto):
    """Lista de señales (palabras, pares de palabras y trigramas de letras) de una frase."""
    palabras = normalizar(texto).split()
    señales = ["w:" + p for p in palabras]
    señales += ["b:" + a + " " + b for a, b in zip(palabras, palabras[1:])]
    for p in palabras:
        marcada = "<" + p + ">"                                    # < y > marcan inicio y fin de palabra
        señales += ["c:" + marcada[i:i + 3] for i in range(len(marcada) - 2)]
    return señales


def vectorizar(texto, indice):
    """Convierte la frase en un vector de 0/1. 'indice' es el vocabulario {señal: posición}."""
    v = np.zeros(len(indice), dtype=np.float32)
    for s in set(caracteristicas(texto)):
        j = indice.get(s)
        if j is not None:          # las señales que la red no conoce se ignoran
            v[j] = 1.0
    return v
