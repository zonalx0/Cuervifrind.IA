# -*- coding: utf-8 -*-
"""
RED NEURONAL (hecha desde cero con NumPy)
-----------------------------------------
Tipo: perceptrón multicapa (red "feed-forward" totalmente conectada).

    entrada  ->  capa oculta 1 (ReLU)  ->  capa oculta 2 (ReLU)  ->  salida (Softmax)
    (señales)       64 neuronas               32 neuronas             1 neurona por intención

- ReLU:     f(z) = max(0, z)           activa solo si la señal es positiva.
- Softmax:  convierte la salida en probabilidades que suman 100 %.
- Pérdida:  entropía cruzada (qué tan lejos está la predicción de la respuesta correcta).
- Aprendizaje: retropropagación (backpropagation) + optimizador Adam.
- Dropout:  apaga neuronas al azar al entrenar para que no memorice (evita sobreajuste).
"""
import json

import numpy as np


class RedNeuronal:
    def __init__(self, capas, semilla=42):
        """capas = [entradas, ocultas1, ocultas2, ..., salidas]"""
        self.capas = list(capas)
        self.rng = np.random.default_rng(semilla)
        # Pesos iniciales aleatorios (inicialización He, buena para ReLU) y sesgos en cero
        self.W = [self.rng.normal(0, np.sqrt(2.0 / capas[i]), (capas[i], capas[i + 1]))
                  for i in range(len(capas) - 1)]
        self.b = [np.zeros(capas[i + 1]) for i in range(len(capas) - 1)]
        # Memoria del optimizador Adam
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ------------------------------------------------------------------ uso
    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)       # truco numérico para evitar desbordes
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predecir(self, X):
        """Propagación hacia adelante. X: (n, entradas) -> probabilidades (n, intenciones)."""
        A = X
        for i in range(len(self.W) - 1):
            A = np.maximum(0, A @ self.W[i] + self.b[i])      # capas ocultas con ReLU
        return self._softmax(A @ self.W[-1] + self.b[-1])      # capa de salida

    # ------------------------------------------------------------ entrenamiento
    def _paso(self, X, Y, lr, dropout, l2):
        """Un paso de aprendizaje con un lote de ejemplos. Devuelve la pérdida del lote."""
        n = X.shape[0]
        # 1) Hacia adelante, guardando lo que necesitamos para retropropagar
        entradas, zs, mascaras = [X], [], []
        A = X
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = np.maximum(0, Z)
            m = (self.rng.random(A.shape) >= dropout) / (1.0 - dropout)   # dropout
            A = A * m
            zs.append(Z); mascaras.append(m); entradas.append(A)
        P = self._softmax(A @ self.W[-1] + self.b[-1])
        perdida = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))

        # 2) Hacia atrás: calcular el gradiente (cuánto aporta cada peso al error)
        dZ = (P - Y) / n                                   # error en la capa de salida
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = entradas[i].T @ dZ + l2 * self.W[i]    # l2 = regularización
            gb[i] = dZ.sum(axis=0)
            if i > 0:
                dA = dZ @ self.W[i].T                      # el error viaja a la capa anterior
                dZ = dA * mascaras[i - 1] * (zs[i - 1] > 0)  # derivada de dropout y de ReLU

        # 3) Actualizar pesos con Adam
        self._adam(gW, gb, lr)
        return perdida

    def _adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                m_hat = m[i] / (1 - b1 ** self.t)
                v_hat = v[i] / (1 - b2 ** self.t)
                p[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def entrenar(self, X, Y, epocas=80, lote=64, lr=0.004, dropout=0.25, l2=1e-4, cada=10, log=print):
        """Entrena la red. Devuelve el historial [(época, pérdida, precisión), ...]."""
        historial = []
        n = X.shape[0]
        for ep in range(1, epocas + 1):
            orden = self.rng.permutation(n)                # mezclar los ejemplos cada época
            perdidas = [self._paso(X[orden[i:i + lote]], Y[orden[i:i + lote]], lr, dropout, l2)
                        for i in range(0, n, lote)]
            precision = float(np.mean(self.predecir(X).argmax(1) == Y.argmax(1)))
            historial.append((ep, float(np.mean(perdidas)), precision))
            if ep == 1 or ep % cada == 0:
                log(f"  época {ep:3d}/{epocas} | pérdida = {np.mean(perdidas):.4f} | precisión = {precision * 100:5.1f} %")
        return historial

    # ------------------------------------------------------------ guardar / cargar
    def guardar(self, ruta, extra):
        """Guarda la red completa (pesos + vocabulario + clases) en un archivo JSON."""
        datos = dict(extra)
        datos["arquitectura"] = self.capas
        datos["pesos"] = [np.round(w, 5).tolist() for w in self.W]
        datos["sesgos"] = [np.round(b, 5).tolist() for b in self.b]
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        """Carga una red guardada con guardar(). Devuelve (red, datos_del_json)."""
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        red = cls(datos["arquitectura"])
        red.W = [np.array(w) for w in datos["pesos"]]
        red.b = [np.array(b) for b in datos["sesgos"]]
        return red, datos
