# -*- coding: utf-8 -*-
"""
CEREBRO DE CUERVIFRIEND
-----------------------
Une las tres piezas del proyecto:
  1. modelo.json      -> la red neuronal ya entrenada: detecta la INTENCIÓN del mensaje.
  2. intents.json     -> respuestas de conversación y datos de cada trámite.
  3. directorio.json  -> oficinas, responsables, correos y teléfonos del TecNL.

Flujo:  mensaje -> vectorizar -> RED NEURONAL -> intención + probabilidad -> lógica de respuesta
"""
import json
import os
import random
from datetime import date

import numpy as np

from procesamiento import normalizar, vectorizar
from red_neuronal import RedNeuronal

BASE = os.path.dirname(os.path.abspath(__file__))
UMBRAL_SEGURIDAD = 0.55   # si la red tiene menos de 55 % de seguridad, no adivina: pide aclaración
UMBRAL_OPCION = 0.12      # trámites con al menos 12 % se ofrecen como opciones cuando hay duda
ANIO_FUNDACION = 1976
# Filtro de palabras: red de seguridad para vulgaridades muy claras (la red neuronal es la que decide primero;
# este filtro solo corrige cuando ella no reconoce la ofensa). Palabras exactas, sin acentos.
PALABRAS_VULGARES = {"pene", "penes", "verga", "vergas", "pito", "pitos", "vagina", "vaginas", "culo", "culos", "nalgas",
                     "tetas", "chichis", "puta", "putas", "puto", "putos", "putazo", "pendejo", "pendejos", "pendeja",
                     "pendejada", "chingada", "chingado", "chingar", "chinga", "cabron", "cabrones", "mierda", "mamada",
                     "mamadas", "culero", "culera", "ojete", "zorra", "joto", "maricon", "imbecil", "idiota", "estupido"}

# Seguro para mensajes de crisis emocional: si la red no lo reconoce, esta lista lo atrapa (se revisa texto sin acentos).
FRASES_CRISIS = ("suicid", "quitarme la vida", "quiero morir", "quiero morirme", "me quiero morir", "no quiero vivir", "ya no quiero vivir",
                 "matarme", "me quiero matar", "acabar con mi vida", "hacerme dano", "lastimarme", "dejar de existir")
NO_ES_CRISIS = ("de risa", "de la risa", "de hambre", "de sueno", "de pena", "de verguenza", "de calor", "de frio", "de flojera", "de aburrimiento")

# Palabras que indican que la pregunta es sobre el Tec en general y no sobre el trámite del que se venía hablando
PALABRAS_TEC = {"tec", "tecnologico", "escuela", "instituto", "itnl", "tecnl", "tecnm", "universidad", "campus",
                "pagina", "web", "sitio", "url", "link", "facebook", "instagram", "redes"}


def _leer(nombre):
    with open(os.path.join(BASE, nombre), encoding="utf-8") as f:
        return json.load(f)


class Cerebro:
    def __init__(self):
        # La red neuronal se carga desde un archivo JSON (pesos, vocabulario y clases)
        self.red, self.modelo = RedNeuronal.cargar(os.path.join(BASE, "modelo.json"))
        self.indice = {s: j for j, s in enumerate(self.modelo["vocabulario"])}
        self.clases = self.modelo["clases"]

        datos = _leer("intents.json")
        self.intenciones = {i["tag"]: i for i in datos["intenciones"]}
        self.plantillas = datos["plantillas_tramite"]
        self.textos = datos["textos"]
        directorio = _leer("directorio.json")
        self.oficinas = directorio["oficinas"]
        self.subdirecciones = directorio["subdirecciones"]

        if set(self.clases) != set(self.intenciones):
            raise RuntimeError("intents.json cambió y modelo.json quedó desactualizado. "
                               "Ejecuta:  py entrenar.py")

    # ------------------------------------------------------------ red neuronal
    def analizar(self, texto, k=3):
        """Pasa el mensaje por la red neuronal. Devuelve [(intención, probabilidad), ...] de mayor a menor."""
        x = vectorizar(texto, self.indice)
        if not x.any():                       # ninguna señal conocida: la red no tiene nada que analizar
            return []
        p = self.red.predecir(x[None, :])[0]
        mejores = np.argsort(-p)[:k]
        return [(self.clases[i], float(p[i])) for i in mejores]

    def info(self):
        m, arq = self.modelo["metricas"], self.modelo["arquitectura"]
        return dict(intenciones=len(self.clases), entradas=arq[0], capas=" → ".join(map(str, arq)),
                    parametros=m["parametros"], precision=round(m["precision_pruebas"] * 100, 1))

    # ------------------------------------------------------------ lógica de respuesta
    def responder(self, texto, contexto=None, ofensas=0):
        """Entiende el mensaje con la red neuronal y arma la respuesta.
        'contexto' = último trámite del que se habló. 'ofensas' = cuántas veces ya se portó mal en esta conversación."""
        if not texto.strip():
            return self._salida("Escríbeme tu duda o usa el micrófono y con gusto te ayudo. 🪶", None, 0, [], contexto)
        top = self.analizar(texto)
        analisis = [dict(tag=t, prob=round(p * 100, 1)) for t, p in top]
        norm = normalizar(texto)
        palabras = norm.split()
        if (any(f in norm for f in FRASES_CRISIS) and not any(x in norm for x in NO_ES_CRISIS)
                and not (top and top[0][0] == "crisis")):
            # Un mensaje así nunca debe terminar en "no entendí" ni en una respuesta alegre
            analisis.insert(0, dict(tag="crisis", prob=100.0, nota="seguro para mensajes de crisis"))
            return self._por_intencion("crisis", norm, 1.0, analisis, contexto, ofensas)
        if PALABRAS_VULGARES & set(palabras) and not (top and top[0][0] == "groseria"):
            # La red neuronal decide primero; si no reconoció una vulgaridad clara, el filtro de palabras la corrige
            analisis.insert(0, dict(tag="groseria", prob=100.0, nota="filtro de palabras"))
            return self._por_intencion("groseria", norm, 1.0, analisis, contexto, ofensas)
        if not top or top[0][1] < UMBRAL_SEGURIDAD:
            return self._no_entendi(top, analisis, contexto)
        # Mensaje cortísimo cuyas palabras la red nunca vio: no se fía de los fragmentos de letras (evita respuestas absurdas)
        if len(palabras) <= 2 and not any("w:" + p in self.indice for p in palabras):
            return self._no_entendi(top, analisis, contexto, desconocida=True)
        tag, prob = top[0]
        if tag == "crisis" and any(x in norm for x in NO_ES_CRISIS):
            tag = "risa"                          # "me quiero morir de la risa" es una expresión, no una crisis
        # "Dónde está ubicada" justo después de hablar de un trámite se refiere a ese trámite, no al Tec completo
        if contexto and tag in ("ubicacion_tec", "contacto_tec") and not PALABRAS_TEC & set(palabras):
            if tag == "ubicacion_tec":
                tag = "ctx_ubicacion"
            elif "correo" in palabras:
                tag = "ctx_correo"
            elif {"telefono", "numero", "llamo"} & set(palabras):
                tag = "ctx_telefono"
        return self._por_intencion(tag, norm, prob, analisis, contexto, ofensas)

    def responder_tag(self, tag, contexto=None):
        """Cuando la persona elige uno de los botones de opciones, ya sabemos qué trámite quiere."""
        if tag not in self.intenciones:
            return self.responder("", contexto)
        analisis = [dict(tag=tag, prob=100.0, nota="elegido por el usuario")]
        return self._por_intencion(tag, "", 1.0, analisis, contexto)

    def _por_intencion(self, tag, norm, prob, analisis, contexto, ofensas=0):
        inten = self.intenciones[tag]
        tipo = inten["tipo"]
        if tipo == "tramite":
            texto = self._ficha(inten)
            saludo = self._saludo_inicial(norm)          # "hola necesito una beca" -> responde el saludo también
            if saludo:
                texto = saludo + " " + texto
            return self._salida(texto, tag, prob, analisis, tag)
        if tipo == "contexto":
            if not contexto:
                return self._salida(self.textos["sin_contexto"], tag, prob, analisis, None)
            return self._salida(self._dato_de_contexto(inten["campo"], contexto), tag, prob, analisis, contexto)
        # charla: respuesta variada (si la intención tiene "escalada", se pone más seria con cada ofensa repetida)
        if inten.get("escalada"):
            nivel = inten["escalada"][min(ofensas, len(inten["escalada"]) - 1)]
            texto = random.choice(nivel)
        else:
            texto = random.choice(inten["respuestas"])
        texto = texto.replace("{saludo}", self._saludo(norm)).replace("{anios}", str(self._anios()))
        salida = self._salida(texto, tag, prob, analisis, contexto)
        salida["ofensa"] = bool(inten.get("escalada"))
        return salida

    def _no_entendi(self, top, analisis, contexto, desconocida=False):
        opciones = [t for t, p in top if p >= UMBRAL_OPCION and self.intenciones[t]["tipo"] == "tramite"][:3]
        if opciones:
            botones = [dict(texto=self.intenciones[t]["tramite"], tag=t) for t in opciones]
            return self._salida(self.textos["opciones"], None, top[0][1], analisis, contexto, botones)
        frases = self.textos["desconocida"] if desconocida else self.textos["no_entendi"]
        return self._salida(random.choice(frases), None, top[0][1] if top else 0, analisis, contexto)

    @staticmethod
    def _salida(texto, tag, prob, analisis, contexto, opciones=None):
        return dict(respuesta=texto, tag=tag, confianza=round(prob * 100, 1), analisis=analisis,
                    contexto=contexto, opciones=opciones or [])

    # ------------------------------------------------------------ piezas de respuesta
    def _datos_oficina(self, inten):
        of = self.oficinas[inten["oficina"]]
        return of, self.subdirecciones[of["subdireccion"]]

    def _ficha(self, inten):
        """Tarjeta del trámite con los datos del directorio (solo muestra lo que el documento sí trae)."""
        of, sub = self._datos_oficina(inten)
        if inten["tramite"].lower() == of["area"].lower():     # evita "Para Becas, el área es Becas"
            intro = random.choice(self.textos["intro_misma_area"]).format(tramite=inten["tramite"], area=of["area"])
        else:
            intro = random.choice(self.plantillas).format(tramite=inten["tramite"], area=of["area"])
        lineas = [intro, ""]
        if of.get("ubicacion"):
            lineas.append("📍 Ubicación: " + of["ubicacion"])
        if of.get("responsable"):
            lineas.append("👤 Responsable: " + of["responsable"])
        lineas.append("✉️ Correo: " + of["correo"])
        if of.get("telefono"):
            lineas.append("📞 Teléfono: " + of["telefono"])
        lineas.append(f"🏛️ {sub['nombre']}: {sub['responsable']} · {sub['ubicacion']} · {sub['correo']} · {sub['telefono']}")
        if inten.get("nota"):
            lineas += ["", inten["nota"]]
        lineas += ["", random.choice(self.textos["cierres"])]
        return "\n".join(lineas)

    def _dato_de_contexto(self, campo, contexto_tag):
        """Responde preguntas de seguimiento (¿dónde queda?, ¿su correo?) sobre el último trámite."""
        inten = self.intenciones[contexto_tag]
        of, sub = self._datos_oficina(inten)
        d = dict(area=of["area"], correo=of["correo"], telefono=of.get("telefono") or "",
                 subdireccion=sub["nombre"], sub_ubicacion=sub["ubicacion"],
                 sub_telefono=sub["telefono"], sub_responsable=sub["responsable"])
        if campo == "ubicacion":
            return f"📍 {of['area']} está en {of['ubicacion']}." if of.get("ubicacion") else self.textos["sin_ubicacion"].format(**d)
        if campo == "correo":
            return f"✉️ El correo de {of['area']} es: {of['correo']}"
        if campo == "telefono":
            return f"📞 El teléfono de {of['area']} es {of['telefono']}." if of.get("telefono") else self.textos["sin_telefono"].format(**d)
        if campo == "responsable":
            return f"👤 {of['area']}: {of['responsable']}." if of.get("responsable") else self.textos["sin_responsable"].format(**d)
        via = self.textos["via_telefono" if of.get("telefono") else "via_correo"].format(**d)   # horario
        return self.textos["sin_horario"].format(area=of["area"], via=via)

    @staticmethod
    def _saludo(norm):
        palabras = norm.split()
        if "dias" in palabras or "dia" in palabras:
            return "¡Buenos días!"
        if "tardes" in palabras or "tarde" in palabras:
            return "¡Buenas tardes!"
        if "noches" in palabras or "noche" in palabras:
            return "¡Buenas noches!"
        return "¡Hola!"

    @staticmethod
    def _saludo_inicial(norm):
        """Si el mensaje EMPIEZA con un saludo (y pide un trámite), devolvemos el saludo correspondiente."""
        if norm.startswith(("buenos dias", "buen dia")):
            return "¡Buenos días!"
        if norm.startswith("buenas tardes"):
            return "¡Buenas tardes!"
        if norm.startswith("buenas noches"):
            return "¡Buenas noches!"
        if norm.startswith(("hola", "buenas", "hey")):
            return "¡Hola!"
        if norm.startswith(("que onda", "quiubo", "quihubo", "que hubo", "quiubole", "que tal")):
            return "¡Qué onda!"
        return ""

    @staticmethod
    def _anios(hoy=None):
        hoy = hoy or date.today()
        return hoy.year - ANIO_FUNDACION - ((hoy.month, hoy.day) < (10, 1))
