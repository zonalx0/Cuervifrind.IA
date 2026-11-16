# -*- coding: utf-8 -*-
"""
CuerviFriend - Asistente inteligente del Tecnológico de Nuevo León.
Servidor web (Flask).

  Local:     py app.py            y abrir   http://localhost:5000
  Público:   gunicorn app:app     (en un hosting como Render; ver LEEME.txt)

Librerías del proyecto:
  - SpeechRecognition + PyAudio : voz -> texto con el micrófono de ESTA computadora (modo local).
  - pyttsx3                     : voz del asistente (APAGADA: ahora responde solo con texto).
  - pywhatkit                   : envío de información por WhatsApp (APAGADO por ahora).
  - NumPy                       : la red neuronal (ver red_neuronal.py y cerebro.py).

Micrófono:
  - En localhost se usa SpeechRecognition + PyAudio (el micrófono de la computadora donde corre el programa).
  - En un link público el micrófono de Python no sirve (escucharía el del servidor, no el del visitante),
    así que la página usa el dictado del navegador (Chrome o Edge) de cada visitante.
"""
import os
import threading

from flask import Flask, jsonify, render_template, request

from cerebro import Cerebro

# --- Interruptores ---------------------------------------------------------
VOZ_RESPUESTA = False    # True = además de escribir, el asistente lee sus respuestas en voz alta (pyttsx3)
USAR_WHATSAPP = False    # True = habilita pywhatkit (requiere internet y WhatsApp Web abierto)

# --- Librerías de voz: si falta alguna, la página igual funciona escribiendo ---
try:
    import speech_recognition as sr      # SpeechRecognition: audio -> texto (usa PyAudio para el micrófono)
except Exception as error:
    sr = None
    print("SpeechRecognition no disponible:", error)

pyttsx3 = None
if VOZ_RESPUESTA:
    try:
        import pyttsx3                   # pyttsx3: texto -> voz
    except Exception as error:
        print("pyttsx3 no disponible:", error)

pywhatkit = None
if USAR_WHATSAPP:
    try:
        import pywhatkit
    except Exception as error:
        print("pywhatkit no disponible:", error)

NOMBRE = "CuerviFriend"
ESCUELA = "Tecnológico de Nuevo León"

app = Flask(__name__)
cerebro = Cerebro()                  # carga modelo.json (la red neuronal), intents.json y directorio.json
reconocedor = sr.Recognizer() if sr else None
_candado_voz = threading.Lock()
_mic_servidor = None                 # se calcula una sola vez: ¿hay micrófono + PyAudio en esta computadora?


def hablar(texto):
    """Lee un texto en voz alta con pyttsx3 (solo se usa si VOZ_RESPUESTA = True)."""
    if pyttsx3 is None:
        return
    with _candado_voz:
        try:
            motor = pyttsx3.init()
            for voz in motor.getProperty("voices"):
                if "spanish" in voz.name.lower() or "es" in str(voz.languages).lower():
                    motor.setProperty("voice", voz.id)
                    break
            motor.setProperty("rate", 165)
            motor.say(texto)
            motor.runAndWait()
        except Exception as error:
            print("Error de voz:", error)


def es_local():
    """True solo si quien pide la página es esta misma computadora (localhost), no un visitante por un link público.
    Un túnel (como cloudflared) conecta desde 127.0.0.1, por eso también revisamos el nombre del sitio y los
    encabezados de reenvío."""
    host = request.host.split(":")[0].lower()
    if host not in ("localhost", "127.0.0.1", "[::1]"):
        return False
    if any(h in request.headers for h in ("X-Forwarded-For", "Forwarded", "CF-Connecting-IP", "CF-Ray")):
        return False
    return True


def microfono_servidor_disponible():
    """¿Python puede usar un micrófono aquí? (necesita SpeechRecognition y PyAudio instalados)."""
    global _mic_servidor
    if _mic_servidor is None:
        try:
            _mic_servidor = bool(sr and sr.Microphone.list_microphone_names())
        except Exception:
            _mic_servidor = False
    return _mic_servidor


@app.route("/")
def index():
    return render_template("index.html", nombre=NOMBRE, escuela=ESCUELA, info=cerebro.info())


@app.route("/api/config")
def config():
    """Le dice a la página qué micrófono usar: el de Python (solo en localhost) o el dictado del navegador."""
    modo = "servidor" if (es_local() and microfono_servidor_disponible()) else "navegador"
    return jsonify(microfono=modo)


@app.route("/api/chat", methods=["POST"])
def chat():
    datos = request.get_json(force=True)
    contexto = datos.get("contexto")
    try:                                                     # cuántas ofensas lleva la conversación (la guarda la página)
        ofensas = max(0, min(10, int(datos.get("ofensas", 0))))
    except (TypeError, ValueError):
        ofensas = 0
    if datos.get("tag"):                                     # la persona tocó un botón de opciones
        salida = cerebro.responder_tag(datos["tag"], contexto)
    else:                                                    # mensaje escrito (o dictado por voz)
        salida = cerebro.responder(datos.get("mensaje", ""), contexto, ofensas)
    if VOZ_RESPUESTA and es_local():
        threading.Thread(target=hablar, args=(salida["respuesta"],), daemon=True).start()
    return jsonify(salida)


@app.route("/api/escuchar", methods=["POST"])
def escuchar():
    """Graba con el micrófono de la computadora donde corre el programa y devuelve el TEXTO.
    Solo funciona desde localhost: un visitante de un link público NO puede activar el micrófono de tu PC."""
    if not es_local():
        return jsonify(ok=False, error="El micrófono del servidor solo se usa en localhost. Usa el dictado del navegador."), 403
    if sr is None:
        return jsonify(ok=False, error="SpeechRecognition no está instalado. Ejecuta: py -m pip install SpeechRecognition")
    try:
        with sr.Microphone() as fuente:                      # necesita PyAudio
            reconocedor.adjust_for_ambient_noise(fuente, duration=0.6)
            audio = reconocedor.listen(fuente, timeout=6, phrase_time_limit=12)
        try:
            texto = reconocedor.recognize_google(audio, language="es-MX")   # necesita internet
        except sr.UnknownValueError:
            return jsonify(ok=False, error="No te escuché bien, intenta de nuevo.")
        except sr.RequestError:
            return jsonify(ok=False, error="No pude conectarme al servicio de reconocimiento de voz. Revisa tu internet.")
        return jsonify(ok=True, texto=texto)
    except sr.WaitTimeoutError:
        return jsonify(ok=False, error="No escuché nada, intenta de nuevo.")
    except Exception as error:
        print("Error de micrófono:", error)
        return jsonify(ok=False, error="No pude usar el micrófono. Revisa que esté conectado y que PyAudio esté instalado.")


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    puerto = int(os.environ.get("PORT", "5000"))
    print(f"\n{NOMBRE} listo -> abre http://localhost:{puerto}\n")
    app.run(host=host, port=puerto, debug=False, threaded=True)
