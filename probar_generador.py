# -*- coding: utf-8 -*-
"""
Prueba del generador de respuestas SIN gastar nada: levanta una API falsa en tu computadora y comprueba que
  - sin llave todo sigue igual,
  - una redacción buena se acepta,
  - una redacción que inventa un correo o un teléfono se descarta,
  - un trámite al que le falta un correo se descarta,
  - si la API falla o tarda se usa la respuesta de siempre,
  - crisis, groserías y JOMADAIS nunca se mandan al modelo,
  - los límites de uso funcionan.
Uso:   py probar_generador.py
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODO = {"valor": "bueno"}
PETICIONES = []


class ApiFalsa(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        PETICIONES.append(dict(llave=self.headers.get("x-api-key"), version=self.headers.get("anthropic-version"), cuerpo=cuerpo))
        hechos = cuerpo["messages"][0]["content"].split("HECHOS (lo único que puedes afirmar):\n", 1)[1]
        modo = MODO["valor"]
        if modo == "error":
            self.send_response(500); self.end_headers(); return
        if modo == "lento":
            time.sleep(2)
        texto = "¡Claro! Aquí lo tienes, con otras palabras:\n" + hechos
        if modo == "invento_correo":
            texto += "\nTambién escríbele a inventado@falso.com"
        if modo == "invento_telefono":
            texto += "\nO llama al 5512345678"
        if modo == "falta_correo":
            texto = "\n".join(l for l in texto.split("\n") if "@" not in l)
        respuesta = json.dumps({"content": [{"type": "text", "text": texto}]}).encode("utf-8")
        try:
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(respuesta)
        except BrokenPipeError:                 # el cliente ya se cansó de esperar (caso "lento")
            pass


servidor = HTTPServer(("127.0.0.1", 0), ApiFalsa)
threading.Thread(target=servidor.serve_forever, daemon=True).start()
os.environ["CUERVI_API_URL"] = f"http://127.0.0.1:{servidor.server_port}/v1/messages"
os.environ["CUERVI_TIMEOUT"] = "1"
os.environ["CUERVI_LIMITE_IP_HORA"] = "6"

import generador                      # noqa: E402  (se importa después de preparar la API falsa)
from app import app                   # noqa: E402

cliente = app.test_client()
fallos = []


def pregunta(mensaje, ip="1.1.1.1"):
    return cliente.post("/api/chat", json={"mensaje": mensaje}, headers={"X-Forwarded-For": ip}).get_json()


def revisar(nombre, condicion):
    print(("  OK   " if condicion else "  FALLA ") + nombre)
    if not condicion:
        fallos.append(nombre)


print("1) Sin llave")
os.environ.pop("ANTHROPIC_API_KEY", None)
r = pregunta("quiero una beca")
revisar("responde con el texto de siempre y no llama a la API", r["generada"] is False and not PETICIONES)
revisar("/api/config avisa que no es generativo", cliente.get("/api/config").get_json()["generativo"] is False)

print("2) Con llave")
os.environ["ANTHROPIC_API_KEY"] = "llave-de-prueba"
revisar("/api/config avisa que sí es generativo", cliente.get("/api/config").get_json()["generativo"] is True)
r = pregunta("ocupo una beca")
revisar("acepta una redacción fiel", r["generada"] is True and r["respuesta"].startswith("¡Claro!") and "respuesta_base" in r)
revisar("manda la llave y la versión de API", PETICIONES[-1]["llave"] == "llave-de-prueba" and PETICIONES[-1]["version"] == "2023-06-01")
revisar("el mensaje del alumno viaja delimitado", "<mensaje>ocupo una beca</mensaje>" in PETICIONES[-1]["cuerpo"]["messages"][0]["content"])

print("3) Guardas anti-invento")
MODO["valor"] = "invento_correo"
r = pregunta("quiero una beca", "2.2.2.2")
revisar("descarta una respuesta con un correo inventado", r["generada"] is False and "inventado@falso.com" not in r["respuesta"])
MODO["valor"] = "invento_telefono"
r = pregunta("quiero una beca", "2.2.2.2")
revisar("descarta una respuesta con un teléfono inventado", r["generada"] is False and "5512345678" not in r["respuesta"])
MODO["valor"] = "falta_correo"
r = pregunta("quiero una beca", "3.3.3.3")
revisar("descarta un trámite al que le falta el correo", r["generada"] is False)
MODO["valor"] = "bueno"
r = pregunta("hola buenas tardes", "3.3.3.3")
revisar("acepta una charla bien redactada", r["generada"] is True)

print("4) Respuestas que nunca se generan")
antes = len(PETICIONES)
for texto in ("me quiero morir", "pene", "quiénes son jomadais", "echamos un tiro"):
    r = pregunta(texto, "4.4.4.4")
    revisar(f"«{texto}» se queda con su texto fijo", r["generada"] is False)
revisar("ninguna de esas llegó a la API", len(PETICIONES) == antes)
r = pregunta("elefante", "4.4.4.4")
revisar("si la red no entendió, tampoco se genera", r["tag"] is None and r["generada"] is False)

print("5) Fallos de la API")
generador._fallos.update(seguidos=0, pausa_hasta=0)
MODO["valor"] = "error"
r = pregunta("quiero una beca", "5.5.5.5")
revisar("si la API da error, responde con el texto de siempre", r["generada"] is False and "Ubicación" in r["respuesta"])
MODO["valor"] = "lento"
r = pregunta("quiero una beca", "5.5.5.5")
revisar("si tarda más del tiempo límite, también", r["generada"] is False)
MODO["valor"] = "error"
pregunta("quiero una beca", "5.5.5.5")
antes = len(PETICIONES)
MODO["valor"] = "bueno"
r = pregunta("quiero una beca", "5.5.5.6")
revisar("tras 3 fallos seguidos hace una pausa y no insiste", r["generada"] is False and len(PETICIONES) == antes)

print("6) Límite por visitante")
generador._fallos.update(seguidos=0, pausa_hasta=0)
resultados = [pregunta("ocupo una constancia", "9.9.9.9")["generada"] for _ in range(8)]
revisar("después de 6 respuestas por hora ya no genera más (6 sí, 2 no)", resultados == [True] * 6 + [False] * 2)

print()
print("TODO BIEN" if not fallos else f"FALLARON {len(fallos)}: {fallos}")
servidor.shutdown()
raise SystemExit(1 if fallos else 0)
