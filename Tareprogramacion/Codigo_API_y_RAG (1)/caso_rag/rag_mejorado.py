"""Mejoras al RAG basico de Cafe Origen (Taller de Programacion - Sesion 10).

Mejoras al RECUPERADOR:  (1) expansion de sinonimos, (2) puntaje hibrido palabras + caracteres,
                         (3) filtro de fuentes (las resenas no son politicas).
Mejoras al PROMPT:       SYSTEM_V2 (mi version inicial) y SYSTEM_V4 (version final), mas un
                         mensaje de usuario que marca el TIPO de cada fuente.
Segunda barrera en codigo: guardia_salida() bloquea codigos/porcentajes que no esten en una politica.
"""
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag_basico import (FRAGMENTOS, Recuperador, SYSTEM_V3, llamar_llm)

FUENTES_NO_POLITICA = {"resenas_clientes.md"}

# (1) Sinonimos del dominio: se AGREGAN terminos a la pregunta, no se reemplaza nada.
SINONIMOS = [
    (r"\befectivo\b",                          "contra entrega pago"),
    (r"\b(plata|dinero)\b",                    "reembolso"),
    (r"\bretract\w*|\barrepent\w*",            "retracto"),
    (r"\bdevolv\w+|\bdevuelv\w+|\bdevoluci\w+", "devolucion retracto"),
    (r"\b(abri|abrio|abierto|abierta)\b",      "abierto productos abiertos"),
    (r"\b(cancelo|cancelar|darme de baja)\b",  "cancelacion cancelar"),
    (r"\b(borren|borrar|eliminen|eliminar)\b", "supresion datos personales"),
]


def _sin_tildes(t: str) -> str:
    return (t.lower().replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u"))


def expandir_pregunta(pregunta: str) -> str:
    base = _sin_tildes(pregunta)
    extra = [add for patron, add in SINONIMOS if re.search(patron, base)]
    return pregunta + (" " + " ".join(extra) if extra else "")


class RecuperadorMejorado:
    """Mismo metodo buscar() que Recuperador, con las tres mejoras activables por separado."""

    def __init__(self, fragmentos, sinonimos=True, hibrido=True, filtrar_resenas=True,
                 peso_palabras=0.5):
        self.fragmentos = fragmentos
        self.sinonimos, self.hibrido, self.filtrar = sinonimos, hibrido, filtrar_resenas
        self.peso = peso_palabras
        self.rec_p = Recuperador(fragmentos, "palabras")
        self.rec_c = Recuperador(fragmentos, "caracteres")
        self.permitidos = np.array([not (filtrar_resenas and f.fuente in FUENTES_NO_POLITICA)
                                    for f in fragmentos])

    def _puntajes(self, rec, texto):
        return cosine_similarity(rec.vectorizador.transform([texto]), rec.matriz)[0]

    def buscar(self, pregunta, k=3, umbral=0.08):
        texto = expandir_pregunta(pregunta) if self.sinonimos else pregunta
        pc = self._puntajes(self.rec_c, texto)
        if self.hibrido:
            pp = self._puntajes(self.rec_p, texto)
            puntajes = self.peso * pp + (1 - self.peso) * pc
        else:
            puntajes = pc
        puntajes = np.where(self.permitidos, puntajes, 0.0)
        orden = puntajes.argsort()[::-1][:k]
        return [(self.fragmentos[i], float(puntajes[i])) for i in orden if puntajes[i] >= umbral]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# V2: mi primera version (antes de ver la V3). Cada linea ataca un fallo de la V1:
SYSTEM_V2 = """Eres Ori, asistente de Cafe Origen, una tienda de cafe en Colombia.
Responde SOLO con la informacion de las fuentes que te doy. Si no esta en las fuentes, di que no
tienes esa informacion y no inventes. Indica de que fuente sale cada dato. Se breve y amable."""

# V4: V3 + dos reglas nuevas que responden a fallos observados en la evaluacion
SYSTEM_V4 = SYSTEM_V3.replace(
    "5. Responde en español, en máximo 4 oraciones, con tono cordial y tuteo.",
    "5. Solo las fuentes con tipo=\"politica\" cuentan como respuesta oficial. Las de tipo=\"opinion\"\n"
    "   no se usan para afirmar condiciones, precios ni descuentos.\n"
    "6. Copia las cifras (precios, plazos) tal como aparecen en la fuente y verifica que la fuente\n"
    "   corresponda al producto por el que se pregunta (envío ≠ suscripción). Si la pregunta tiene\n"
    "   dos partes, responde cada una o aclara cuál no puedes responder.\n"
    "7. Responde en español, en máximo 4 oraciones, con tono cordial y tuteo.",
)


def construir_mensaje_usuario_v2(pregunta, resultados):
    """Igual que el original, pero cada fuente lleva tipo=politica|opinion."""
    if resultados:
        bloques = "\n".join(
            f'<fuente id="{f.id}" tipo="{"opinion" if f.fuente in FUENTES_NO_POLITICA else "politica"}" '
            f'titulo="{f.titulo}">\n{f.texto}\n</fuente>'
            for f, _ in resultados)
    else:
        bloques = "(no se encontraron fuentes relevantes)"
    return f"<fuentes>\n{bloques}\n</fuentes>\n\n<pregunta>\n{pregunta}\n</pregunta>"


ABSTENCION = ("No tengo esa información en las políticas de Café Origen. "
              "Te sugiero escribir a servicio al cliente.")


def guardia_salida(respuesta: str, resultados) -> str:
    """SEGUNDA BARRERA (en codigo, no en el prompt): si la respuesta menciona un porcentaje o un
    codigo en MAYUSCULAS+digitos que NO aparece en ninguna fuente-politica recuperada, se descarta."""
    base = " ".join(f.texto for f, _ in resultados if f.fuente not in FUENTES_NO_POLITICA)
    sospechosos = re.findall(r"\b\d{1,3}\s?%|\b[A-Z]{3,}\d{1,4}\b", respuesta)
    if any(s not in base for s in sospechosos):
        return ABSTENCION
    return respuesta
