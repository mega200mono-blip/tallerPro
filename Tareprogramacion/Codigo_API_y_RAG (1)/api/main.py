"""API REST de "Ori", asistente de Cafe Origen (empresa ficticia). Sesiones 9 y 10.
Ejecutar:  uvicorn main:app --reload   (desde la carpeta api/)  ->  http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path as PathParam, Query, status
from pydantic import BaseModel, Field

# Reutiliza el RAG del caso de estudio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "caso_rag"))
from rag_basico import FRAGMENTOS, llamar_llm  # noqa: E402
from rag_mejorado import (ABSTENCION, SYSTEM_V4, RecuperadorMejorado,  # noqa: E402
                          construir_mensaje_usuario_v2, guardia_salida)

app = FastAPI(
    title="Ori - Asistente de Cafe Origen",
    description="API REST que expone politicas de la tienda y responde preguntas con RAG basico "
                "(TF-IDF + prompt con fuentes citadas). Empresa ficticia con fines academicos.",
    version="1.0.0",
)
RECUPERADOR = RecuperadorMejorado(FRAGMENTOS)


# ----------------------------- Modelos Pydantic (schemas) -----------------------------
class PreguntaIn(BaseModel):
    pregunta: str = Field(..., min_length=3, max_length=300, examples=["¿Cuánto cuesta el envío?"])
    k: int = Field(3, ge=1, le=5, description="Cuántos fragmentos recuperar")


class FragmentoOut(BaseModel):
    id: str = Field(examples=["envios.md#2"])
    fuente: str
    titulo: str
    texto: str


class FuenteRecuperada(BaseModel):
    id: str
    puntaje: float = Field(ge=0, le=1)


class RespuestaOut(BaseModel):
    pregunta: str
    recuperados: list[FuenteRecuperada]
    respuesta: str
    se_abstuvo: bool


# ----------------------------- Pedidos: modelos -----------------------------
EstadoPedido = Literal["pendiente", "en_preparacion", "enviado", "entregado", "cancelado"]


class PedidoIn(BaseModel):
    """Datos que envia el cliente para crear un pedido (el id lo genera el servidor)."""
    producto: str = Field(..., min_length=2, max_length=100, examples=["Café molido 500 g"])
    direccion: str = Field(..., min_length=5, max_length=200, examples=["Calle 80 # 45-20, Bogotá"])
    estado: EstadoPedido = Field("pendiente", description="Estado inicial del pedido")


class EstadoIn(BaseModel):
    """Cuerpo para actualizar unicamente el estado de un pedido."""
    estado: EstadoPedido


class PedidoOut(BaseModel):
    id: int = Field(examples=[1])
    producto: str
    estado: EstadoPedido
    direccion: str


# "Base de datos" en memoria (se reinicia al reiniciar el servidor)
PEDIDOS: dict[int, dict] = {}
_siguiente_id = 1


# ----------------------------- Endpoints -----------------------------
@app.get("/salud", tags=["sistema"], summary="Estado del servicio")
def salud() -> dict:
    return {"estado": "ok", "fragmentos_indexados": len(FRAGMENTOS), "pedidos_registrados": len(PEDIDOS)}


@app.get("/politicas", response_model=list[FragmentoOut], tags=["politicas"],
         summary="Lista fragmentos de politicas")
def listar_politicas(
    fuente: Annotated[str | None, Query(description="Filtra por documento, p. ej. envios.md")] = None,
    limite: Annotated[int, Query(ge=1, le=50)] = 20,
):
    datos = [f for f in FRAGMENTOS if fuente is None or f.fuente == fuente]
    return datos[:limite]


@app.get("/politicas/{fragmento_id}", response_model=FragmentoOut, tags=["politicas"],
         summary="Obtiene un fragmento por su identificador")
def obtener_politica(fragmento_id: Annotated[str, PathParam(pattern=r"^[a-z_]+\.md#\d+$",
                                                            examples=["envios.md#2"])]):
    for f in FRAGMENTOS:
        if f.id == fragmento_id:
            return f
    raise HTTPException(status_code=404, detail=f"No existe el fragmento '{fragmento_id}'")


# ----------------------------- Pedidos: CRUD -----------------------------
@app.post("/pedidos", response_model=PedidoOut, status_code=status.HTTP_201_CREATED,
          tags=["pedidos"], summary="Crea un pedido (Create)")
def crear_pedido(datos: PedidoIn):
    global _siguiente_id
    pedido = {"id": _siguiente_id, **datos.model_dump()}
    PEDIDOS[_siguiente_id] = pedido
    _siguiente_id += 1
    return pedido


@app.get("/pedidos", response_model=list[PedidoOut], tags=["pedidos"],
         summary="Lista todos los pedidos (Read)")
def listar_pedidos(
    estado: Annotated[EstadoPedido | None, Query(description="Filtra por estado")] = None,
):
    return [p for p in PEDIDOS.values() if estado is None or p["estado"] == estado]


@app.get("/pedidos/{pedido_id}", response_model=PedidoOut, tags=["pedidos"],
         summary="Busca un pedido por su id (Read)")
def obtener_pedido(pedido_id: Annotated[int, PathParam(ge=1, examples=[1])]):
    pedido = PEDIDOS.get(pedido_id)
    if pedido is None:
        raise HTTPException(status_code=404, detail=f"No existe el pedido con id {pedido_id}")
    return pedido


@app.patch("/pedidos/{pedido_id}/estado", response_model=PedidoOut, tags=["pedidos"],
           summary="Actualiza el estado de un pedido (Update)")
def actualizar_estado_pedido(pedido_id: Annotated[int, PathParam(ge=1, examples=[1])],
                             datos: EstadoIn):
    pedido = PEDIDOS.get(pedido_id)
    if pedido is None:
        raise HTTPException(status_code=404, detail=f"No existe el pedido con id {pedido_id}")
    pedido["estado"] = datos.estado
    return pedido


@app.delete("/pedidos/{pedido_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["pedidos"],
            summary="Elimina un pedido (Delete)")
def eliminar_pedido(pedido_id: Annotated[int, PathParam(ge=1, examples=[1])]):
    if PEDIDOS.pop(pedido_id, None) is None:
        raise HTTPException(status_code=404, detail=f"No existe el pedido con id {pedido_id}")


@app.post("/preguntar", response_model=RespuestaOut, tags=["asistente"],
          summary="Responde una pregunta usando RAG")
def preguntar(datos: PreguntaIn):
    resultados = RECUPERADOR.buscar(datos.pregunta, k=datos.k)
    respuesta = llamar_llm(SYSTEM_V4, construir_mensaje_usuario_v2(datos.pregunta, resultados))
    respuesta = guardia_salida(respuesta, resultados)
    return RespuestaOut(
        pregunta=datos.pregunta,
        recuperados=[FuenteRecuperada(id=f.id, puntaje=round(min(p, 1.0), 3)) for f, p in resultados],
        respuesta=respuesta,
        se_abstuvo=respuesta == ABSTENCION,
    )