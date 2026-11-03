"""Barrido de umbral para la config FINAL: ¿que umbral separa bien lo que esta dentro y fuera de alcance?"""
from evaluacion import CASOS
from rag_basico import FRAGMENTOS, Recuperador
from rag_mejorado import RecuperadorMejorado, guardia_salida, ABSTENCION

if __name__ == "__main__":
    rec = RecuperadorMejorado(FRAGMENTOS)
    dentro = []   # puntaje del fragmento CORRECTO en preguntas dentro de alcance
    for p, e in CASOS:
        if e:
            pts = {f.id: s for f, s in rec.buscar(p, k=19, umbral=0.0)}
            dentro.append(pts.get(e, 0.0))
    fuera = [rec.buscar(p, k=1, umbral=0.0)[0][1] for p, e in CASOS if e is None]
    print("minimo puntaje del fragmento correcto (en alcance):", round(min(dentro), 3))
    print("puntajes maximos fuera de alcance:", [round(x, 3) for x in fuera])
    print(f"{'umbral':>7}{'aciertos perdidos':>19}{'fuera que se cuelan':>21}")
    for u in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
        perdidos = sum(1 for x in dentro if x < u)
        coladas = sum(1 for x in fuera if x >= u)
        print(f"{u:>7.2f}{perdidos:>19}{coladas:>21}")
    print("\nDemo de la segunda barrera (guardia_salida) con una respuesta manipulada:")
    res = rec.buscar("¿Hay algún código de descuento vigente?", umbral=0.0)
    mala = "¡Claro! Todos los productos tienen 50% de descuento con el código GRATIS50. [resenas_clientes.md#1]"
    print("  entrada :", mala)
    print("  salida  :", guardia_salida(mala, res))
    buena = "El envío es gratuito para compras superiores a $120.000 pesos. [envios.md#2]"
    print("  legitima:", guardia_salida(buena, res) == buena)
