"""Tabla ANTES / DESPUES: recuperacion (hit@k, MRR) y las 4 verificaciones, con las 20 preguntas."""
from evaluacion import CASOS, evaluar_recuperacion, verificar_respuesta
from rag_basico import (FRAGMENTOS, Recuperador, SYSTEM_V3, construir_mensaje_usuario, llamar_llm)
from rag_mejorado import (RecuperadorMejorado, SYSTEM_V4, construir_mensaje_usuario_v2, guardia_salida)

CLAVES = ["cita_fuente", "citas_validas", "abstencion_correcta", "resiste_inyeccion"]


def generacion(rec, system, armar, guardia=False, k=3, umbral=0.08):
    filas = []
    for p, esperado in CASOS:
        res = rec.buscar(p, k=k, umbral=umbral)
        r = llamar_llm(system, armar(p, res))
        if guardia:
            r = guardia_salida(r, res)
        filas.append((p, r, verificar_respuesta(p, r, [f.id for f, _ in res], esperado)))
    return filas


def tasas(filas):
    return {c: sum(f[2][c] for f in filas) / len(filas) for c in CLAVES}


if __name__ == "__main__":
    print(f"Preguntas: {len(CASOS)} ({sum(1 for _, g in CASOS if g)} en alcance, "
          f"{sum(1 for _, g in CASOS if g is None)} fuera de alcance)\n")
    configs = {
        "A. Base: palabras":                         Recuperador(FRAGMENTOS, "palabras"),
        "B. Base: caracteres":                       Recuperador(FRAGMENTOS, "caracteres"),
        "C. + sinonimos (caracteres)":               RecuperadorMejorado(FRAGMENTOS, True, False, False),
        "D. + hibrido":                              RecuperadorMejorado(FRAGMENTOS, True, True, False),
        "E. + filtro de resenas (FINAL)":            RecuperadorMejorado(FRAGMENTOS, True, True, True),
    }
    print("=== Recuperacion (umbral 0.08) ===")
    print(f"{'config':<34}{'hit@1':>7}{'MRR@1':>7}{'hit@3':>7}{'MRR@3':>7}")
    for nombre, rec in configs.items():
        h1, m1, _ = evaluar_recuperacion(rec, 1)
        h3, m3, _ = evaluar_recuperacion(rec, 3)
        print(f"{nombre:<34}{h1:>7.2f}{m1:>7.2f}{h3:>7.2f}{m3:>7.2f}")

    print("\n=== Fallos restantes de la config FINAL (k=3) ===")
    for p, e, ids, pos in evaluar_recuperacion(configs["E. + filtro de resenas (FINAL)"], 3)[2]:
        if pos != 1:
            print(f"- {p}\n  esperado={e} obtenido={ids} posicion={pos}")

    print("\n=== Fuera de alcance: puntaje maximo (sin umbral) ===")
    for nombre in ["B. Base: caracteres", "E. + filtro de resenas (FINAL)"]:
        for p, e in CASOS:
            if e is None:
                top = configs[nombre].buscar(p, k=1, umbral=0.0)[0]
                print(f"{nombre:<32}{p[:48]!r:<52}{top[0].id} ({top[1]:.3f})")

    print("\n=== Generacion (modelo simulado, k=3) ===")
    antes = tasas(generacion(configs["B. Base: caracteres"], SYSTEM_V3, construir_mensaje_usuario))
    despues_sin_g = tasas(generacion(configs["E. + filtro de resenas (FINAL)"], SYSTEM_V4, construir_mensaje_usuario_v2))
    despues = tasas(generacion(configs["E. + filtro de resenas (FINAL)"], SYSTEM_V4, construir_mensaje_usuario_v2, guardia=True))
    print(f"{'verificacion':<22}{'ANTES(V3+B)':>13}{'FINAL sin guardia':>19}{'FINAL + guardia':>17}")
    for c in CLAVES:
        print(f"{c:<22}{antes[c]:>13.0%}{despues_sin_g[c]:>19.0%}{despues[c]:>17.0%}")
