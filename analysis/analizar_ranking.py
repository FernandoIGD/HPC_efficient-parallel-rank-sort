"""Analiza las mediciones del ordenamiento por ranking y genera sus figuras."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from scipy.optimize import nnls


PROCESOS = (1, 4, 9, 16, 25)
TAMANOS = (3600, 7200, 10800, 14400, 18000)
MARCADORES = ("o", "s", "^", "D", "P")
TINTA = "#1a1a1a"
GRIS = "#858585"


@dataclass(frozen=True)
class Medicion:
    """Una observación experimental de tiempos para una configuración."""

    procesos: int
    tamano: int
    ejecucion: float
    computo: float
    comunicacion: float


@dataclass(frozen=True)
class Resumen:
    """Estadísticos y métricas de una configuración experimental."""

    procesos: int
    tamano: int
    ejecucion_mediana: float
    ejecucion_q1: float
    ejecucion_q3: float
    computo_mediana: float
    computo_q1: float
    computo_q3: float
    comunicacion_mediana: float
    comunicacion_q1: float
    comunicacion_q3: float
    speedup: float = math.nan
    eficiencia: float = math.nan


@dataclass(frozen=True)
class ModeloTeorico:
    """Coeficientes no negativos del modelo teórico de referencia ajustado."""

    coeficiente_latencia: float
    coeficiente_ancho_banda: float
    coeficiente_computo: float
    r2_ejecucion: float

    def predecir_computo(self, tamano: int, procesos: int) -> float:
        """Predice el tiempo de cómputo.

        Args:
            tamano: Número total de elementos N.
            procesos: Número de procesos p.

        Returns:
            Tiempo de cómputo ajustado en segundos.
        """
        return (
            self.coeficiente_computo
            * tamano
            * math.log(tamano)
            / math.sqrt(procesos)
        )

    def predecir_comunicacion(self, tamano: int, procesos: int) -> float:
        """Predice el tiempo de comunicación.

        Args:
            tamano: Número total de elementos N.
            procesos: Número de procesos p.

        Returns:
            Tiempo de comunicación ajustado en segundos.
        """
        return (
            self.coeficiente_latencia * (math.sqrt(procesos) - 1)
            + self.coeficiente_ancho_banda * tamano * (1 - 1 / procesos)
        )

    def predecir_ejecucion(self, tamano: int, procesos: int) -> float:
        """Predice el tiempo total como cómputo más comunicación.

        Args:
            tamano: Número total de elementos N.
            procesos: Número de procesos p.

        Returns:
            Tiempo total ajustado en segundos.
        """
        return self.predecir_computo(tamano, procesos) + self.predecir_comunicacion(
            tamano, procesos
        )


def cargar_mediciones(ruta_csv: Path) -> list[Medicion]:
    """Carga y valida el CSV de observaciones.

    Args:
        ruta_csv: Archivo con procesos, N y los tres tiempos medidos.

    Returns:
        Lista de 500 mediciones válidas.

    Raises:
        ValueError: Si faltan columnas, configuraciones o repeticiones.
    """
    columnas = {
        "processes",
        "N",
        "execution_s",
        "computation_s",
        "communication_s",
    }
    mediciones: list[Medicion] = []
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if set(lector.fieldnames or ()) != columnas:
            raise ValueError("El CSV no contiene exactamente las cinco columnas esperadas")
        for fila in lector:
            medicion = Medicion(
                procesos=int(fila["processes"]),
                tamano=int(fila["N"]),
                ejecucion=float(fila["execution_s"]),
                computo=float(fila["computation_s"]),
                comunicacion=float(fila["communication_s"]),
            )
            if min(medicion.ejecucion, medicion.computo, medicion.comunicacion) < 0:
                raise ValueError("Los tiempos deben ser no negativos")
            if abs(medicion.ejecucion - medicion.computo - medicion.comunicacion) > 1e-6:
                raise ValueError("Ejecución no coincide con cómputo más comunicación")
            mediciones.append(medicion)

    conteos: dict[tuple[int, int], int] = {}
    for medicion in mediciones:
        clave = (medicion.procesos, medicion.tamano)
        conteos[clave] = conteos.get(clave, 0) + 1
    esperadas = {(p, n) for p in PROCESOS for n in TAMANOS}
    if set(conteos) != esperadas or any(conteos[clave] != 20 for clave in esperadas):
        raise ValueError("Se esperaban 20 repeticiones para cada una de 25 configuraciones")
    return mediciones


def _estadisticos(valores: list[float]) -> tuple[float, float, float]:
    arreglo = np.asarray(valores, dtype=float)
    q1, mediana, q3 = np.percentile(arreglo, [25, 50, 75])
    return float(mediana), float(q1), float(q3)


def resumir_mediciones(mediciones: list[Medicion]) -> list[Resumen]:
    """Calcula mediana, IQR, speedup y eficiencia.

    Args:
        mediciones: Observaciones validadas.

    Returns:
        Un resumen por cada par (p, N).
    """
    resumenes: list[Resumen] = []
    for tamano in TAMANOS:
        por_proceso: list[Resumen] = []
        for procesos in PROCESOS:
            grupo = [
                medicion
                for medicion in mediciones
                if medicion.procesos == procesos and medicion.tamano == tamano
            ]
            ejecucion = _estadisticos([medicion.ejecucion for medicion in grupo])
            computo = _estadisticos([medicion.computo for medicion in grupo])
            comunicacion = _estadisticos([medicion.comunicacion for medicion in grupo])
            por_proceso.append(
                Resumen(
                    procesos=procesos,
                    tamano=tamano,
                    ejecucion_mediana=ejecucion[0],
                    ejecucion_q1=ejecucion[1],
                    ejecucion_q3=ejecucion[2],
                    computo_mediana=computo[0],
                    computo_q1=computo[1],
                    computo_q3=computo[2],
                    comunicacion_mediana=comunicacion[0],
                    comunicacion_q1=comunicacion[1],
                    comunicacion_q3=comunicacion[2],
                )
            )
        tiempo_serial = por_proceso[0].ejecucion_mediana
        resumenes.extend(
            Resumen(
                **{
                    **resumen.__dict__,
                    "speedup": tiempo_serial / resumen.ejecucion_mediana,
                    "eficiencia": tiempo_serial
                    / (resumen.procesos * resumen.ejecucion_mediana),
                }
            )
            for resumen in por_proceso
        )
    return resumenes


def ajustar_modelo_teorico(resumenes: list[Resumen]) -> ModeloTeorico:
    """Ajusta globalmente la forma asintótica de la implementación de referencia.

    Args:
        resumenes: Medianas experimentales por configuración.

    Returns:
        Modelo con coeficientes físicos no negativos y R cuadrado total.
    """
    base_computo = np.asarray(
        [
            resumen.tamano * math.log(resumen.tamano) / math.sqrt(resumen.procesos)
            for resumen in resumenes
        ],
        dtype=float,
    )[:, None]
    tiempos_computo = np.asarray(
        [resumen.computo_mediana for resumen in resumenes], dtype=float
    )
    coeficiente_computo = nnls(base_computo, tiempos_computo)[0][0]

    base_comunicacion = np.asarray(
        [
            [
                math.sqrt(resumen.procesos) - 1,
                resumen.tamano * (1 - 1 / resumen.procesos),
            ]
            for resumen in resumenes
        ],
        dtype=float,
    )
    tiempos_comunicacion = np.asarray(
        [resumen.comunicacion_mediana for resumen in resumenes], dtype=float
    )
    coeficiente_latencia, coeficiente_ancho_banda = nnls(
        base_comunicacion, tiempos_comunicacion
    )[0]

    predicciones = (
        base_computo[:, 0] * coeficiente_computo
        + base_comunicacion[:, 0] * coeficiente_latencia
        + base_comunicacion[:, 1] * coeficiente_ancho_banda
    )
    observados = np.asarray(
        [resumen.ejecucion_mediana for resumen in resumenes], dtype=float
    )
    suma_residuos = float(np.sum((observados - predicciones) ** 2))
    suma_total = float(np.sum((observados - np.mean(observados)) ** 2))
    r2 = 1.0 - suma_residuos / suma_total
    return ModeloTeorico(
        coeficiente_latencia=float(coeficiente_latencia),
        coeficiente_ancho_banda=float(coeficiente_ancho_banda),
        coeficiente_computo=float(coeficiente_computo),
        r2_ejecucion=r2,
    )


def guardar_resumen(resumenes: list[Resumen], ruta_salida: Path) -> None:
    """Guarda las métricas agregadas para reproducir las figuras.

    Args:
        resumenes: Estadísticos por configuración.
        ruta_salida: Destino del CSV agregado.
    """
    columnas = (
        "procesos",
        "N",
        "ejecucion_mediana_s",
        "ejecucion_q1_s",
        "ejecucion_q3_s",
        "computo_mediana_s",
        "computo_q1_s",
        "computo_q3_s",
        "comunicacion_mediana_s",
        "comunicacion_q1_s",
        "comunicacion_q3_s",
        "speedup",
        "eficiencia",
        "costo_paralelo_s",
    )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for resumen in resumenes:
            escritor.writerow(
                {
                    "procesos": resumen.procesos,
                    "N": resumen.tamano,
                    "ejecucion_mediana_s": resumen.ejecucion_mediana,
                    "ejecucion_q1_s": resumen.ejecucion_q1,
                    "ejecucion_q3_s": resumen.ejecucion_q3,
                    "computo_mediana_s": resumen.computo_mediana,
                    "computo_q1_s": resumen.computo_q1,
                    "computo_q3_s": resumen.computo_q3,
                    "comunicacion_mediana_s": resumen.comunicacion_mediana,
                    "comunicacion_q1_s": resumen.comunicacion_q1,
                    "comunicacion_q3_s": resumen.comunicacion_q3,
                    "speedup": resumen.speedup,
                    "eficiencia": resumen.eficiencia,
                    "costo_paralelo_s": resumen.procesos
                    * resumen.ejecucion_mediana,
                }
            )


def _configurar_estilo() -> tuple[dict[int, object], dict[int, str]]:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": TINTA,
            "axes.linewidth": 0.8,
            "axes.labelcolor": TINTA,
            "text.color": TINTA,
            "xtick.color": TINTA,
            "ytick.color": TINTA,
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    tonos = mpl.colormaps["Blues"]([0.38, 0.52, 0.66, 0.80, 0.96])
    colores = {tamano: tonos[indice] for indice, tamano in enumerate(TAMANOS)}
    marcadores = {
        tamano: MARCADORES[indice] for indice, tamano in enumerate(TAMANOS)
    }
    return colores, marcadores


def _estilizar_eje_procesos(eje: plt.Axes) -> None:
    eje.set_xscale("log")
    eje.set_xlim(0.85, 29)
    eje.xaxis.set_major_locator(FixedLocator(PROCESOS))
    eje.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{int(valor)}"))
    eje.set_xlabel(r"Número de procesos $p$")
    eje.set_axisbelow(True)


def _estilizar_eje_logaritmico(eje: plt.Axes) -> None:
    eje.set_yscale("log")
    eje.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
    eje.yaxis.set_minor_formatter(NullFormatter())


def _grupo_por_tamano(resumenes: list[Resumen], tamano: int) -> list[Resumen]:
    return sorted(
        [resumen for resumen in resumenes if resumen.tamano == tamano],
        key=lambda resumen: resumen.procesos,
    )


def _dibujar_tiempo_medido(
    eje: plt.Axes,
    resumenes: list[Resumen],
    prefijo: str,
    colores: dict[int, object],
    marcadores: dict[int, str],
) -> None:
    for tamano in TAMANOS:
        grupo = _grupo_por_tamano(resumenes, tamano)
        medianas = np.asarray([getattr(resumen, f"{prefijo}_mediana") for resumen in grupo])
        q1 = np.asarray([getattr(resumen, f"{prefijo}_q1") for resumen in grupo])
        q3 = np.asarray([getattr(resumen, f"{prefijo}_q3") for resumen in grupo])
        eje.errorbar(
            PROCESOS,
            medianas,
            yerr=np.vstack((medianas - q1, q3 - medianas)),
            color=colores[tamano],
            marker=marcadores[tamano],
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.7,
            elinewidth=0.8,
            capsize=2.2,
            label=rf"$N={tamano:,}$".replace(",", r"\,"),
        )
    _estilizar_eje_procesos(eje)
    _estilizar_eje_logaritmico(eje)
    eje.set_ylabel("Tiempo [s]")


def _dibujar_tiempo_teorico(
    eje: plt.Axes,
    modelo: ModeloTeorico,
    componente: str,
    colores: dict[int, object],
    marcadores: dict[int, str],
) -> None:
    predictores = {
        "ejecucion": modelo.predecir_ejecucion,
        "computo": modelo.predecir_computo,
        "comunicacion": modelo.predecir_comunicacion,
    }
    predecir = predictores[componente]
    for tamano in TAMANOS:
        eje.plot(
            PROCESOS,
            [predecir(tamano, procesos) for procesos in PROCESOS],
            color=colores[tamano],
            marker=marcadores[tamano],
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.7,
            label=rf"$N={tamano:,}$".replace(",", r"\,"),
        )
    _estilizar_eje_procesos(eje)
    _estilizar_eje_logaritmico(eje)


def _guardar_figura(figura: plt.Figure, directorio: Path, nombre: str) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    figura.savefig(directorio / f"{nombre}.png")
    plt.close(figura)


def graficar_tiempos(
    resumenes: list[Resumen], modelo: ModeloTeorico, directorio: Path
) -> None:
    """Compara tiempos medidos y teóricos por componente.

    Args:
        resumenes: Estadísticos por configuración.
        directorio: Carpeta para las figuras PNG.
    """
    colores, marcadores = _configurar_estilo()
    figura, ejes = plt.subplots(
        3,
        2,
        figsize=(9.6, 8.2),
        sharex="col",
        sharey="row",
    )
    componentes = (
        ("ejecucion", "Ejecución [s]"),
        ("computo", "Cómputo [s]"),
        ("comunicacion", "Comunicación [s]"),
    )
    for fila, (componente, etiqueta_y) in enumerate(componentes):
        _dibujar_tiempo_medido(
            ejes[fila, 0], resumenes, componente, colores, marcadores
        )
        _dibujar_tiempo_teorico(
            ejes[fila, 1], modelo, componente, colores, marcadores
        )
        ejes[fila, 0].set_ylabel(etiqueta_y)
        if fila < 2:
            ejes[fila, 0].set_xlabel("")
            ejes[fila, 1].set_xlabel("")

    maximo_comunicacion = max(
        max(resumen.comunicacion_q3 for resumen in resumenes),
        max(
            modelo.predecir_comunicacion(tamano, procesos)
            for tamano in TAMANOS
            for procesos in PROCESOS
        ),
    )
    ejes[2, 0].set_yscale("linear")
    ejes[2, 0].set_ylim(0, maximo_comunicacion * 1.08)

    ejes[0, 0].set_title("Tiempo medido")
    ejes[0, 1].set_title("Tiempo teórico")
    asas, etiquetas = ejes[0, 0].get_legend_handles_labels()
    figura.legend(
        asas,
        etiquetas,
        loc="lower center",
        ncol=len(TAMANOS),
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figura.tight_layout(rect=(0, 0.065, 1, 1))
    _guardar_figura(figura, directorio, "tiempos_comparacion")


def graficar_speedup_eficiencia(
    resumenes: list[Resumen], modelo: ModeloTeorico, directorio: Path
) -> None:
    """Compara speedup y eficiencia medidos con sus valores teóricos.

    Args:
        resumenes: Estadísticos por configuración.
        directorio: Carpeta para las figuras PNG.
    """
    colores, marcadores = _configurar_estilo()
    figura, ejes = plt.subplots(
        2,
        2,
        figsize=(9.6, 7.2),
        sharex="col",
        sharey="row",
    )
    valores_p = np.asarray(PROCESOS, dtype=float)
    for columna in range(2):
        ejes[0, columna].plot(
            valores_p,
            valores_p,
            "--",
            color=GRIS,
            linewidth=1.2,
            label="Ideal",
        )
        ejes[1, columna].axhline(
            1.0,
            ls="--",
            color=GRIS,
            linewidth=1.2,
            label=r"Ideal ($E=1$)",
        )

    for tamano in TAMANOS:
        grupo = _grupo_por_tamano(resumenes, tamano)
        etiqueta = rf"$N={tamano:,}$".replace(",", r"\,")
        estilo = {
            "color": colores[tamano],
            "marker": marcadores[tamano],
            "markersize": 5.5,
            "markeredgecolor": "white",
            "markeredgewidth": 0.6,
            "linewidth": 1.7,
        }
        ejes[0, 0].plot(
            PROCESOS,
            [resumen.speedup for resumen in grupo],
            label=etiqueta,
            **estilo,
        )
        ejes[1, 0].plot(
            PROCESOS,
            [resumen.eficiencia for resumen in grupo],
            **estilo,
        )
        tiempos_teoricos = np.asarray(
            [modelo.predecir_ejecucion(tamano, procesos) for procesos in PROCESOS]
        )
        speedup_teorico = tiempos_teoricos[0] / tiempos_teoricos
        eficiencia_teorica = speedup_teorico / valores_p
        ejes[0, 1].plot(PROCESOS, speedup_teorico, **estilo)
        ejes[1, 1].plot(PROCESOS, eficiencia_teorica, **estilo)

    for columna in range(2):
        _estilizar_eje_logaritmico(ejes[0, columna])
        _estilizar_eje_procesos(ejes[0, columna])
        _estilizar_eje_procesos(ejes[1, columna])
        ejes[0, columna].set_xlabel("")
        ejes[1, columna].set_ylim(0, 1.05)

    ejes[0, 0].set_title("Valores medidos")
    ejes[0, 1].set_title("Valores teóricos")
    ejes[0, 0].set_ylabel(r"Speedup $S(N,p)$")
    ejes[1, 0].set_ylabel(r"Eficiencia $E(N,p)$")

    asas_n, etiquetas_n = ejes[0, 0].get_legend_handles_labels()
    figura.legend(
        asas_n,
        etiquetas_n,
        loc="lower center",
        ncol=len(TAMANOS) + 1,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figura.tight_layout(rect=(0, 0.065, 1, 1))
    _guardar_figura(figura, directorio, "speedup_eficiencia")


def imprimir_resultados(resumenes: list[Resumen], modelo: ModeloTeorico) -> None:
    """Imprime coeficientes y configuraciones óptimas.

    Args:
        resumenes: Estadísticos por configuración.
        modelo: Modelo teórico ajustado.
    """
    print("Modelo teórico ajustado")
    print(f"  a (latencia): {modelo.coeficiente_latencia:.10e}")
    print(f"  b (ancho de banda): {modelo.coeficiente_ancho_banda:.10e}")
    print(f"  c (cómputo): {modelo.coeficiente_computo:.10e}")
    print(f"  R² de ejecución: {modelo.r2_ejecucion:.6f}")
    print("Configuraciones de menor mediana de ejecución")
    for tamano in TAMANOS:
        mejor = min(_grupo_por_tamano(resumenes, tamano), key=lambda resumen: resumen.ejecucion_mediana)
        print(
            f"  N={tamano}: p={mejor.procesos}, "
            f"T={mejor.ejecucion_mediana:.10f} s, "
            f"S={mejor.speedup:.4f}, E={mejor.eficiencia:.4f}"
        )


def main() -> None:
    """Ejecuta el análisis reproducible completo."""
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("csv", type=Path, help="Ruta de results_rank.csv")
    analizador.add_argument("--resumen", type=Path, required=True, help="CSV agregado de salida")
    analizador.add_argument("--figuras", type=Path, required=True, help="Directorio de figuras")
    argumentos = analizador.parse_args()

    mediciones = cargar_mediciones(argumentos.csv)
    resumenes = resumir_mediciones(mediciones)
    modelo = ajustar_modelo_teorico(resumenes)
    guardar_resumen(resumenes, argumentos.resumen)
    graficar_tiempos(resumenes, modelo, argumentos.figuras)
    graficar_speedup_eficiencia(resumenes, modelo, argumentos.figuras)
    imprimir_resultados(resumenes, modelo)


if __name__ == "__main__":
    main()
