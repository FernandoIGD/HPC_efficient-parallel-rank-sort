# Ordenamiento paralelo por ranking

Implementación en C++/MPI del algoritmo de ordenamiento por ranking y software en Python para analizar su performance. El experimento compara distintos tamaños de entrada y números de procesos, y genera gráficos de tiempos, *speedup* y eficiencia.

## Archivos principales

- `main.cpp`: implementación paralela con MPI.
- `ranking_job.sh`: ejecución de mediciones mediante SLURM.
- `analysis/parse_results.py`: conversión de los logs a CSV.
- `analysis/analizar_ranking.py`: resumen estadístico, ajuste NNLS y generación de figuras.
- `results_rank.csv`: mediciones utilizadas por el análisis.

## Compilación y ejecución en Khipu

```bash
module load gnu12/12.4.0 openmpi4/4.1.6
mpic++ -O3 -std=c++17 main.cpp -o ranking_sort
mkdir -p logs
sbatch --ntasks=9 --nodelist=n003 ranking_job.sh 20
```

El `20` indica el número de repeticiones que se ejecutará para cada configuración. El número de procesos debe ser un cuadrado perfecto. El script también valida que cada configuración respete el *buffer* fijo de 10 000 caracteres del código de referencia.

## Análisis de resultados

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis/analizar_ranking.py results_rank.csv \
  --resumen results/resumen_rank.csv \
  --figuras figures
```

El programa genera `tiempos_comparacion.png` y `speedup_eficiencia.png`. El CSV de entrada debe contener las columnas `processes`, `N`, `execution_s`, `computation_s` y `communication_s`.

> **Nota:** el tiempo denominado comunicación se calcula como un residuo del tiempo total; por ello también incluye sincronización y otros costos no clasificados como cómputo.
