# NIDS con GNNs (Proyectos IV)

Repositorio del trabajo final de la asignatura **Proyectos IV**, sobre **detección de intrusiones en red** usando **grafos**.  
La idea es convertir flujos de red (CSV) en grafos por ventanas de tiempo y entrenar un **GATv2** como baseline.


## Que tenemos
- **Dataset**: Improved CSE-CIC-IDS2018 (flows CSV) descargable desde [Aquí](https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/).
- **Pipeline**: limpieza → grafos → entrenamiento → plots.
- **Código core**: en `nids/` (preprocessing, graphs, training, api).
- **Scripts**: en la raíz (`load.py`, `build_graphs.py`, `train_gatv2.py`, `plot_training.py`).


## Requisitos rápidos
- Python 3.10+
- Dependencias completas:
  ```bash
  pip install -r requirements.txt
  ```
- Solo ingesta/limpieza:
  ```bash
  pip install pandas numpy scikit-learn pyarrow
  ```


## Estructura mínima de datos
Coloca los CSV aquí:
```
data/flows/
    Wednesday-14-02-2018.csv
    Thursday-15-02-2018.csv
    ...
```


## Pipeline básico (recomendado)
1) **Limpieza + features**
```bash
python load.py
```
Salidas:
- `data/processed/feature_cols.json`
- `data/processed/cleaned_parquet/*.parquet` (si `PROCESS_MODE="per_file_parquet"`)

2) **Construir grafos**
```bash
python build_graphs.py
```
Salida:
- `data/graphs/*.pt`

3) **Entrenar baseline (GATv2)**
```bash
python train_gatv2.py
```
Salidas:
- `data/processed/training/train_history.json`
- `data/processed/training/best_model.pt`
- `data/processed/training/feature_stats.json`
- `data/processed/training/pos_weight.json`

4) **Plots**
```bash
python plot_training.py --input data/processed/training/train_history.json
```
Salida:
- `data/processed/training/plots/`


## Configuración clave (rápido)
Los parámetros están **al inicio** de cada script:
- `load.py`: `FLOW_DIR`, `PROCESS_MODE`, `DEV_MODE`, `ATTEMPTED_POLICY`, etc.
- `build_graphs.py`: `WINDOW_SIZE_SECONDS`, `MIN_NODES_PER_GRAPH`, `MAX_NODES_PER_GRAPH`, `K_IP`, `K_TIME`.
- `train_gatv2.py`: `TRAIN_CAPTURES`, `VAL_CAPTURES`, `TEST_CAPTURES`, `EPOCHS`, `DEVICE`.

Para iterar rápido:
```python
DEV_MODE = True
MAX_ROWS_PER_FILE = 250_000
```


## Etiquetas
- `label_bin`: `0 = BENIGN`, `1 = ATTACK`.
- `Attempted Category` se controla con `ATTEMPTED_POLICY`:
  - `ignore`, `drop`, `relabel_benign` (default).



## API  (FastAPI)
Si ya tienes modelo entrenado:
```bash
uvicorn nids.api.app:app --reload
```
Endpoints:
- `GET /health`
- `POST /predict/graph` (sube un `.pt`)
- `POST /predict/csv` (streaming SSE)


## Tests (opcionales)
Los tests del backend necesitan los ficheros entrenados:
- `data/processed/feature_cols.json`
- `data/processed/training/best_model.pt`

```bash
pytest -q
```


## Notas finales
Esto es un proyecto académico en progreso. El código prioriza **claridad y reproducibilidad** sobre optimización extrema.  
Si algo no está claro, lo normal es revisar primero los scripts `load.py`, `build_graphs.py` y `train_gatv2.py`.
