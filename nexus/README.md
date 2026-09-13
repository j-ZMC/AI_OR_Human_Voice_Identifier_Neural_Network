# Modelos de Nexus

El entrenamiento organiza cinco modelos en carpetas separadas bajo `models/`:

- `models/time_predict/`: modelo basado en 5 estadisticas de `reaction_time`.
- `models/sustantive_predict/`: modelo basado en 24 caracteristicas de toda la llamada.
- `models/best_model/`: copia del modelo con mejor AUC en `val`.
- `models/legacy_model/`: pipeline `model.joblib` conservado para compatibilidad.
- `models/random_forest/`: bosque final que combina los modelos de tiempos,
    caracteristicas de llamada, Mini-AST y Wav2Vec2.

Cada carpeta contiene el modelo y su `normalization.json`. Los checkpoints
`.pt` tambien incluyen esa informacion para que el modelo sea autocontenido.
`models/registry.json` mantiene el contrato de nombres, columnas y rutas para
agregar despues un nodo de decision tree.

## Entrenamiento

Desde la carpeta raiz del workspace:

```powershell
c:/Users/jesus/anaconda3/envs/faiss_env/python.exe nexus/train.py
```

Por defecto usa `nexus/manifest.csv` y busca los JSON en
`hackmty26/turns`. Se pueden cambiar las rutas:

```powershell
c:/Users/jesus/anaconda3/envs/faiss_env/python.exe nexus/train.py --manifest ruta/manifest.csv --turns-dir ruta/turns --output-dir nexus
```

Durante el entrenamiento se genera `nexus/acoustic_predictions.csv` con las
probabilidades de `mini_ast_best.pt` y `wav2vec2_finetuned_model_run`. Ese
cache se reutiliza en ejecuciones posteriores mientras contenga todas las
llamadas del manifiesto.

## Prediccion

El comando acepta un JSON con `{ "turns": [...] }`, una lista directa de
turnos, o un CSV. El CSV puede tener `channel,start,end`, las columnas de
caracteristicas agregadas, o una columna `reaction_time`.

```powershell
c:/Users/jesus/anaconda3/envs/faiss_env/python.exe nexus/predict.py hackmty26/turns/call_0181ce113ebe.json
c:/Users/jesus/anaconda3/envs/faiss_env/python.exe nexus/predict.py nexus/dataset.csv
```

La inferencia usa `models/random_forest/model.joblib` como nodo final. El bosque
recibe las probabilidades out-of-fold de `time_predict` y `sustantive_predict`,
mas `mini_ast_probability` y `wav2vec_probability`. Las entradas con audio
local ejecutan los dos detectores acusticos; una entrada que solo contiene
turnos no puede hacerlo y usa `0.5` como valor neutro para esas dos columnas.
Si un CSV solo contiene las 5 estadisticas de reaccion, selecciona
automaticamente `models/time_predict/model.pt` porque no puede reconstruir las
24 caracteristicas necesarias para el segundo nodo base.