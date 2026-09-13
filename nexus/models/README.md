# Registro de modelos

Cada subcarpeta tiene un artefacto de modelo y su `normalization.json`:

- `time_predict`: modelo de tiempos de reaccion.
- `sustantive_predict`: modelo con caracteristicas completas de la llamada.
- `best_model`: alias del mejor modelo de validacion.
- `legacy_model`: pipeline scikit-learn anterior.
- `mini_ast_predict`: detector acustico basado en el Mel-espectrograma.
- `wav2vec_predict`: detector acustico basado en Wav2Vec2.
- `random_forest`: nodo final que combina los cuatro modelos anteriores.

`registry.json` es el indice estable para construir un grafo posteriormente.
El nodo `decision_tree` queda declarado como pendiente. El `random_forest` ya
se entrena y se usa en la prediccion por defecto.