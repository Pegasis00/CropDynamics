# model_weights/

Drop your already-trained artifacts from the notebook here. The app does
**not** train anything at runtime — `ModelRegistry` just loads whatever it
finds in this folder at startup.

Expected files (from the notebook's Model Building / export cells):

```
model_smooth_RandomForest.joblib
model_smooth_XGBoost.joblib
model_smooth_LightGBM.joblib
model_smooth_SVM.joblib
model_raw_RandomForest.joblib
model_raw_XGBoost.joblib
model_raw_LightGBM.joblib
model_raw_SVM.joblib
label_encoder.joblib
```

Only the artifact matching `configs/model.yaml`'s `selected_algorithm` /
`selected_feature_set` is required for the app to start and serve
predictions; the rest are optional (only used if you later enable the
opt-in agreement-check module across all 8 models).

If you retrain later with `backend/models/trainer.py`, it writes its
output straight into this folder.

When you swap in a model trained with wheat, replace `label_encoder.joblib`
from the same training run so the API can return the new `Wheat` class.
