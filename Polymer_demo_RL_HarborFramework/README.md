# Polyurethane–urea FTIR and tensile-strength demo

This project analyzes `synthetic_polyurethane_urea.csv`, summarizes FTIR peak positions and mechanical properties, and predicts tensile strength with a held-out test set. The regression script uses only the Python standard library, so no package installation is needed.

## Run

From this project directory, run:

```powershell
python .\analyze_polyurethane_urea.py
```

The defaults use an 80/20 train/test split with seed 42 and ridge penalty 1.0. To choose a different input, output folder, split, seed, or penalty:

```powershell
python .\analyze_polyurethane_urea.py --input .\synthetic_polyurethane_urea.csv --output-dir .\outputs --test-size 0.2 --seed 42 --alpha 1.0
```

## Outputs

- `outputs/predictions.csv`: held-out sample IDs, observed tensile strengths, predicted strengths, and prediction errors.
- `outputs/evaluation_metrics.json`: split details, model configuration, held-out MAE/RMSE/R², and descriptive statistics for FTIR measurements and mechanical properties.

The model predicts tensile strength from formulation and FTIR variables. Elongation at break and Young's modulus are described in the report but excluded from model inputs to avoid using other measured mechanical outcomes to predict the target.
