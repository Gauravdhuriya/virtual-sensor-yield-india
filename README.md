# Indian district experiments: step-by-step

Tests whether physics-inspired virtual sensors (VSSM, VSST, VSCS) predict district crop yields better than
raw weather, on ICRISAT yields + NASA POWER daily weather + SoilGrids pH.

## A. What you do

1. Put these in ONE folder:
   - all the `.py` files from this package
   - your ICRISAT file (any name starting with `ICRISAT` and ending `.csv`)
   - the `india_data/` folder made by `fetch_india_data.py` (it must contain `india_districts.csv` and `india_weather_part*.npz`)
2. Install: `pip install -r requirements.txt`
3. Smoke test (2-5 min): `python run_all.py --quick` then `python make_report.py --quick`
   Open `report_quick/results.md`. If it has tables, everything works. These quick numbers are NOT results.
4. Full run: `python run_all.py`
   Order: XGBoost comparison -> ablation -> sensitivity -> Random Forest comparison (slowest, last).
   You can stop it any time and run the same command again; finished parts are skipped.
   Rough time: 30-60 min for the XGBoost stages, then 1-3 hours for Random Forest on a 4-8 core laptop.
   `python run_all.py --skip-rf` leaves Random Forest out.
5. Optional neural baselines (LSTM, Transformer, physics-guided network; 1-3 hours on CPU): `python run_nn.py`
6. Build the report: `python make_report.py`
7. Send back `report/results.md` (small text file). Figures are in `report/fig_*.png`.

Do not edit numbers in `results.md`. If something looks wrong, send the file as it is.

## B. What the code does, step by step

**1. Yields.** Six crops: rice, maize, kharif sorghum, pearl millet (kharif, 1 Jun - 31 Oct) and wheat, chickpea
(rabi, 1 Nov - 31 Mar of the next year). A district-crop-year is used if yield > 0, crop area >= 1,000 ha, yield is
not above 5x the crop median, and the three previous years are also valid. Yield is in t/ha.

**2. Weather.** Daily rainfall (mm), air temperature (C), wind speed (m/s), solar radiation (MJ/m2/day) and relative
humidity (%) for 1984-2017 at each district headquarters. Gaps are filled with that district's day-of-year mean.
Soil pH (0-30 cm) is one value per district; missing values take the state median.

**3. Virtual sensors, run daily and continuously from 1 Jan 1984 for every district (Eq. 4-7 as in the paper):**
- ET(t) = lam * T/45 + (1 - lam) * wind/10, each term clipped to [0, 1]
- VSSM(t) = clip(VSSM(t-1) + alpha * rain(t) - beta * ET(t), 0, 1)
- VSST(t) = gamma * VSST(t-1) + (1 - gamma) * (T(t) + delta * solar(t)), gamma = exp(-1/tau)
- VSCS(t) = omega1 * (1 - VSSM) + omega2 * VSST/45 + omega3 * |pH - 6.5| / 6.5

Parameters are fixed from physical reasoning, never tuned on yield:
alpha = 0.8/150 (80% infiltration into a 150 mm root-zone store), beta = 10/150 (peak demand 10 mm/day),
lam = 0.7, tau = 5 days (gamma = 0.82), delta = 0.1 C per MJ/m2/day, omega = (0.5, 0.3, 0.2), VSSM(0) = 0.5.

**4. Season features.** Each growing season is cut into three equal sub-periods. Every daily series is averaged
within each sub-period. Raw arm: 5 weather variables x 3 sub-periods + soil pH. Latent arm: 3 sensors x 3 sub-periods.
Raw + latent: both. Every arm also gets crop identity (one-hot). Features are standardised with training statistics.

**5. Temporal memory.** Season features of years t, t-1 and t-2 are used (k = 2).

**6. Three targets.**
- Yield level, climate only: predict yield (t/ha) from the climate representation.
- Yield level, with yield memory: also give the standardised yields of years t-1 and t-2 (the paper's H_t, Eq. 11).
- Yield anomaly, climate only: predict log(yield / mean of the previous three years). This removes district level
  and trend, so it is the cleanest test of whether a representation captures weather effects.

**7. Five hold-outs.**
- Temporal: train on years up to 2007, test on 2008 onward.
- Cross-district: 5 folds; unseen districts, with other districts of the same state in training.
- Multi-state: 5 folds of whole states held out.
- Multi-crop: leave one crop out (6 folds).
- Combined: held-out states x held-out crops (5 folds); training contains neither.

**8. Models.** Random Forest (100 trees, defaults), XGBoost (300 trees, lr 0.05, depth 6, subsample 0.8),
LSTM (32 units), Transformer encoder (2 layers, 4 heads, d = 32), and a physics-guided network in which Eq. 4-7 are
a differentiable layer and alpha, beta, lam, gamma, delta, omega are learned end to end. No hyperparameter tuning.

**9. Repeats and statistics.** XGBoost 5 seeds, Random Forest and neural models 3 seeds; mean ± SD.
Significance: RMSE is computed per district (per state for multi-state, per crop for multi-crop); raw and latent are
compared with a paired Wilcoxon signed-rank test and a bootstrap 95% CI of the mean difference.

**10. Ablation.** Each sensor alone, each pair, all three; memory window k = 0..3; no climate at all;
yield memory only. Run on the temporal and multi-state hold-outs.

**11. Sensitivity.** alpha, beta, lam, tau, delta, omega1 each changed by -50%, -20%, +20%, +50% with the others
fixed; sensors recomputed and the model retrained each time.

## C. Files
- `adt_india.py`   data, virtual sensors, features, hold-outs, models, statistics
- `run_all.py`     tree-model experiments (cached in `out/`)
- `run_nn.py`      neural baselines (optional)
- `make_report.py` tables and figures from `out/`
- `fetch_india_data.py`  the weather / pH download script

The pipeline was tested end to end on synthetic weather in the same file format; it has not yet been run on the real
weather files.
