# Integrated multi-period LP for production mix and raw-material procurement in an NPK fertilizer plant

Code for the article

> E. M. Waluyo and Y. P. Astuti, "Integrated Multi-Period Linear Programming for Production Mix and Raw-Material
> Procurement in an NPK Fertilizer Plant," *ZERO: Jurnal Sains, Matematika dan Terapan*.

The model is a multi-period linear program (equations 4–9 of the paper) that chooses the monthly production mix of six
products and the purchases and stocks of seven raw materials. The code computes the scenarios S0–S3, the shadow price of
granulation capacity with a uniqueness check and month-by-month ranging, the sensitivity and robustness analyses, the
rolling-horizon tests with naive, moving-average, drift and ARIMA forecasts, the threshold purchasing rule, and the
bootstrap simulation.

## Data

The inputs of the paper are confidential procurement and production records of the company (referred to as PT. X) and
licensed CRU Fertilizer Week prices. **They are not included and cannot be shared.**

To make the code runnable, `synthetic_data/` contains a **fully synthetic** dataset in the same file layout, written by
`code/make_synthetic_data.py`. Nothing in it is derived from company or CRU data: price levels are round numbers chosen
by hand, price paths are correlated random series, and production, capacity and tank figures are invented. The synthetic
run reproduces the **structure** of Tables 3 and 5 of the paper, not their values.

| File | Content |
|---|---|
| `synthetic_data/company_config.py` | invented production by year, capacity, tank sizes (same keys as `code/company_config_TEMPLATE.py`) |
| `synthetic_data/cru_monthly_selected.csv` | synthetic monthly reference prices, 2013–2025 |
| `synthetic_data/raw/purchases_monthly_companywide.csv` | synthetic monthly purchases (quantity, value, price), 2015–2025 |

## Running the demo

```bash
pip install -r requirements.txt
python code/run_demo.py            # about 1.5 minutes; add --figures and/or --bootstrap for Figures 1-6 and the simulation
```

`run_demo.py` creates `demo_run/`, writes the synthetic inputs, and runs the pipeline

1. `prep_prices_v3.py` – price preprocessing (Section 2.1): basis correction, anomaly rule, calibration (eqs. 1–2), filling
   of months without a realized price, product values (eq. 3)
2. `run_main3.py` – scenarios S0–S3 for 2021–2024, duals and ranging, sensitivity, robustness variants, dynamic analysis
3. `run_rolling3.py` – rolling-horizon tests, continuous horizon and lead time, forecast errors, threshold rule (eq. 13)
4. `make_tables.py` – Tables 3 and 5 (`Table3.csv`, `Table5.csv`) and the threshold rule for all safety margins
   (`Table5_theta.csv`)

The outputs of the synthetic run are in `demo_output/` for comparison.

## Running with real data

Copy `code/company_config_TEMPLATE.py` to `company_config.py`, fill in the values, extract the monthly purchases with
`00c_extract_purchases.py`, place the CRU selection in `cru_monthly_selected.csv`, and run the scripts above in the same
order (then `run_bootstrap3.py`, `ranging_detail.py`, `figures3.py`).

## Files

| Script | Purpose |
|---|---|
| `model3.py` | the LP and its variants (purchase bounds, minimum lots as MILP, annual commitment, quarter-end stock, fixed purchases) |
| `common.py`, `setup3.py` | sets, consumption ratios and capacity factors (Table 1); year-specific inputs |
| `prices.py` | price series and product values |
| `00c_extract_purchases.py` | extraction of monthly purchases from the company workbooks (not runnable without them) |
| `run_bootstrap3.py`, `ranging_detail.py`, `figures3.py` | bootstrap simulation, ranging detail, figures |
| `make_synthetic_data.py`, `run_demo.py`, `make_tables.py` | synthetic data, demo runner, Tables 3 and 5 |

Tested with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18.1 (HiGHS 1.12.0), statsmodels 0.15.0, matplotlib 3.11.

## Licence

MIT, see `LICENSE`.
