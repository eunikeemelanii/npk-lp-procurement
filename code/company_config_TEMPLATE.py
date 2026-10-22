"""Template of the company-specific inputs. The values are confidential records of PT. X and are not distributed.
Copy this file to company_config.py and fill in the values to run the code."""
DATA_DIR = None             # folder with the procurement-realization workbooks
PURCHASE_WORKBOOKS = None   # [(file name, [sheet names = years]), ...]; see 00c_extract_purchases.py for the layout
R_BY_YEAR = None            # {year: {product: realized annual production (t)}} for 2021-2024 (2024 totals in Table 1)
K_DESIGN = None             # installed capacity of the NPK granulation lines (t T15-equivalent per year)
TANK = None                 # {'Amonia': t, 'Asam Fosfat': t, 'Asam Sulfat': t} tank capacities
OWN_AMMONIA_CAPACITY = None
OWN_SULFURIC_CAPACITY_EXCL_JV = None
SAFE_DAYS = 7
KCL_OTHER = None            # production outside the model that uses KCl and the corresponding ratios
