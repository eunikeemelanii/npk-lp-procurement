"""SYNTHETIC company inputs written by make_synthetic_data.py (invented values, not company data)."""
DATA_DIR = None
PURCHASE_WORKBOOKS = None
R_BY_YEAR = {2021: {'NPK 15-10-12': 1200000, 'NPK 15-15-15': 250000, 'NPS': 60000, 'DAP': 50000, 'SP-36': 120000, 'SP-20': 0}, 2022: {'NPK 15-10-12': 1300000, 'NPK 15-15-15': 150000, 'NPS': 50000, 'DAP': 60000, 'SP-36': 80000, 'SP-20': 0}, 2023: {'NPK 15-10-12': 1250000, 'NPK 15-15-15': 100000, 'NPS': 40000, 'DAP': 70000, 'SP-36': 30000, 'SP-20': 0}, 2024: {'NPK 15-10-12': 1400000, 'NPK 15-15-15': 50000, 'NPS': 40000, 'DAP': 80000, 'SP-36': 10000, 'SP-20': 0}}
K_DESIGN = 1800000
TANK = {'Amonia': 30000, 'Asam Fosfat': 60000, 'Asam Sulfat': 40000}
OWN_AMMONIA_CAPACITY = 300000
OWN_SULFURIC_CAPACITY_EXCL_JV = 600000
SAFE_DAYS = 7
KCL_OTHER = dict(npk_granulation=100000, other_npk=20000, k2so4=10000, r_gran=0.22, r_other_low=0.22, r_other_high=0.45,
                 r_k2so4=0.84)
