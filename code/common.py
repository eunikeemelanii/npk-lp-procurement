import pandas as pd, numpy as np
PRODUCTS=['NPK 15-10-12','NPK 15-15-15','NPS','DAP','SP-36','SP-20']
MATS=['Urea','ZA','Asam Sulfat','KCl','Asam Fosfat','Amonia','Batuan Fosfat']
# consumption ratios a[m,p] (t raw material / t product), Table 1
A=pd.DataFrame({
 'NPK 15-10-12':[0.001,0.27,0.25,0.21,0.23,0.12,0],
 'NPK 15-15-15':[0.03,0.13,0.20,0.255,0.285,0.137,0],
 'NPS':[0.04,0.25,0.24,0,0.38,0.17,0],
 'DAP':[0,0.08,0.06,0,0.84,0.21,0],
 'SP-36':[0,0,0.15,0,0.40,0,0.53],
 'SP-20':[0,0,0.03,0,0.08,0,0.556],
},index=MATS)
# capacity conversion factors (Table 1): one ton of product p uses 1/kappa t of T15-equivalent capacity
KAPPA={'NPK 15-10-12':0.95,'NPK 15-15-15':1.0,'NPS':0.7,'DAP':1200/1800}
NPK_LINE=['NPK 15-10-12','NPK 15-15-15','NPS','DAP']
# realized 2024 production and installed capacity: confidential inputs in company_config.py
from company_config import R_BY_YEAR as _R, K_DESIGN
R2024 = _R[2024]
