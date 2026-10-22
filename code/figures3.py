"""Figures 1-6 of the revised manuscript (revision 2). Run after run_main3.py."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import PRODUCTS, MATS, KAPPA, NPK_LINE
os.makedirs('fig', exist_ok=True)
plt.rcParams['font.family'] = 'Liberation Serif'; plt.rcParams['mathtext.fontset'] = 'stix'
INK = '#1a1a1a'; INK2 = '#52514e'; FILL = '#eef3fb'; FILL2 = '#fdf1e8'; FILL3 = '#f3f3f1'; GRAY = '#b5b3ad'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; LS = ['-', '--', '-.', ':']


def box(ax, x, y, w, h, t, fc=FILL, fs=8.6):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec=INK, lw=0.9))
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, linespacing=1.2, color=INK)


def term(ax, x, y, w, h, t):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02,rounding_size=%f' % (h/2*0.95),
                                fc='white', ec=INK, lw=1.1))
    ax.text(x, y, t, ha='center', va='center', fontsize=9.1, fontweight='bold')


def arr(ax, pts, text=None, ty=0.0):
    for (a, b) in zip(pts[:-2], pts[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.9)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=9, lw=0.9, color=INK, shrinkA=0, shrinkB=0))
    if text:
        ax.text((pts[-2][0]+pts[-1][0])/2, (pts[-2][1]+pts[-1][1])/2+ty, text, fontsize=8.3, ha='center', va='center', style='italic')


# ------------------------------------------------------------------ Figure 1: research flowchart
fig, ax = plt.subplots(figsize=(6.6, 2.75)); ax.set_xlim(0, 16.3); ax.set_ylim(0.2, 5.9); ax.axis('off')
W, H, G = 2.5, 1.85, 0.42
xs = [0.72]+[1.4+G+W/2+i*(W+G) for i in range(5)]
yt, yb = 4.25, 1.35
term(ax, xs[0], yt, 1.25, 0.72, 'Start')
top = ['Data: company\nrecords and\nCRU prices',
       'Preprocessing:\nbasis, anomalies,\ngap filling,\nproduct values',
       'Multi-period LP\nformulation\n(4)–(9)',
       'Implementation\n(SciPy, HiGHS);\nS0 check',
       'Scenarios S0–S3,\n2021–2024\n(perfect\nforesight)']
for i, t in enumerate(top): box(ax, xs[i+1], yt, W, H, t, fs=8.0)
arr(ax, [(xs[0]+0.625, yt), (xs[1]-W/2, yt)])
for i in range(1, 5): arr(ax, [(xs[i]+W/2, yt), (xs[i+1]-W/2, yt)])
bot = ['Robustness to\noperational\nconstraints and\nprice series',
       'Rolling horizon\nwith forecasts;\nbootstrap\nsimulation',
       'Duals, ranging\nand sensitivity\nanalysis',
       'Discussion and\nanswers to\nRQ1–RQ3']
cols = [5, 4, 3, 2]
for c_, t in zip(cols, bot): box(ax, xs[c_], yb, W, H, t, fc=FILL2, fs=8.0)
arr(ax, [(xs[5], yt-H/2), (xs[5], yb+H/2)])
for a, b in zip(cols[:-1], cols[1:]): arr(ax, [(xs[a]-W/2, yb), (xs[b]+W/2, yb)])
term(ax, xs[1], yb, 1.4, 0.72, 'End')
arr(ax, [(xs[2]-W/2, yb), (xs[1]+0.7, yb)])
fig.savefig('fig/Figure1_flowchart.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: model structure
fig, ax = plt.subplots(figsize=(6.5, 4.1)); ax.set_xlim(0, 13.6); ax.set_ylim(0.4, 8.6); ax.axis('off')
mats = ['Urea*', 'ZA', 'Sulfuric acid', 'KCl', 'Phosphoric acid', 'Ammonia', 'Phosphate rock']
ys = [7.3, 6.3, 5.3, 4.3, 3.3, 2.3, 1.3]
ax.text(1.1, 8.15, 'Procurement\n$y_{m,t}$ at price $c_{m,t}$', ha='center', va='center', fontsize=8.8, fontweight='bold')
ax.text(4.05, 8.15, 'Raw-material inventory\n$I_{m,t} \\leq S_m$', ha='center', va='center', fontsize=8.8, fontweight='bold')
ax.text(7.65, 8.15, 'Production lines\n(capacity)', ha='center', va='center', fontsize=8.8, fontweight='bold')
ax.text(11.9, 8.15, 'Products $x_{p,t}$\nvalued at $v_{p,t}$', ha='center', va='center', fontsize=8.8, fontweight='bold')
for m, y in zip(mats, ys):
    box(ax, 1.1, y, 1.95, 0.62, m, fc=FILL3, fs=7.5 if len(m) > 12 else 8.6)
    arr(ax, [(2.075, y), (2.75, y)])
ax.add_patch(FancyBboxPatch((2.75, 0.95), 2.6, 6.7, boxstyle='round,pad=0.02,rounding_size=0.12', fc=FILL2, ec=INK, lw=0.9))
ax.text(4.05, 4.3, 'Inventory\nbalance\n\n$I_{m,t}=I_{m,t-1}$\n$+\\,y_{m,t}$\n$-\\sum_p a_{m,p}x_{p,t}$',
        ha='center', va='center', fontsize=8.8, linespacing=1.5)
box(ax, 7.65, 5.4, 3.0, 1.9, '', fs=8.6)
ax.text(7.65, 5.85, 'NPK granulation\nlines', ha='center', va='center', fontsize=8.6, linespacing=1.25)
ax.text(7.65, 4.85, '$\\sum_p x_{p,t}/\\kappa_p \\leq K_t$', ha='center', va='center', fontsize=8.6)
box(ax, 7.65, 2.1, 3.0, 1.3, 'Phosphate line\n(SP-36, SP-20)', fs=8.6)
arr(ax, [(5.35, 5.4), (6.15, 5.4)], '$a_{m,p}$', ty=0.28); arr(ax, [(5.35, 2.1), (6.15, 2.1)], '$a_{m,p}$', ty=0.28)
prods = [('NPK 15-10-12', 6.9), ('NPK 15-15-15', 6.05), ('NPS 20-20-0+13S', 5.2), ('DAP 18-46-0', 4.35), ('SP-36', 2.5), ('SP-20', 1.65)]
for n, y in prods:
    box(ax, 11.9, y, 2.7, 0.6, n, fc='#ffffff', fs=8.6)
    src = 5.4 if y > 3.5 else 2.1
    arr(ax, [(9.15, src+(y-(5.625 if y > 3.5 else 2.075))*0.35), (10.55, y)])
ax.text(11.9, 0.8, '$(1-\\delta)R_p/12 \\leq x_{p,t} \\leq (1+\\delta)R_p/12$', ha='center', fontsize=8.6)
ax.text(0.15, 0.55, '*Urea is transferred internally just in time ($I_{urea,t}=0$)', ha='left', fontsize=7.6, style='italic')
fig.savefig('fig/Figure2_model_structure.png', dpi=300, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ chart style
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold'})
c = pd.read_csv('harga_bahan_baku_bulanan.csv', index_col=[0, 1]); src = pd.read_csv('sumber_harga_bahan_baku.csv', index_col=[0, 1])
res = json.load(open('results3.json'))
dates = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in c.index])

# ------------------------------------------------------------------ Figure 3: prices
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ser = ['Asam Fosfat', 'KCl', 'Amonia', 'ZA']; lab = ['Phosphoric acid', 'KCl', 'Ammonia', 'ZA']
ax.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'), color='#f3f3f1', zorder=0, lw=0)
ax.text(pd.Timestamp('2024-07-01'), 1290, 'Case\nstudy\n2024', ha='center', va='top', fontsize=7.8, color=INK2)
xlab = dates[-1]+pd.Timedelta(days=75)
ylab = {'Phosphoric acid': 700, 'Ammonia': 470, 'KCl': 300, 'ZA': 175}
for i, (s_, l) in enumerate(zip(ser, lab)):
    ax.plot(dates, c[s_].values, color=C[i], lw=1.2, ls=LS[i], label=l)
    filled = (src[s_] != 'realized').values
    ax.plot(dates[filled], c[s_].values[filled], ls='none', marker='o', ms=2.6, mfc='white', mec=C[i], mew=0.8)
    y_end = c[s_].values[-1]; y_l = ylab[l]
    ax.plot([dates[-1]+pd.Timedelta(days=8), xlab-pd.Timedelta(days=12)], [y_end, y_l], color=C[i], lw=0.7)
    ax.text(xlab, y_l, l.replace('Phosphoric acid', 'Phosphoric\nacid'), color=INK, fontsize=8, va='center', linespacing=1.0)
ax.plot([], [], ls='none', marker='o', ms=3, mfc='white', mec=INK2, label='Filled month')
ax.set_ylabel('Price (USD/ton)'); ax.set_ylim(0, 1300)
ax.set_xlim(dates[0], dates[-1]+pd.Timedelta(days=420))
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=8, handlelength=2.4)
fig.tight_layout(); fig.savefig('fig/Figure3_prices.png', dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure 4: procurement 2024, S0 vs S3
y0 = pd.read_csv('out_y_S0_2024.csv', index_col=0); y3 = pd.read_csv('out_y_S3_2024.csv', index_col=0)
mats = ['Asam Fosfat', 'Amonia', 'ZA']; titles = ['Phosphoric acid', 'Ammonia', 'ZA']
fig, axs = plt.subplots(2, 3, figsize=(6.4, 3.9), sharex=True, gridspec_kw={'height_ratios': [1, 1.6]})
mon = np.arange(1, 13)
for j, m in enumerate(mats):
    a = axs[0, j]; a.plot(mon, c.loc[2024][m].values, color=INK2, lw=1.2, marker='o', ms=3)
    a.set_title(titles[j])
    if j == 0: a.set_ylabel('Price\n(USD/ton)')
    b = axs[1, j]
    b.bar(mon-0.2, y0[m].values/1000, width=0.38, color=GRAY, label='S0 (just-in-time)')
    b.bar(mon+0.2, y3[m].values/1000, width=0.38, color=C[0], label='S3 (optimal)')
    b.set_xticks(mon); b.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
    if j == 0: b.set_ylabel('Procurement\n(thousand tons)')
h_, l_ = axs[1, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(h_pad=0.6, w_pad=0.8, rect=(0, 0.06, 1, 1)); fig.savefig('fig/Figure4_procurement.png', dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure 5: dynamic (counterfactual) analysis
dyn = pd.read_csv('out_dynamic_2015_2025.csv')
dd = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in zip(dyn.Tahun, dyn.Bulan)])
fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.3), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
a = axs[0]
ends = {}
for i, p in enumerate(NPK_LINE):
    s_ = (dyn['m_'+p]*KAPPA[p]).rolling(3, min_periods=1).mean()
    a.plot(dd, s_, color=C[i], lw=1.3, ls=LS[i], label=p); ends[p] = s_.values[-1]
# separate right-hand labels by at least 22 USD
order = sorted(ends, key=lambda k: ends[k]); pos_ = {}
prev = -1e9
for p in order:
    y_ = max(ends[p], prev+22); pos_[p] = y_; prev = y_
for i, p in enumerate(NPK_LINE):
    xe = dd[-1]+pd.Timedelta(days=60)
    a.plot([dd[-1]+pd.Timedelta(days=8), xe-pd.Timedelta(days=10)], [ends[p], pos_[p]], color=C[i], lw=0.7)
    a.text(xe, pos_[p], p.replace('NPK ', ''), fontsize=7.8, va='center', color=INK)
a.axhline(0, color='#8a8984', lw=0.7)
a.set_ylabel('Raw-material margin\n(USD/ton T15-equivalent)')
a.set_title('(a) Margin per unit of NPK line capacity (3-month moving average)', loc='left', fontweight='normal', fontsize=9)
a.legend(ncol=4, fontsize=7.8, loc='upper left', handlelength=2.6)
a.set_xlim(dd[0], dd[-1]+pd.Timedelta(days=380))
b = axs[1]
b.plot(dd, dyn.cap_dual, color=C[0], lw=1.3)
b.fill_between(dd, 0, dyn.cap_dual, color=C[0], alpha=0.12, lw=0)
b.set_ylabel('Shadow price\n(USD/ton T15-equivalent)')
b.set_title('(b) Shadow price of NPK line capacity', loc='left', fontweight='normal', fontsize=9)
fig.tight_layout(h_pad=0.5); fig.savefig('fig/Figure5_dynamic.png', dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure 6: sensitivity
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.55), gridspec_kw={'width_ratios': [1.25, 1, 1]})
p10 = res['price10']; order = sorted(p10, key=lambda k: p10[k]['dZpct'])
names = {'Asam Fosfat': 'Phosphoric acid', 'KCl': 'KCl', 'Amonia': 'Ammonia', 'ZA': 'ZA', 'Asam Sulfat': 'Sulfuric acid',
         'Urea': 'Urea', 'Batuan Fosfat': 'Phosphate rock'}
vals = [p10[k]['dZpct'] for k in order]
a = axs[0]; a.set_axisbelow(True); yy = np.arange(len(order))[::-1]
a.barh(yy, vals, color=C[1], height=0.6)
a.set_yticks(yy); a.set_yticklabels([names[k] for k in order], fontsize=7.8)
for y_, v_ in zip(yy, vals):
    t_ = (f'{v_:.1f}' if abs(v_) >= 0.1 else f'{v_:.2f}').replace('-', '−')
    a.text(v_-0.3, y_, t_, va='center', ha='right', fontsize=7.4, color=INK)
a.set_xlim(-15.5, 0.5); a.set_xticks([-12, -8, -4, 0]); a.set_xticklabels(['−12', '−8', '−4', '0'])
a.set_xlabel('Change in Z (%)', fontsize=8.2); a.grid(axis='y', visible=False)
a.set_title('(a) +10% price, all months', loc='left', fontweight='normal', fontsize=8.6)
b = axs[1]; vo = res['volatility']; ks = sorted(vo, key=float)
b.plot([float(k) for k in ks], [vo[k]['S1'] for k in ks], color=C[0], lw=1.3, marker='o', ms=3.5, label='S1 (timing)')
b.plot([float(k) for k in ks], [vo[k]['S3'] for k in ks], color=C[1], lw=1.3, ls='--', marker='s', ms=3.5, label='S3 (integrated)')
b.set_xlabel('Volatility scale $k$', fontsize=8.2); b.set_ylabel('Gain vs S0 (%)', fontsize=8.2)
b.set_title('(b) Price volatility', loc='left', fontweight='normal', fontsize=8.6); b.legend(fontsize=7.4, loc='upper left')
b.set_ylim(0, 7.5)
d = axs[2]; sd = res['sens_delta']; ks = sorted(sd, key=float)
d.plot([float(k)*100 for k in ks], [sd[k]['S2'] for k in ks], color=C[2], lw=1.3, marker='o', ms=3.5, label='S2 (mix)')
d.plot([float(k)*100 for k in ks], [sd[k]['S3'] for k in ks], color=C[1], lw=1.3, ls='--', marker='s', ms=3.5, label='S3 (integrated)')
d.set_xlabel('Mix band δ (%)', fontsize=8.2)
d.set_title('(c) Mix flexibility', loc='left', fontweight='normal', fontsize=8.6); d.legend(fontsize=7.4, loc='upper left')
d.set_ylim(0, 7.5)
fig.tight_layout(w_pad=0.8); fig.savefig('fig/Figure6_sensitivity.png', dpi=300); plt.close(fig)
print('ok')
