import pandas as pd
import numpy as np
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
import sys
from pathlib import Path
# ==========================================
# 1. SETUP & STYLE (Nature Human Behavior)
# ==========================================
def set_figure_style():
    """Sets plot style to Nature Human Behavior standards (sans-serif, clean)."""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

set_figure_style()

#  Color Palette
colors = {
    'blue': '#377eb8',    # Learning
    'orange': '#ff7f00',  # Complexity Increase
    'green': '#4daf4a',   # Memory
    'purple': '#984ea3',  # Exploration / Bias
    'grey': '#999999',    # Other
    'red': '#e41a1c',      # Perseveration
    'yellow': '#ffff33',   # Valuation
    'brown': '#a65628'     # Control
}

cat_colors = {
    'Learning': colors['blue'], 
    'Perseveration': colors['red'], 
    'Memory': colors['green'], 
    'Exploration': colors['purple'], 
    'Other': colors['grey'],
    'Valuation': colors['yellow'],
    'Control': colors['brown']

}

# ==========================================
# 2. DATA PROCESSING PIPELINE
# ==========================================
def robust_parse_mech(x):
    """Parses string representations of mechanisms into lists."""
    try:
        if isinstance(x, list): return x
        return ast.literal_eval(x)
    except:
        try:
            # Fallback for messy strings
            content = x.strip("[]").replace("'", "").replace("\n", " ").replace('"', '')
            items = [s.strip() for s in content.split(",") if s.strip()] if "," in content else [s.strip() for s in content.split() if s.strip()]
            return items
        except: return []

def parse_params(row):
    """Extracts parameter values (LR, Beta, Stickiness) from fitted_parameters."""
    try:
        val_str = re.sub(r'np\.float64\((.*?)\)', r'\1', row['fitted_parameters'])
        values = ast.literal_eval(val_str)
        name_str = row['parsed_params']
        # Handle space-separated names if comma is missing
        if ' ' in name_str and ',' not in name_str:
            names = name_str.strip("[]").replace("'", "").replace('"', '').split()
        else:
            names = ast.literal_eval(name_str)
        
        params = {}
        if len(names) == len(values):
            for n, v in zip(names, values):
                k = n.lower()
                if any(x in k for x in ['lr_pos', 'alpha_pos']): params['lr_pos'] = v
                elif any(x in k for x in ['lr_neg', 'alpha_neg']): params['lr_neg'] = v
                elif any(x in k for x in ['lr', 'learning']) and 'pos' not in k and 'neg' not in k:
                    params['lr_pos'] = v; params['lr_neg'] = v
                if any(x in k for x in ['stick', 'pers']): params['stickiness'] = v
                if 'w' == k: params['w'] = v
        return params
    except: return {}

# def categorize(m):
#     """Maps mechanism names to 4 broad categories using comprehensive keywords."""
#     if not isinstance(m, str): return 'Other'
#     m = m.lower()
    
#     # 1. Perseveration (Habits)
#     if any(x in m for x in ['stick', 'pers', 'habit']): 
#         return 'Perseveration'
    
#     # 2. Memory (Decay, Forgetting, Traces)
#     # Includes 'trace' for Eligibility Traces
#     if any(x in m for x in ['decay', 'forget', 'mem', 'trace']): 
#         return 'Memory'
    
#     # 3. Learning (Updates, Rates, Risk, Alien)
#     if any(x in m for x in ['learn', 'updat', 'trans', 'risk', 'alien', 'delta']): 
#         return 'Learning'
    
#     # 4. Exploration (Bias, Temperature, Noise)
#     if any(x in m for x in ['bias', 'explor', 'temp', 'noise']): 
#         return 'Exploration/Bias'
    
#     return 'Other'
def categorize(mech_name):
    m = str(mech_name).lower()
    
    # 1. Control
    # if any(k in m for k in ['independent-mb-strength', 'control', 'arbitration']): 
    #     return 'Control'
    
    # 2. Perseveration (Habits, stickiness) - Keep high priority
    if any(k in m for k in ['stick', 'pers', 'habit', 'stay', 'rigidity']): 
        return 'Perseveration'
    
    # 3. Memory (Decay, forgetting) - MOVE UP to catch 'value-decay'
    if any(k in m for k in ['decay', 'forget', 'mem', 'trace']): 
        return 'Memory'
    
    # 4. Learning (Update rules) - MOVE UP to catch 'risk-sensitive-learning'
    if any(k in m for k in ['learn', 'updat', 'trans', 'risk', 'alien', 'delta', 'rate', 'plasticity', 'precision''valu', 'loss', 'util', 'reward', 'sens', 'independent-mb-strength', 'control', 'arbitration']): 
        return 'Learning'
    
    # # 5. Valuation (Subjective perception) - Move down so it only catches pure valuation items
    # if any(k in m for k in ['valu', 'loss', 'util', 'reward', 'sens']): 
    #     return 'Valuation'
    
    # 6. Exploration
    if any(k in m for k in ['bias', 'explor', 'temp', 'noise', 'conf', 'mixture']): 
        return 'Exploration'
    
    return 'Other'
    
# --- Load & Process ---
project_root = Path(__file__).resolve().parents[2]
df = pd.read_csv(project_root / 'results' / 'two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual' / 'gecco_baseline_comparison.csv')

# Baseline = group GeCCo model instead of the hybrid MB/MF model.
# Per-participant BICs of the group model evaluated on the test split
# (participants [14:], same split as the individual GeCCo run).
import json
group_dir = project_root / 'results' / 'two_step_psychiatry_group_function_ocibalanced_maxsetting'
group_bics = json.load(open(group_dir / 'bics' / 'best_bic_on_test_run0.json'))['individual_BIC']
group_params = pd.read_csv(group_dir / 'parameters' / 'best_params_on_test_run0.csv').columns.tolist()
test_ids = sorted(df['participant'].unique())
assert len(test_ids) == len(group_bics), "participant/test-split mismatch with group BICs"
df['baseline_bic'] = df['participant'].map(dict(zip(test_ids, group_bics)))
df['baseline_params'] = str(group_params)

df_clean = df.drop_duplicates(subset=['participant', 'model_type']).copy()
df_clean['bic_improvement'] = df_clean['baseline_bic'] - df_clean['best_model_bic']

# Apply Parsing
# NOTE: unique_mechanisms were extracted by the LLM relative to the HYBRID
# baseline (compare_gecco_baseline.py), not relative to the group GeCCo model.
df_clean['mechanisms'] = df_clean['unique_mechanisms'].apply(robust_parse_mech)
params_list = df_clean.apply(parse_params, axis=1).tolist()
df_full = pd.concat([df_clean.reset_index(drop=True), pd.DataFrame(params_list)], axis=1)

# Impute 'w' (Model-Based Weight)
def impute_w(row):
    if pd.notnull(row.get('w')): return row['w']
    m = row['model_type'].lower()
    if 'model-based' in m and 'model-free' not in m: return 1.0
    if 'model-free' in m and 'model-based' not in m: return 0.0
    return np.nan
df_full['w_imputed'] = df_full.apply(impute_w, axis=1)

# Calc Complexity Difference
df_full['n_base'] = df_full['baseline_params'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
df_full['n_gecco'] = df_full['num_params']
df_full['param_diff'] = df_full['n_gecco'] - df_full['n_base']

# Load OCI Data (if available)
try:
    df_oci = pd.read_csv(project_root / 'data' / 'two_step_gillan_2016_ocibalanced.csv')
    df_traits = df_oci.groupby('participant')[['oci']].first().reset_index()
    df_merged = pd.merge(df_full, df_traits, on='participant', how='inner')
    df_merged['lr_asymmetry'] = df_merged['lr_pos'] - df_merged['lr_neg']
except FileNotFoundError:
    print("OCI file not found. Skipping OCI-related plots.")
    df_merged = df_full

# Explode for Mechanism-Level Analysis (One row per mechanism)
df_exploded = df_full.explode('mechanisms')
df_exploded['Category'] = df_exploded['mechanisms'].apply(categorize)

# ==========================================
# 3. FIGURE GENERATION
# ==========================================

# --- Fig 1: BIC Improvement (Bar) ---
plt.figure(figsize=(10, 6))
df_sorted = df_full.sort_values('bic_improvement', ascending=False)
sns.barplot(data=df_sorted, x='participant', y='bic_improvement', color=colors['blue'], order=df_sorted['participant'])
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('$\\|Delta$BIC (Group GeCCo - Individual GeCCo)'); plt.xlabel('Participant ID')
plt.title('Model Fit Improvement over Group GeCCo')
plt.savefig('Fig1_BIC_Improvement_vs_groupgecco.png')

# --- Fig 2: Individual Mechanism Impact (Bar) ---
mech_stats = df_exploded.groupby(['mechanisms', 'Category'])['bic_improvement'].mean().reset_index().sort_values('bic_improvement', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(data=mech_stats, y='mechanisms', x='bic_improvement', hue='Category', dodge=False, palette=cat_colors, edgecolor='black', linewidth=0.5)
plt.xlabel('Average $\\|Delta$BIC'); plt.ylabel('')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
plt.tight_layout(); plt.savefig('Fig2_Mechanisms_vs_groupgecco.png')

# --- Fig 3: Parameter Distributions (Box+Strip) ---
plot_cols = ['lr_pos', 'lr_neg', 'w_imputed', 'stickiness']
df_melt = df_full.melt(id_vars='participant', value_vars=[c for c in plot_cols if c in df_full.columns], var_name='Parameter', value_name='Value')
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melt, x='Parameter', y='Value', color='white', showfliers=False, width=0.5)
sns.stripplot(data=df_melt, x='Parameter', y='Value', color='black', alpha=0.4, jitter=0.2, size=5)
plt.title('Parameter Estimates')
plt.savefig('Fig3_Parameters_vs_groupgecco.png')

# --- Fig 4: Complexity Difference (Bar) ---
df_ord = df_full.sort_values('participant')
bar_cols = [colors['blue'] if x < 0 else colors['orange'] if x > 0 else colors['grey'] for x in df_ord['param_diff']]
plt.figure(figsize=(12, 5))
sns.barplot(data=df_ord, x='participant', y='param_diff', palette=bar_cols)
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('$\\|Delta$ Parameters'); plt.xlabel('Participant ID')
plt.title('Model Complexity Change')
plt.savefig('Fig4_Complexity_vs_groupgecco.png')

# --- Fig 5: OCI vs Asymmetry (Scatter) ---
if 'oci' in df_merged.columns:
    plt.figure(figsize=(6, 6))
    sns.regplot(data=df_merged, x='oci', y='lr_asymmetry', color=colors['purple'], scatter_kws={'s': 50, 'alpha':0.8})
    plt.xlabel('OCI Score'); plt.ylabel('Optimism Bias ($LR_{pos} - LR_{neg}$)')
    plt.savefig('Fig5_OCI_Correlations_vs_groupgecco.png')

# --- Fig 6: Complexity vs Performance (Scatter) ---
plt.figure(figsize=(7, 6))
sns.regplot(data=df_full, x='param_diff', y='bic_improvement', color=colors['blue'], scatter_kws={'s': 80, 'alpha': 0.6, 'edgecolor':'white'}, line_kws={'color': colors['red']}, x_jitter=0.1)
plt.xlabel('Change in Parameters'); plt.ylabel('$\\|Delta$BIC')
plt.title('Complexity vs Performance')
plt.savefig('Fig6_Complexity_vs_Performance_vs_groupgecco.png')

# --- Fig 7: OCI by Mechanism Category (Box) ---
if 'oci' in df_merged.columns:
    # Use exploded df to get categories, merge OCI, then deduplicate by (participant, Category) to count each person once per category
    df_oci_plot = df_exploded[['participant', 'Category']].drop_duplicates().merge(df_traits, on='participant')
    plt.figure(figsize=(8, 6))
    order = df_oci_plot.groupby('Category')['oci'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_oci_plot, x='Category', y='oci', order=order, palette=cat_colors, showfliers=False, width=0.5)
    sns.stripplot(data=df_oci_plot, x='Category', y='oci', order=order, color='black', alpha=0.4, jitter=0.2)
    plt.ylabel('OCI Score'); plt.xlabel('')
    plt.savefig('Fig7_OCI_Mechanism_Link_vs_groupgecco.png')

# --- Fig 8: Mechanism Frequency (Total Counts) ---
# Counting ALL instances (including multiple per participant)
cat_counts = df_exploded['Category'].value_counts().reset_index()
cat_counts.columns = ['Category', 'Count']
cat_counts = cat_counts.sort_values('Count', ascending=False)

plt.figure(figsize=(7, 6))
sns.barplot(data=cat_counts, x='Category', y='Count', palette=cat_colors, edgecolor='black', linewidth=1.0)
plt.xlabel('Mechanism Category'); plt.ylabel('Frequency (Total Mechanism Changes)')
plt.yticks(np.arange(0, cat_counts['Count'].max() + 5, 5))
for i, row in enumerate(cat_counts.itertuples()):
    plt.text(i, row.Count + 0.3, str(row.Count), ha='center', va='bottom', fontsize=12)
plt.title('Frequency of Mechanism Changes')
plt.savefig('Fig8_Mechanism_Frequency_Final_vs_groupgecco.png')

# --- Fig 9: Mechanism Combinations ---
def get_combo(m): return " + ".join(sorted(m)) if m else "None"
df_full['Combo'] = df_full['mechanisms'].apply(get_combo)
combo_stats = df_full.groupby('Combo')['bic_improvement'].agg(['mean', 'count']).sort_values('mean', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(data=combo_stats.reset_index(), y='Combo', x='mean', color=colors['blue'], edgecolor='black', linewidth=0.5)
plt.xlabel('Mean $\\|Delta$BIC'); plt.ylabel('Combination')
plt.title('Mechanism Combinations')
plt.savefig('Fig9_Combinations_vs_groupgecco.png')

# --- Fig 10: Category Impact (Mean + SEM) ---
# Counting ALL instances
stats = df_exploded.groupby('Category')['bic_improvement'].agg(['mean', 'sem', 'count']).sort_values('mean', ascending=False)
plt.figure(figsize=(8, 6))
x = np.arange(len(stats))
plt.bar(x, stats['mean'], yerr=stats['sem'], capsize=5, color=[cat_colors.get(c, 'grey') for c in stats.index], edgecolor='black', linewidth=1.0)
plt.xticks(x, stats.index); plt.ylabel('Mean $\\|Delta$BIC')
plt.title('Impact of Mechanism Categories')
for i, (idx, row) in enumerate(stats.iterrows()):
    plt.text(i, row['mean'] + row['sem'] + 2, f"n={int(row['count'])}", ha='center', fontsize=12)
plt.savefig('Fig10_Category_BIC_Impact_Final_vs_groupgecco.png')

# --- Fig 11: Regression Analysis (Forest Plot) ---
# Prepare Binary Matrix (One-hot encoding of categories per participant)
df_binary = pd.crosstab(df_exploded['participant'], df_exploded['Category']).reset_index()
cols = df_binary.columns.drop('participant')
df_binary[cols] = (df_binary[cols] > 0).astype(int) # 1 if present (even if multiple), 0 if absent

# Merge back target
df_reg = pd.merge(df_full[['participant', 'bic_improvement']], df_binary, on='participant')
for c in ['Learning', 'Memory', 'Perseveration', 'Exploration/Bias']:
    if c not in df_reg.columns: df_reg[c] = 0

# Run OLS
X = df_reg[['Learning', 'Memory', 'Perseveration', 'Exploration/Bias']]
X = sm.add_constant(X)
y = df_reg['bic_improvement']
model = sm.OLS(y, X).fit()

# Plot Forest
results_df = pd.DataFrame({
    'Factor': model.params.index, 'Coef': model.params.values, 
    'Conf_Low': model.conf_int()[0].values, 'Conf_High': model.conf_int()[1].values,
    'P': model.pvalues.values
})
results_df = results_df[results_df['Factor'] != 'const'].sort_values('Coef', ascending=False)
plot_colors = [cat_colors.get(f, 'black') for f in results_df['Factor']]

plt.figure(figsize=(8, 6))
plt.errorbar(x=results_df['Coef'], y=results_df['Factor'], xerr=results_df['Conf_High'] - results_df['Coef'], fmt='o', color='black', ecolor='gray', capsize=5)
for i, row in enumerate(results_df.itertuples()):
    plt.plot(row.Coef, row.Factor, 'o', color=plot_colors[i], markersize=10)
    sig_txt = f"p={row.P:.3f}" + ("*" if row.P < 0.05 else "")
    plt.text(row.Coef, i + 0.3, sig_txt, ha='center', fontsize=10)

plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Regression Coefficient ($\\Delta$BIC)')
plt.title('Drivers of Improvement (Regression)')
plt.savefig('Fig11_Regression_Analysis_vs_groupgecco.png')

print("All 11 Figures Generated Successfully.")
print(model.summary())