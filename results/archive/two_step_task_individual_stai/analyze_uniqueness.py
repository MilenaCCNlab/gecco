"""
Cognitive Model Uniqueness Analyzer
Analyzes what makes each participant's model unique vs shared

This helps draw insights into:
1. What computational strategies are shared across participants
2. What unique mechanisms each participant uses
3. How anxiety (STAI) affects different computational components differently per participant
"""

import pandas as pd
import re
from collections import defaultdict
import numpy as np

INPUT_FILE = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/combined_models_bics.csv'
RECON_FILE = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned/reconstructed_models.py'

def get_best_models(csv_path):
    df = pd.read_csv(csv_path)
    df['participant'] = df['filename'].apply(lambda x: int(x.split('participant')[1].split('.')[0]))
    best_models = {}
    for p in df['participant'].unique():
        p_df = df[df['participant'] == p]
        best_row = p_df.loc[p_df['bic'].idxmin()]
        best_models[p] = best_row['code']
    return best_models


def analyze_stai_modulation(code):
    """Extract unique STAI modulation formulas"""
    # Find all expressions involving stai
    patterns = []
    
    # Look for various STAI formulas
    formulas = {
        'stai_linear_decay': r'w_mb\s*=\s*(?:np\.clip\()?.*\(1\.0?\s*-\s*stai',  # w_mb = (1 - stai) * ...
        'stai_linear_amplify': r'\(1\.0?\s*\+\s*stai\)',  # (1 + stai) * 
        'stai_scaled_half': r'\(0\.5\s*\*\s*stai\)',  # (0.5 * stai)
        'stai_upper_half': r'\(0\.5\s*\+\s*0\.5\s*\*\s*stai\)',  # (0.5 + 0.5*stai)
        'stai_asymmetric': r'alpha_(?:pos|neg).*stai',  # asymmetric learning rates
        'stai_rare_modulate': r'(?:rare|is_rare).*stai',  # rare transition modulation
        'stai_surprise': r'surprise.*stai|stai.*surprise',  # surprise modulation
        'stai_uncertainty': r'(?:unc|uncertainty|entropy).*stai|stai.*(?:unc|uncertainty|entropy)',  # uncertainty
        'stai_decay': r'decay.*stai|stai.*decay',  # decay modulation
        'stai_perseveration': r'pers.*stai|stai.*pers',  # perseveration
        'stai_volatility': r'(?:vol|volatility).*stai|stai.*(?:vol|volatility)',  # volatility
    }
    
    for name, pattern in formulas.items():
        if re.search(pattern, code, re.IGNORECASE):
            patterns.append(name)
    
    return patterns


def analyze_mechanisms(code):
    """Identify unique computational mechanisms"""
    mechanisms = []
    
    # Model-based vs model-free
    if 'T @' in code or 'T @' in code:
        mechanisms.append('Model-Based (T @ max_q2)')
    if 'q1_mb' in code and 'q1' in code and ('w_mb' in code or 'omega' in code):
        mechanisms.append('MF-MB Hybrid')
    
    # Learning mechanisms
    if 'alpha_pos' in code and 'alpha_neg' in code:
        mechanisms.append('Asymmetric Learning Rates')
    if 'lambda_' in code or 'lambda ' in code:
        mechanisms.append('Eligibility Trace (lambda)')
    if 'spill' in code:
        mechanisms.append('Value Spillover')
        
    # Uncertainty/risk
    if 'entropy' in code.lower() or '* np.log' in code:
        mechanisms.append('Entropy Computation')
    if 'uncertainty' in code.lower() or 'unc' in code:
        mechanisms.append('Uncertainty-Based')
    if 'risk' in code.lower():
        mechanisms.append('Risk Sensitivity')
    if 'var' in code and ('sqrt' in code or 'np.sqrt' in code):
        mechanisms.append('Variance/Std Computation')
    
    # Heuristics
    if 'wsls' in code.lower() or 'win_stay' in code.lower():
        mechanisms.append('Win-Stay-Lose-Shift')
    if 'rare' in code and 'is_common' not in code.lower():
        mechanisms.append('Rare Transition Handling')
    if 'surprise' in code:
        mechanisms.append('Surprise Signal')
    if 'volatility' in code or 'vol ' in code:
        mechanisms.append('Volatility Tracking')
        
    # Memory
    if 'decay' in code or 'forget' in code:
        mechanisms.append('Memory Decay/Forgetting')
    if 'leak' in code:
        mechanisms.append('Value Leakage')
    if 'prior' in code:
        mechanisms.append('Decay to Prior')
        
    # Perseveration
    if 'pers' in code or 'stick' in code:
        mechanisms.append('Perseveration/Stickiness')
    if 'last_a1' in code or 'prev_a' in code:
        mechanisms.append('Choice History Tracking')
        
    # Utility
    if 'loss' in code.lower() and ('aversion' in code.lower() or 'nu' in code):
        mechanisms.append('Loss Aversion')
    if 'curv' in code or 'util' in code:
        mechanisms.append('Utility Curvature')
    if 'bias_safe' in code or 'safe' in code:
        mechanisms.append('Safety Bias')
        
    # Transition learning
    if 'alpha_T' in code or 'alpha_t' in code or 'tau_T' in code:
        mechanisms.append('Transition Learning')
    if 'T_counts' in code:
        mechanisms.append('Transition Counting')
        
    return list(set(mechanisms))


def count_library_usage(recon_code, pid):
    """Count how many library functions each participant uses"""
    lib_functions = [
        'StateInit_', 'ParamMod_', 'MemDecay_', 'MBValuation_',
        'ValueInt_', 'ActionSel_', 'TDUpdate_', 'Likelihood_', 'Other_'
    ]
    
    # Find the participant's code block
    pattern = f"# Participant {pid}\n"
    start = recon_code.find(pattern)
    if start == -1:
        return {}
    
    # Find end (next participant)
    end = recon_code.find("# Participant", start + len(pattern))
    if end == -1:
        end = len(recon_code)
    
    block = recon_code[start:end]
    
    usage = {}
    for func_prefix in lib_functions:
        count = block.count(func_prefix)
        if count > 0:
            usage[func_prefix.rstrip('_')] = count
    
    return usage


def main():
    print("=" * 80)
    print("COGNITIVE MODEL UNIQUENESS ANALYSIS")
    print("=" * 80)
    
    # Load models
    best_models = get_best_models(INPUT_FILE)
    with open(RECON_FILE, 'r') as f:
        recon_code = f.read()
    
    # Analyze each participant
    all_mechanisms = defaultdict(int)
    all_stai_patterns = defaultdict(int)
    participant_profiles = {}
    
    for pid, code in sorted(best_models.items()):
        mechanisms = analyze_mechanisms(code)
        stai_patterns = analyze_stai_modulation(code)
        lib_usage = count_library_usage(recon_code, pid)
        
        for m in mechanisms:
            all_mechanisms[m] += 1
        for p in stai_patterns:
            all_stai_patterns[p] += 1
            
        participant_profiles[pid] = {
            'mechanisms': mechanisms,
            'stai_patterns': stai_patterns,
            'lib_usage': lib_usage
        }
    
    # Print mechanism frequency analysis
    print("\n" + "=" * 80)
    print("MECHANISM FREQUENCY (How many participants use each)")
    print("=" * 80)
    for mech, count in sorted(all_mechanisms.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {mech:<35} {count:2d}/45 {bar}")
    
    # Print STAI modulation patterns
    print("\n" + "=" * 80)
    print("STAI MODULATION PATTERNS")
    print("=" * 80)
    for pattern, count in sorted(all_stai_patterns.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {pattern:<35} {count:2d}/45 {bar}")
    
    # Print participant-specific insights
    print("\n" + "=" * 80)
    print("INDIVIDUAL PARTICIPANT INSIGHTS")
    print("=" * 80)
    
    for pid in sorted(participant_profiles.keys()):
        profile = participant_profiles[pid]
        
        # Find unique mechanisms (used by < 10 participants)
        unique_mechs = [m for m in profile['mechanisms'] if all_mechanisms[m] < 10]
        unique_stai = [p for p in profile['stai_patterns'] if all_stai_patterns[p] < 10]
        
        # Find shared mechanisms (used by > 30 participants)
        shared_mechs = [m for m in profile['mechanisms'] if all_mechanisms[m] > 30]
        
        print(f"\nParticipant {pid}:")
        print(f"  Library calls: {sum(profile['lib_usage'].values())} total ({', '.join(f'{k}:{v}' for k,v in profile['lib_usage'].items())})")
        
        if unique_mechs:
            print(f"  UNIQUE mechanisms: {', '.join(unique_mechs)}")
        if unique_stai:
            print(f"  UNIQUE STAI patterns: {', '.join(unique_stai)}")
        if shared_mechs:
            print(f"  Shared mechanisms: {', '.join(shared_mechs)}")
    
    # Cluster participants by strategy
    print("\n" + "=" * 80)
    print("PARTICIPANT CLUSTERS BY STRATEGY")
    print("=" * 80)
    
    strategy_clusters = {
        'Uncertainty-Driven': [],
        'Asymmetric Learners': [],
        'WSLS/Heuristic': [],
        'Volatility-Sensitive': [],
        'Risk/Loss-Averse': [],
        'Memory-Decaying': [],
        'Pure MB/MF': [],
    }
    
    for pid, profile in participant_profiles.items():
        mechs = set(profile['mechanisms'])
        
        if 'Uncertainty-Based' in mechs or 'Entropy Computation' in mechs:
            strategy_clusters['Uncertainty-Driven'].append(pid)
        if 'Asymmetric Learning Rates' in mechs:
            strategy_clusters['Asymmetric Learners'].append(pid)
        if 'Win-Stay-Lose-Shift' in mechs:
            strategy_clusters['WSLS/Heuristic'].append(pid)
        if 'Volatility Tracking' in mechs:
            strategy_clusters['Volatility-Sensitive'].append(pid)
        if 'Risk Sensitivity' in mechs or 'Loss Aversion' in mechs:
            strategy_clusters['Risk/Loss-Averse'].append(pid)
        if 'Memory Decay/Forgetting' in mechs or 'Value Leakage' in mechs:
            strategy_clusters['Memory-Decaying'].append(pid)
        if 'Model-Based (T @ max_q2)' in mechs and 'MF-MB Hybrid' not in mechs:
            strategy_clusters['Pure MB/MF'].append(pid)
    
    for cluster_name, pids in strategy_clusters.items():
        if pids:
            print(f"\n  {cluster_name} ({len(pids)} participants):")
            print(f"    {pids}")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
