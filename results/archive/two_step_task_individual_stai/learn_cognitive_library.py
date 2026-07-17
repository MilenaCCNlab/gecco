"""
Cognitive Library Learner - Interpretable Version
==================================================
Extracts common cognitive primitives from participant models and outputs
a library with clear, interpretable function names.

Output:
  - cognitive_library.py: Interpretable cognitive primitives
  - reconstructed_models.py: Participant models using the library

Usage:
  python learn_library.py
"""

import pandas as pd
import ast
import astunparse
import os
from collections import Counter, defaultdict
import re
import copy
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/combined_models_bics.csv'
OUTPUT_DIR = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum frequency for extraction (higher = more shared, less functions)
MIN_FREQ = 5

# =============================================================================
# INTERPRETABLE FUNCTION NAMES
# =============================================================================

# Map from (category, base_name) to final interpretable name
INTERPRETABLE_NAMES = {
    # Model-Based Planning
    ('MBValuation', 'compute_mb_values'): 'compute_mb_values',
    ('MBValuation', 'get_max_q2'): 'max_q2_per_state',
    
    # MF-MB Integration
    ('ValueInt', 'integrate_mf_mb'): 'integrate_mf_mb',
    ('ValueInt', 'mf_weight'): 'mf_weight',
    
    # Memory & Forgetting
    ('MemDecay', 'decay_complement'): 'memory_retention',
    ('MemDecay', 'apply_decay'): 'apply_decay',
    ('MemDecay', 'decay_to_prior'): 'decay_to_prior',
    ('MemDecay', 'apply_retention'): 'apply_retention',
    
    # STAI/Anxiety Modulation
    ('ParamMod', 'extract_stai'): 'extract_stai',
    ('ParamMod', 'stai_scale_half'): 'stai_scale',
    ('ParamMod', 'stai_scale_to_upper_half'): 'stai_modulate',
    ('ParamMod', 'stai_complement'): 'stai_complement',
    ('ParamMod', 'stai_modulate'): 'stai_modulate',
    ('ParamMod', 'modulate_param'): 'modulate_param',
    
    # TD Learning
    ('TDUpdate', 'learning_step'): 'td_step',
    ('TDUpdate', 'td_update_q2'): 'update_q2',
    ('TDUpdate', 'td_update_q1'): 'update_q1',
    
    # Action Selection
    ('ActionSel', 'softmax_exp'): 'softmax_numerator',
    ('ActionSel', 'exp_logits'): 'exp_logits',
    ('ActionSel', 'center_values'): 'center_values',
    ('ActionSel', 'normalize_probs'): 'normalize_probs',
    ('ActionSel', 'softmax_stage1'): 'softmax_stage1',
    ('ActionSel', 'softmax_stage2'): 'softmax_stage2',
    ('ActionSel', 'softmax_choice'): 'softmax',
    ('ActionSel', 'record_stage1_choice'): 'record_stage1',
    ('ActionSel', 'record_stage2_choice'): 'record_stage2',
    
    # Initialization
    ('StateInit', 'init_q_values'): 'init_q_and_T',
    ('StateInit', 'init_mf_values'): 'init_mf_values',
    ('StateInit', 'init_choice_probs'): 'init_trial_arrays',
    ('StateInit', 'init_trial_vars'): 'init_trial_vars',
    ('StateInit', 'init_transition_matrix'): 'init_transition_matrix',
    ('StateInit', 'get_trial_actions'): 'get_actions',
    ('StateInit', 'get_trial_outcomes'): 'get_outcomes',
    
    # Likelihood
    ('Likelihood', 'compute_nll'): 'compute_nll',
    ('Likelihood', 'log_likelihood_sum'): 'log_likelihood',
    
    # Other
    ('Other', 'mechanism'): 'helper',
    ('Other', 'calc'): 'calc',
}

# Descriptions for each function
FUNCTION_DOCS = {
    'compute_mb_values': 'Model-based planning: expected value via transition model.',
    'max_q2_per_state': 'Maximum value per state for planning.',
    'integrate_mf_mb': 'Hybrid MF-MB: weighted combination of strategies.',
    'memory_retention': 'Memory retention factor: 1 - decay.',
    'stai_scale': 'STAI linear scaling: 0.5 * stai.',
    'stai_modulate': 'STAI affine modulation: 0.5 * stai + 0.5.',
    'td_step': 'Single TD update: alpha * delta.',
    'update_q2': 'Update stage-2 Q-value, return TD error.',
    'update_q1': 'Update stage-1 Q from stage-2 bootstrap.',
    'softmax': 'Softmax probability distribution.',
    'center_values': 'Center values for numerical stability: q - max(q).',
    'init_q_and_T': 'Initialize Q-values and transition matrix.',
    'init_trial_arrays': 'Initialize trial probability arrays.',
    'get_actions': 'Get actions for trial t.',
    'compute_nll': 'Compute negative log-likelihood.',
    'log_likelihood': 'Sum of log probabilities.',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_best_models(csv_path):
    """Load best model per participant based on BIC."""
    df = pd.read_csv(csv_path)
    df['participant'] = df['filename'].apply(lambda x: int(x.split('participant')[1].split('.')[0]))
    best_models = {}
    for p in df['participant'].unique():
        p_df = df[df['participant'] == p]
        best_row = p_df.loc[p_df['bic'].idxmin()]
        best_models[p] = best_row['code']
    return best_models


# =============================================================================
# VARIABLE NORMALIZATION
# =============================================================================

class VariableNormalizer(ast.NodeTransformer):
    """Normalize variable names to canonical forms."""
    
    def __init__(self):
        self.base_mapping = {
            # Input sequences
            'action_1': 'a1_seq', 'action_2': 'a2_seq', 'state': 's_seq', 
            'reward': 'r_seq', 'stai': 'stai_arr', 'model_parameters': 'params',
            
            # Trial count
            'n_trials': 'num_trials', 'N': 'num_trials',
            
            # STAI scalar
            'stai_score': 'stai', 'stai_val': 'stai', 'st': 'stai', 's_anx': 'stai',
            'stai0': 'stai',
            
            # Q-values stage 1
            'q_stage1_mf': 'q1', 'q1_mf': 'q1', 'Q1_mf': 'q1', 'Q1': 'q1',
            'q_stage1_mb': 'q1_mb', 'Q1_mb': 'q1_mb', 'Q1_MB': 'q1_mb',
            'q1_hybrid': 'q1_combined', 'q1_eff': 'q1_combined', 'q1_net': 'q1_combined',
            'q1_combined': 'q1_combined', 'q1_total': 'q1_combined', 'q1_comb': 'q1_combined',
            'q1_policy_vals': 'q1_combined',
            
            # Q-values stage 2 row
            'q2_s': 'q2_row', 'q2_vec': 'q2_row', 'q2_state': 'q2_row',
            'q2_policy_vals': 'q2_row', 'pref2': 'q2_row',
            
            # Transition matrix
            'transition_matrix': 'T', 'T_known': 'T', 'T_fixed': 'T', 'T_est': 'T',
            'trans_counts': 'T_counts', 'T_counts': 'T_counts',
            
            # Probability sequences
            'p_choice_1': 'p1_seq', 'p_choice_2': 'p2_seq',
            
            # Logits / centered values
            'logits1': 'logits', 'logits2': 'logits', 'l1': 'logits', 'l2': 'logits',
            'q2c': 'q_centered', 'q1c': 'q_centered', 'q_centered': 'q_centered',
            'c_q1': 'q_centered', 'c_q2': 'q_centered',
            
            # Decay/forgetting
            'forget': 'decay', 'rho': 'decay', 'rho_forget': 'decay', 
            'tau_forget': 'decay', 'f_forget': 'decay', 'forget0': 'decay_base',
            'decay_eff': 'decay_eff', 'decay_eff_1': 'decay_eff', 'decay_eff_2': 'decay_eff',
            'keep': 'keep', 'f': 'decay',
            
            # Prediction errors
            'pe1': 'delta1', 'pe2': 'delta2', 'td1': 'delta1', 'td2': 'delta2',
            'delta1_boot': 'delta1', 'delta1_bootstrap': 'delta1',
            
            # Learning rates
            'eta': 'alpha', 'rho_v': 'alpha', 'alphaQ': 'alpha',
            'alpha_pos': 'alpha_pos', 'alpha_neg': 'alpha_neg',
            'alpha_plus': 'alpha_pos', 'alpha_minus': 'alpha_neg',
            'eff_alpha_pos': 'alpha_pos', 'eff_alpha_neg': 'alpha_neg',
            'alpha2': 'alpha', 'alpha_q': 'alpha', 'alpha_r': 'alpha',
            'mu_win': 'alpha_pos', 'mu_loss': 'alpha_neg',
            
            # Inverse temperature
            'beta1': 'beta', 'beta2': 'beta', 'beta_eff': 'beta', 
            'b1': 'beta', 'b2': 'beta', 'beta_t': 'beta',
            'beta_base_eff': 'beta',
            
            # Perseveration
            'kappa': 'pers', 'kappa_eff': 'pers_eff', 'kappa0': 'pers', 'kappa1': 'pers',
            'pi': 'pers', 'pi_eff': 'pers_eff', 'stick': 'pers',
            'persev': 'pers', 'persev_eff': 'pers_eff', 'perseveration': 'pers',
            
            # MB weight
            'w': 'w_mb', 'omega': 'w_mb', 'omega_eff': 'w_mb_eff', 'w_eff': 'w_mb_eff',
            'w_mb': 'w_mb', 'mb_weight': 'w_mb', 'weight_mb': 'w_mb', 'phi': 'w_mb',
            
            # Lambda (eligibility)
            'lam': 'lambda_', 'lambd': 'lambda_', 'lambda_e': 'lambda_',
            
            # Previous action
            'prev_a1': 'last_a1', 'a1_prev': 'last_a1',
            
            # Max Q values
            'max_Q2': 'max_q2', 'max_q_stage2': 'max_q2', 'vmax': 'max_q2',
            
            # Epsilon
            'eps': 'epsilon', 'eps_h': 'epsilon',
        }
        self.q2_tables = set()
        self.t_tables = set()

    def analyze(self, tree):
        """Identify Q2 and T matrices by usage patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                if isinstance(node.left, ast.Name):
                    self.t_tables.add(node.left.id)
            
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                dims = []
                if isinstance(node.slice, ast.Tuple):
                    dims = node.slice.elts
                if len(dims) == 2:
                    row = dims[0]
                    col = dims[1]
                    row_name = row.id if isinstance(row, ast.Name) else None
                    col_name = col.id if isinstance(col, ast.Name) else None
                    
                    if row_name in ['state', 's', 's_seq', 's2'] and col_name in ['action_2', 'a2', 'a2_seq']:
                        self.q2_tables.add(node.value.id)
                    elif row_name in ['action_1', 'a1', 'a1_seq'] and col_name in ['state', 's', 's_seq']:
                        self.t_tables.add(node.value.id)

    def visit_Name(self, node):
        if node.id in self.t_tables:
            return ast.copy_location(ast.Name(id='T', ctx=node.ctx), node)
        if node.id in self.q2_tables:
            return ast.copy_location(ast.Name(id='Q2', ctx=node.ctx), node)
        if node.id in self.base_mapping:
            return ast.copy_location(ast.Name(id=self.base_mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node):
        if node.arg in self.t_tables:
             return ast.copy_location(ast.arg(arg='T', annotation=node.annotation), node)
        if node.arg in self.q2_tables:
             return ast.copy_location(ast.arg(arg='Q2', annotation=node.annotation), node)
        if node.arg in self.base_mapping:
            return ast.copy_location(ast.arg(arg=self.base_mapping[node.arg], annotation=node.annotation), node)
        return node


class NumpyNormalizer(ast.NodeTransformer):
    """Normalize numpy calls."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'np':
                new_keywords = []
                for kw in node.keywords:
                    if kw.arg == 'dtype':
                        if isinstance(kw.value, ast.Name) and kw.value.id == 'float':
                            continue
                        if isinstance(kw.value, ast.Attribute):
                            if kw.value.attr in ['float64', 'float32', 'float_']:
                                continue
                    new_keywords.append(kw)
                node.keywords = new_keywords
        return node


class ExpressionCanonicalizer(ast.NodeTransformer):
    """Sort commutative operations for consistent comparison."""
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_str = astunparse.unparse(node.left).strip()
            right_str = astunparse.unparse(node.right).strip()
            if right_str < left_str:
                node.left, node.right = node.right, node.left
        return node


class StatementSorter(ast.NodeTransformer):
    """Sort independent statements for consistent ordering."""
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        
        # Split into groups
        groups = []
        current_group = []
        for stmt in node.body:
            if isinstance(stmt, ast.For):
                if current_group:
                    groups.append(('assign_block', current_group))
                    current_group = []
                groups.append(('for', [stmt]))
            elif isinstance(stmt, ast.Return):
                if current_group:
                    groups.append(('assign_block', current_group))
                    current_group = []
                groups.append(('return', [stmt]))
            else:
                current_group.append(stmt)
        if current_group:
            groups.append(('assign_block', current_group))
        
        # Sort assignment blocks
        new_body = []
        for gtype, stmts in groups:
            if gtype == 'assign_block':
                stmts_strs = [(astunparse.unparse(s).strip(), s) for s in stmts]
                stmts_strs.sort(key=lambda x: x[0])
                new_body.extend([s for _, s in stmts_strs])
            else:
                new_body.extend(stmts)
        node.body = new_body
        return node


# =============================================================================
# CODE CORPUS
# =============================================================================

class CodeCorpus:
    """Manage normalized ASTs for all participants."""
    
    def __init__(self, models_dict):
        self.original_models = models_dict
        self.normalized_asts = {}
        self._normalize_all()
    
    def _normalize_all(self):
        for pid, code in self.original_models.items():
            try:
                tree = ast.parse(code)
                
                normalizer = VariableNormalizer()
                normalizer.analyze(tree)
                tree = normalizer.visit(tree)
                
                tree = NumpyNormalizer().visit(tree)
                tree = ExpressionCanonicalizer().visit(tree)
                tree = StatementSorter().visit(tree)
                
                ast.fix_missing_locations(tree)
                self.normalized_asts[pid] = tree
            except Exception as e:
                print(f"  Warning: Failed to normalize participant {pid}: {e}")
    
    def extract_variables(self, stmt_nodes, known_functions=None):
        """Extract input/output variables from statements."""
        if known_functions is None:
            known_functions = set()
        
        assigned = set()
        used = set()
        
        class VarVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                assigned.add(elt.id)
                self.visit(node.value)
            
            def visit_AugAssign(self, node):
                if isinstance(node.target, ast.Name):
                    assigned.add(node.target.id)
                    used.add(node.target.id)
                self.visit(node.value)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used.add(node.id)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in known_functions:
                    pass
                for arg in node.args:
                    self.visit(arg)
                for kw in node.keywords:
                    self.visit(kw.value)

        for stmt in stmt_nodes:
            VarVisitor().visit(stmt)
        
        # Inputs: used but not assigned locally
        skip = {'np', 'int', 'float', 'range', 'len', 'True', 'False', 'None'}
        skip.update(known_functions)
        inputs = sorted([v for v in (used - assigned) if v not in skip])
        outputs = sorted(list(assigned))
        
        return inputs, outputs
    
    def find_common_sequence(self, min_len=2, max_len=8, min_freq=5):
        """Find most valuable repeated statement sequence."""
        sequences = Counter()
        
        for pid, tree in self.normalized_asts.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.FunctionDef)):
                    stmts = node.body if hasattr(node, 'body') else []
                    norm_stmts = [astunparse.unparse(s).strip() for s in stmts]
                    
                    for n in range(min_len, min(max_len + 1, len(stmts) + 1)):
                        for i in range(len(stmts) - n + 1):
                            has_return = any(isinstance(stmts[i+k], ast.Return) for k in range(n))
                            if has_return:
                                continue
                            seq = tuple(norm_stmts[i : i+n])
                            sequences[seq] += 1
        
        candidates = []
        for seq, freq in sequences.items():
            if freq < min_freq:
                continue
            n_stmts = len(seq)
            score = (n_stmts * freq) - n_stmts
            candidates.append((score, seq, freq))
        
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]
    
    def replace_sequence(self, seq_strs, func_name, inputs, outputs):
        """Replace sequence with function call in all models."""
        seq_set = set(seq_strs)
        count = 0
        
        class SeqReplacer(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)
                node.body = self._replace_in_body(node.body)
                return node
            
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                node.body = self._replace_in_body(node.body)
                return node
            
            def _replace_in_body(self, stmts):
                nonlocal count
                new_body = []
                i = 0
                while i < len(stmts):
                    matched = False
                    if i + len(seq_strs) <= len(stmts):
                        candidate = [astunparse.unparse(stmts[i+k]).strip() for k in range(len(seq_strs))]
                        if tuple(candidate) == seq_strs:
                            # Create function call
                            args = [ast.Name(id=v, ctx=ast.Load()) for v in inputs]
                            call = ast.Call(
                                func=ast.Name(id=func_name, ctx=ast.Load()),
                                args=args, keywords=[]
                            )
                            
                            if outputs:
                                if len(outputs) == 1:
                                    target = ast.Name(id=outputs[0], ctx=ast.Store())
                                else:
                                    target = ast.Tuple(
                                        elts=[ast.Name(id=o, ctx=ast.Store()) for o in outputs],
                                        ctx=ast.Store()
                                    )
                                new_stmt = ast.Assign(targets=[target], value=call)
                            else:
                                new_stmt = ast.Expr(value=call)
                            
                            new_body.append(new_stmt)
                            i += len(seq_strs)
                            count += 1
                            matched = True
                    
                    if not matched:
                        new_body.append(stmts[i])
                        i += 1
                return new_body
        
        for pid, tree in self.normalized_asts.items():
            SeqReplacer().visit(tree)
            ast.fix_missing_locations(tree)
        
        return count


# =============================================================================
# CATEGORIZATION
# =============================================================================

def categorize_function(block_code):
    """Categorize a code block into a cognitive category."""
    
    # StateInitialization
    if "np.zeros" in block_code and ("p1_seq" in block_code or "p2_seq" in block_code):
        return "StateInit", "init_choice_probs"
    if "np.zeros" in block_code and "Q2" in block_code:
        return "StateInit", "init_q_values"
    if "np.zeros" in block_code and "q1" in block_code:
        return "StateInit", "init_mf_values"
    if "len(a1_seq)" in block_code:
        return "StateInit", "init_trial_vars"
    if "np.array([[0.7" in block_code or "[[0.7, 0.3]" in block_code:
        return "StateInit", "init_transition_matrix"
    
    # ParameterModulation
    if "stai_arr[0]" in block_code:
        return "ParamMod", "extract_stai"
    if "stai" in block_code and ("np.clip" in block_code or "* stai" in block_code):
        return "ParamMod", "modulate_param"
    
    # MemoryDecay
    if "(1.0 - decay)" in block_code or "decay_complement" in block_code:
        return "MemDecay", "decay_complement"
    if "*= keep" in block_code or "* keep" in block_code:
        return "MemDecay", "apply_retention"
    
    # ModelBasedValuation
    if "T @" in block_code or "@ max_q2" in block_code:
        return "MBValuation", "compute_mb_values"
    if "np.max(Q2, axis=1)" in block_code:
        return "MBValuation", "get_max_q2"
    
    # ValueIntegration
    if ("1.0 - w_mb" in block_code or "(1 - w_mb)" in block_code) and "q1_mb" in block_code:
        return "ValueInt", "integrate_mf_mb"
    
    # ActionSelection
    if "np.exp" in block_code and "np.sum" in block_code:
        if "probs1" in block_code or "p1_seq" in block_code:
            return "ActionSel", "softmax_stage1"
        if "probs2" in block_code or "p2_seq" in block_code:
            return "ActionSel", "softmax_stage2"
        return "ActionSel", "softmax_choice"
    if "- np.max" in block_code:
        return "ActionSel", "center_values"
    if "p1_seq[t]" in block_code:
        return "ActionSel", "record_stage1_choice"
    if "p2_seq[t]" in block_code:
        return "ActionSel", "record_stage2_choice"
    
    # TDValueUpdate
    if "delta2" in block_code and "Q2" in block_code and "+=" in block_code:
        return "TDUpdate", "td_update_q2"
    if "delta1" in block_code and "q1" in block_code and "+=" in block_code:
        return "TDUpdate", "td_update_q1"
    if "alpha * delta" in block_code:
        return "TDUpdate", "learning_step"
    
    # LikelihoodEstimation
    if "np.log" in block_code and ("nll" in block_code or "p1_seq" in block_code):
        return "Likelihood", "compute_nll"
    
    # Get actions/outcomes
    if "a1_seq[t]" in block_code and "a2_seq[t]" in block_code:
        return "StateInit", "get_trial_actions"
    if "r_seq[t]" in block_code and "s_seq[t]" in block_code:
        return "StateInit", "get_trial_outcomes"
    
    return "Other", "mechanism"


def get_interpretable_name(category, base_name, counter):
    """Get interpretable function name, avoiding duplicates."""
    key = (category, base_name)
    if key in INTERPRETABLE_NAMES:
        base = INTERPRETABLE_NAMES[key]
    else:
        base = base_name
    
    # Check if we need to add a number
    if counter.get(base, 0) > 0:
        return f"{base}_{counter[base] + 1}"
    return base


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Cognitive Library Learner - Interpretable Version")
    print("=" * 70)
    
    # Load models
    print("\n[1] Loading participant models...")
    best_models = get_best_models(INPUT_FILE)
    print(f"    Loaded {len(best_models)} participants")
    
    # Create corpus
    print("\n[2] Normalizing code...")
    corpus = CodeCorpus(best_models)
    print(f"    Normalized {len(corpus.normalized_asts)} models")
    
    # Extract library
    print("\n[3] Extracting cognitive primitives...")
    library = {}
    name_counter = defaultdict(int)
    func_count = 0
    skipped_seqs = set()  # Track skipped sequences to avoid infinite loop
    iterations = 0
    max_iterations = 100
    
    while func_count < 30 and iterations < max_iterations:
        iterations += 1
        best = corpus.find_common_sequence(min_len=2, max_len=8, min_freq=MIN_FREQ, skip_seqs=skipped_seqs)
        if not best:
            break
        
        score, seq_strs, freq = best
        if score <= 0:
            break
        
        block_code = "\n".join(seq_strs)
        
        # Skip problematic patterns
        if "= Q2[s]" in block_code and "Q2[(" not in block_code:
            skipped_seqs.add(seq_strs)
            continue
        if "probs1 = " in block_code or "probs2 = " in block_code:
            skipped_seqs.add(seq_strs)
            continue
        
        try:
            dummy_code = f"def dummy():\n" + "\n".join(["    " + s for s in seq_strs])
            dummy_tree = ast.parse(dummy_code)
            stmt_nodes = dummy_tree.body[0].body
        except:
            continue
        
        inputs, outputs = corpus.extract_variables(stmt_nodes, known_functions=set(library.keys()))
        
        # Get category and interpretable name
        category, base_name = categorize_function(block_code)
        func_name = get_interpretable_name(category, base_name, name_counter)
        name_counter[func_name.split('_')[0] if '_' in func_name else func_name] += 1
        
        print(f"    [{category:12}] {func_name:<25} (freq={freq:2d}, lines={len(seq_strs)})")
        
        library[func_name] = {
            'category': category,
            'code': seq_strs,
            'inputs': inputs,
            'outputs': outputs,
        }
        
        corpus.replace_sequence(seq_strs, func_name, inputs, outputs)
        func_count += 1
    
    print(f"\n    Extracted {len(library)} functions")
    
    # Save library
    print("\n[4] Saving library...")
    lib_path = os.path.join(OUTPUT_DIR, "cognitive_library.py")
    
    # Group by category
    by_category = defaultdict(list)
    for name, data in library.items():
        by_category[data['category']].append((name, data))
    
    category_order = ['MBValuation', 'ValueInt', 'MemDecay', 'ParamMod', 
                      'TDUpdate', 'ActionSel', 'StateInit', 'Likelihood', 'Other']
    
    with open(lib_path, "w") as f:
        f.write('"""Cognitive Library - Interpretable Primitives\n\n')
        f.write('Automatically extracted cognitive modeling primitives.\n')
        f.write('Each function represents a distinct cognitive mechanism.\n')
        f.write('"""\n\n')
        f.write("import numpy as np\n")
        
        for cat in category_order:
            if cat not in by_category:
                continue
            
            f.write(f"\n\n# {'='*60}\n")
            f.write(f"# {cat}\n")
            f.write(f"# {'='*60}\n")
            
            for name, data in by_category[cat]:
                args = ", ".join(data['inputs'])
                doc = FUNCTION_DOCS.get(name, '')
                
                f.write(f"\ndef {name}({args}):\n")
                if doc:
                    f.write(f'    """{doc}"""\n')
                
                for line in data['code']:
                    f.write(f"    {line}\n")
                
                if data['outputs']:
                    rets = ", ".join(data['outputs'])
                    f.write(f"    return {rets}\n")
    
    print(f"    Library saved to: {lib_path}")
    
    # Save models
    print("\n[5] Saving reconstructed models...")
    models_path = os.path.join(OUTPUT_DIR, "reconstructed_models.py")
    
    with open(models_path, "w") as f:
        f.write("import numpy as np\n")
        f.write("from .cognitive_library import *\n\n")
        
        for pid in sorted(corpus.normalized_asts.keys()):
            tree = corpus.normalized_asts[pid]
            func_def = tree.body[0]
            code = astunparse.unparse(func_def)
            # Fix slice syntax
            code = re.sub(r'\[\((:,?\s*\d*)\)\]', r'[\1]', code)
            code = re.sub(r'\[\((\d+,?\s*:)\)\]', r'[\1]', code)
            f.write(f"# Participant {pid}\n")
            f.write(code)
            f.write("\n")
    
    print(f"    Models saved to: {models_path}")
    
    # Stats
    orig_lines = sum(len(code.split('\n')) for code in best_models.values())
    with open(models_path, 'r') as f:
        recon_lines = len(f.readlines())
    with open(lib_path, 'r') as f:
        lib_lines = len(f.readlines())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Functions extracted:  {len(library)}")
    print(f"  Original lines:       {orig_lines}")
    print(f"  Compressed lines:     {recon_lines + lib_lines}")
    print(f"  Compression ratio:    {orig_lines / (recon_lines + lib_lines):.2f}x")
    print("\n  Functions by category:")
    for cat in category_order:
        if cat in by_category:
            print(f"    {cat}: {len(by_category[cat])}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
