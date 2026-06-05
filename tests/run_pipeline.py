"""Run the full cellcall pipeline with timing."""
import gc
import time
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cellcall.connect_profile import create_nichcon_object, connect_profile

# Load from parquet (much faster and less memory than Rdata)
data_path = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata", "example_data.parquet")
count_matrix = pd.read_parquet(data_path)
gc.collect()

print(f"Loaded: {count_matrix.shape}")

obj = create_nichcon_object(
    data=count_matrix, min_feature=0, names_field=2, names_delim="_",
    source="UMI", org="Homo sapiens"
)
del count_matrix
gc.collect()

t0 = time.time()
result = connect_profile(
    obj, p_value_cor=0.05, cor_value=0.1, top_target_cor=1,
    method="weighted", p_adjust=0.05, use_type="median", probs=0.75,
    org="Homo sapiens", is_core=True, verbose=True
)
t1 = time.time()

print(f"Total: {t1-t0:.2f}s")
print(f"expr_l_r_log2_scale shape: {result['expr_l_r_log2_scale'].shape}")
print(f"Non-zero L-R pairs: {(result['expr_l_r_log2_scale'] > 0).any(axis=1).sum()}")
print(f"Max value: {result['expr_l_r_log2_scale'].max().max():.6f}")
print(f"Regulons non-zero: {(result['regulons_matrix'] > 0).sum().sum()}")

# Save outputs for parity testing
ref_dir = os.path.join(os.path.dirname(__file__), "reference")
os.makedirs(ref_dir, exist_ok=True)
for key in ["expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2", "expr_l_r_log2_scale"]:
    result[key].to_csv(os.path.join(ref_dir, f"{key}.csv"))
    print(f"Saved {key}")
