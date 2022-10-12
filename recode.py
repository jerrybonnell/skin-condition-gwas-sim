import pandas as pd
import sys
import os
import itertools
import numpy as np
import random
from pandarallel import pandarallel
import requests
import json
from tqdm import tqdm
import glob

"""
cat recode_chunks_new | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/gwas_sim && ./run_query.sh"
"""

tqdm.pandas()

assert len(sys.argv) == 3
df_map = pd.read_csv(
    "hapmap3_r2_b36_fwd.qc.poly/CEU/hapmap3_r3_b36_fwd.CEU.qc.poly_tmp_filtered.map",
    sep="\t", header=None)
df_map['index'] = range(0, len(df_map))

with open("variantinfo/all.json", "r") as f:
    all_json = json.load(f)
assert len(all_json.keys()) >= len(df_map)

def variant_info_from_json(rsid):
    rsid_content = all_json[rsid]
    if False in rsid_content:
        return np.nan, np.nan
    # extract that G>T and take the T as mutation and G as reference
    ref2alt = [(entry[entry.index(">")-1], entry[entry.index(">")+1])
            for entry in rsid_content if ">" in entry]
    if len(ref2alt) == 0:
        return -1, np.nan
    # while mutations could be different the reference should be the same
    # across the board
    if len(set([r[0] for r in ref2alt])) == 1:
        return ref2alt[0][0], list(set([r[1] for r in ref2alt]))
    else:
        return -2, np.nan


df_map['ref'], df_map['alt'] = zip(
    *df_map[1].progress_apply(variant_info_from_json))
df_map['ped_index'] = 2*df_map['index']+6


def codify_nucleic_acid_pair(x, pos1, pos2):
    # keep missing data missing
    if x[pos1] != x[pos1] or x[pos2] != x[pos2]:
        return np.nan
    ref = df_map[df_map['ped_index'] == pos1]['ref'].to_list()
    alt = df_map[df_map['ped_index'] == pos1]['alt'].to_list()[0]
    # the allele has an appearance of something other than ref
    # or alt
    if sum([allele not in ref+alt for allele in x.to_list()]) > 0:
        return 3
    mutations = sum([allele in alt for allele in x.to_list()])
    return mutations


print("loading ped...")
df = pd.read_hdf(sys.argv[1])

# 1513253
assert 2*len(df_map)+6 == len(df.columns)-1
assert len(df.columns)-1 == 1513251+1
alelle_pairs = list(zip(range(6,len(df.columns)-1, 2),
                        range(7,len(df.columns)-1, 2)))
alelle_pairs = [list(p) for p in alelle_pairs]
assert len(alelle_pairs) == len(df_map)

os.makedirs("recode_chunks", exist_ok=True)

begin_chunk, end_chunk = sys.argv[2].split(" ")
begin_chunk = int(begin_chunk)
end_chunk = int(end_chunk)
print((begin_chunk, end_chunk))

recoded = []
for pair in tqdm(alelle_pairs[begin_chunk:end_chunk]):
    recoded.append(df[pair].apply(
        codify_nucleic_acid_pair,
        args=(pair[0], pair[1]), axis=1))

pd.concat(recoded, axis=1).\
    to_hdf(f'recode_chunks/recode_chunk_{begin_chunk}_{end_chunk}.h5',
    key='db', mode='w')
