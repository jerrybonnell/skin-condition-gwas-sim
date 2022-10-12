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

df_map = pd.read_csv(
    "hapmap3_r2_b36_fwd.qc.poly/CEU/hapmap3_r3_b36_fwd.CEU.qc.poly_tmp_filtered.map",
    sep="\t", header=None)
df_map['index'] = range(0, len(df_map))

alelle_pairs = list(zip(range(6, 2*len(df_map)+6, 2),
                        range(7, 2*len(df_map)+6, 2)))
alelle_pairs = [list(p) for p in alelle_pairs]

chunk_size = 2000

a = np.append(np.arange(0, len(alelle_pairs), chunk_size),
          np.array(len(alelle_pairs)))
b = np.append(np.arange(chunk_size, len(alelle_pairs), chunk_size),
          np.array(len(alelle_pairs)))
chunks = list(zip(a,b))


assert len(alelle_pairs) == len(df_map)
print(len(alelle_pairs))

pd.DataFrame(chunks).\
    to_csv("recode_chunks.txt",
    sep=" ", header=0,index=False)
