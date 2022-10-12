import pandas as pd
import sys
import os
import itertools
import numpy as np

inp_bim = sys.argv[1]  # "data/hapmap_CEU_r23a.bim"
if len(sys.argv) > 2 and sys.argv[2] == "hapmap3":
    inp_ped = os.path.split(inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + ".ped"
    inp_map = os.path.split(
        inp_bim)[0] + "/" +  os.path.split(inp_bim)[-1].split(".bim")[0] + ".map"
else:
    inp_ped = os.path.split(
        inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + "_tmp.ped"
    inp_map = os.path.split(inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + "_tmp.map"
print(inp_bim)
print(inp_ped)
print(inp_map)

snps_23me = list(
    pd.read_csv("23andme_snps_2018.txt", header=3, delimiter="\t")['snp'])
print(len(snps_23me))

if len(sys.argv) > 3:
    print(f"will use {sys.argv[3]} to filter further SNPs from 23andme")
    bad_snps = list(pd.read_csv(sys.argv[3], header=None)[0])
    snps_23me = list(set(snps_23me) - set(bad_snps))
    print(len(snps_23me))

print("working on bim...")
col_names = ["code", "variant_id", "position", "base_pair_coord",
             "allele1", "allele2"]
df_bim = pd.read_csv(inp_bim, delimiter="\t", header=None,
    names=col_names)
#print(len(df_bim))
df_bim = df_bim[df_bim["variant_id"].isin(snps_23me)]
#print(len(df_bim))
out = os.path.split(
    inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + "_filtered.bim"
df_bim.to_csv(out, sep="\t", header=False, index=False)
print(out)

# df_ped.drop(df_ped.columns[to_exclude_ped], axis=1).to_csv("a.csv", header=False, index=False)


print("working on ped...")
df_ped = pd.read_csv(inp_ped, sep=None, header=None)
# ped works together with map file
df_map = pd.read_csv(inp_map, sep="\t", header=None)
df_map['index'] = range(0, len(df_map))
to_exclude = list(df_map[~df_map[1].isin(snps_23me)]['index'])
# NOTE assumes order is same as in map file
to_exclude_ped = [(t*2, t*2+1) for t in to_exclude]
to_exclude_ped = list(np.array(list(itertools.chain(*to_exclude_ped)))+6)

df_ped_filtered = df_ped.drop(df_ped.columns[to_exclude_ped], axis=1)\
    .to_csv(os.path.split(
        inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + "_filtered.ped",
            sep=" ", header=False, index=False)

print("working on map...")
df_map.drop('index', axis=1, inplace=True)
df_map = df_map[df_map[1].isin(snps_23me)]
df_map.to_csv(os.path.split(
    inp_bim)[0] + "/" + os.path.split(inp_bim)[-1].split(".bim")[0] + "_filtered.map",
              sep="\t", header=False, index=False)

#df_bim.drop(df_bim.columns[[0, 1]], axis=1)




