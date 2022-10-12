import httplib2
import pandas as pd
import time
import json
import glob
import sys
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

h = httplib2.Http()
tqdm.pandas()

# cohort_dir = "variantinfo"
cohort_dir = "variantinfo_vcf"

if os.path.exists(f"{cohort_dir}/all.json"):
    with open(f"{cohort_dir}/all.json", "r") as f:
        all_json = json.load(f)


# query_variant_info("rs3094315")
def query_variant_info(num, rsid_list):
    # http://myvariant.info/v1/query/
    #url = f'https://myvariant.info/v1/variant/{rsid}?fields=_id'
    rsid_str = ",".join(rsid_list)
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    params = f'q={rsid_str}&scopes=all&fields=_id'
    res, con = h.request('http://myvariant.info/v1/query',
                        'POST', params, headers=headers)
    assert res.status == 200, print(res)
    print(f"{num} response received")
    response_json = json.loads(con)
    #response_json = eval(con.decode())
    if type(response_json) is not list:
        response_json = [response_json]
    print(len(response_json))
    with open(f"{cohort_dir}/ref2alt/{num}.json", "w") as f:
        json.dump(response_json, f)
    return response_json

def query_db():
    os.makedirs(f"{cohort_dir}/ref2alt", exist_ok=True)
    if cohort_dir == "variantinfo":
        fn = "hapmap3_r2_b36_fwd.qc.poly/CEU/hapmap3_r3_b36_fwd.CEU.qc.poly_tmp_filtered.map"
        print(fn)
        df_map = pd.read_csv(fn, sep="\t", header=None)
    elif cohort_dir == "variantinfo_vcf":
        fn = 'IGSR/23andme_rel_meta.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
        print(fn)
        df_map = pd.read_hdf(fn)
    else:
        raise ValueError("unknown")
    df_map['index'] = range(0, len(df_map))

    json_batch_list = []
    steps = list(range(0, len(df_map), 20000))+[len(df_map)]
    for i in tqdm(range(1, len(steps)), total=len(steps)):
        if cohort_dir == "variantinfo":
            query_variant_info(i, list(df_map[1])[steps[i-1]:steps[i]])
        elif cohort_dir == "variantinfo_vcf":
            query_variant_info(i, list(df_map['id'])[steps[i-1]:steps[i]])

def gen_json_from_chunks():
    chunks = glob.glob(f"{cohort_dir}/ref2alt/*.json")
    all_json = []
    for chunk in tqdm(chunks):
        with open(chunk, "r") as f:
            chunk_json = json.load(f)
        for entry in chunk_json:
            assert type(entry) is dict
            all_json.append(entry)

    clean_json = defaultdict(list)
    for entry in all_json:
        if "notfound" in entry:
            clean_json[entry['query']].append(False)
        else:
            clean_json[entry['query']].append(entry['_id'])
    with open(f"{cohort_dir}/all.json", "w") as f:
        json.dump(clean_json, f, indent=4)


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


def gen_exclude_list():
    df_map = pd.read_csv(
        "hapmap3_r2_b36_fwd.qc.poly/CEU/hapmap3_r3_b36_fwd.CEU.qc.poly_tmp_filtered.map",
        sep="\t", header=None)
    df_map['index'] = range(0, len(df_map))

    assert len(all_json.keys()) == len(df_map)
    df_map['ref'], df_map['alt'] = zip(
        *df_map[1].progress_apply(variant_info_from_json))

    pd.concat([df_map[df_map['ref'] == -1],
               df_map[df_map['ref'] == -2],
               df_map[df_map['ref'] != df_map['ref']]])[1]\
        .to_csv("bad_snp_list.txt", index=False, header=False)
    print("wrote bad_snp_list.txt")


assert len(sys.argv) == 2

if sys.argv[1] == "query":
    query_db()
elif sys.argv[1] == "json":
    gen_json_from_chunks()
elif sys.argv[1] == "exclude":
    gen_exclude_list()
else:
    raise ValueError("not implemented")


