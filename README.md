# skin-condition-gwas-sim

The notebook provides `ancestry.ipynb`  provides data selection for the 1KG cohort. The script `filter_hapmap.py` provides data selection/handling for the HapMap3 cohort.

The notebooks `filter.ipynb` and `filter_vcf.ipynb` provide preprocessing for the HapMap3 and 1KG cohorts, respectively.

The scripts `gen_recode_chunks.py`, `query_variant_info.py`, and `recode.py` are helpers during the preprocessing work, especially when parallelization is needed to achieve speed-ups.

`run_ml.py` provides the script for machine learning model development.

`unsupervised.ipynb` provides the script for unsupervised analysis.
