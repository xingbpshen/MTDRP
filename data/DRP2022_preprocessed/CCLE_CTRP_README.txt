CCLE/CTRP DATA

FILTERING AND PREPROCESSING STEPS

1. CCLE gene expression taken from DepMap (CCLE_22Q2), this is already log2(TPM+1)
2. Convert gene symbols to ensembl_gene_id (GeneMunge python library)
3. Filter/process gene expressions in the following order
	- if no conversion in #2, drop the gene
	- drop genes with nan values
	- for genes with the same ensembl_gene_id, get the average
	- drop genes that are expressed (log2(TPM+1) > 0) for at least 10% of the samples
	- drop genes that have stdev < 0.1
	- GENE EXPRESSION: depmap/ccle_log2tpm.csv

3. Load CTRPv2.1 (CTRPv2.1_2016_pub_NatChemBiol_12_109)
	- Map ccl_name (CTRP) to DepMap's stripped_cell_line_name, then to broad_id (broad_id is used in GEx)

4. Calculate drug features using RDKit (SMILES from: CTRPv2.1_2016_pub_NatChemBiol_12_109/v21.meta.per_compound.txt)
	- For drugs with identical drug features, keep the drug with more samples
	- MORGAN FP: drug_features/ctrp_morgan.csv
	- DESCRIPTORS: drug_features/ctrp_drug_descriptors.csv

5. Split data into 5-folds using LCO and LPO
	- make sure all drugs are in all folds
	- for ccls without GEx data, fold = -1
