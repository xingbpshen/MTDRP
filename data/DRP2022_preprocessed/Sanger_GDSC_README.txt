SANGER/GDSC DATA

FILTERING AND PREPROCESSING STEPS

1. Merge GDSC1 and GDSC2 drug screening data (from GDSC website: GDSC1_fitted_dose_response_24Jul22.xlsx, GDSC2_fitted_dose_response_24Jul22.xlsx)
	- For drugs that exist in both GDSC1 and GDSC2, keep only GDSC2 (this is newer)
	- For drugs with multiple DRUG_IDs, keep the DRUG_ID that has more samples

2. Find SMILES encoding of all the drugs (pubchempy)
	- If no SMILES encoding is available, remove the drug

3. Calculate drug features using RDKit
	- For drugs with identical drug features, keep the drug with more samples
	- MORGAN FP: drug_features/gdsc_morgan.csv
	- DESCRIPTORS: drug_features/gdsc_drug_descriptors.csv

4. Load RNASeq TPM from Sanger (data from sanger cell model passports: rnaseq_tpm_20220624.csv)
	- Remove organoid samples, keep only CCLs

5. Convert gene IDs/symbols to ensembl_gene_id (converter from sanger cell model passports: gene_identifiers_20191101.csv)

6. Filter/process gene expressions in the following order
	- if no conversion in #5, drop the gene
	- drop genes with nan values
	- for genes with the same ensembl_gene_id, get the average
	- keep genes that are expressed (TPM>1) for at least 10% of the samples
	- log2(TPM+1)
	- drop genes that have stdev < 0.1
	- GENE EXPRESSION: sanger/sanger_broad_ccl_log2tpm.csv

7. Split data into 5-folds using LCO and LPO
	- make sure all drugs are in all folds
	- for ccls without GEx data, fold = -1
