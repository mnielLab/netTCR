# netTCR

# Note, the repository is outdated. Please refrain from using, and refer to https://github.com/mnielLab/NetTCR-2.0_data for access to updated data sets and https://services.healthtech.dtu.dk/service.php?NetTCR-2.0 for the updated prediction model.
predicting peptide and TCR interaction

`python scripts/netTCR.py -infile test_data/data.txt -outfile NetTCR_predictions.txt`

or with TCR list and peptide spcification:

`python scripts/netTCR.py -infile test_data/data_tcr_list.txt -peptides GLCTLVAML,NLVPMVATV -outfile NetTCR_predictions.txt`

the options for peptides should be:

GILGFVFTL, GLCTLVAML, NLVPMVATV and YVLDHLIVV. 
