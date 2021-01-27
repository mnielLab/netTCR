# netTCR

# Note, the repository is outdated. Please refrain from using, and use netTCR-2.0 instead.
predicting peptide and TCR interaction

`python scripts/netTCR.py -infile test_data/data.txt -outfile NetTCR_predictions.txt`

or with TCR list and peptide spcification:

`python scripts/netTCR.py -infile test_data/data_tcr_list.txt -peptides GLCTLVAML,NLVPMVATV -outfile NetTCR_predictions.txt`

the options for peptides should be:

GILGFVFTL, GLCTLVAML, NLVPMVATV and YVLDHLIVV. 
