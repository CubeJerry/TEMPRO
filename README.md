# TEMPRO
Nanobody Melting Temperature Prediction using Protein Embeddings


Protein language models in current literature are of growing interest in contexts of protein stability changes, protein sequencing/scaffold filling, and other biophysical property predictions. Recently, protein sequence's numerical representations or embeddings have been used for protein engineering, such as predicted solubility, thermophilicity, and other property prediction or classification tasks. We present a method estimating a nanobody’s melting temperature using their protein structure embeddings as predictive features of thermostability.


**Files:**

1) embedding_generator.zip - contains the code for generating embeddings using ESM and fasta file of all 567 nanobodies used in the study (we cannot upload our own embeddings here as the output csv files are too large).
   
2) paper_results.zip - contains the codes and necessary datasets to reproduce our results from the paper.
   
3) user.zip - contains our pre-trained models for predicting your own nanobody thermostability.



**Instructions:**

To predict your own nanobody melting temperature using its protein sequence:

1) Please install the required packages using the requirements.txt file such as:
   
   'conda install -r requirements.txt'
   
   or
   
   'pip install -r requirements.txt.'
   
3) Navigate to the user folder, open "TEMPRO.ipynb" Jupyter Notebook, and run the codes.


Upon use, please cite our paper:


Notes for reproducing paper results (inside paper_results.zip file):
1) "sdab_data.xlsx" is the processed dataset containing all 567 single-domain antibodies used for the analyses.
2) "raw_nbthermo_json_data.csv" contains the raw sdab data from NbThermo database.
3) "raw_nrl_data.xlsx" contains the raw sdab data manually curated by the authors.

