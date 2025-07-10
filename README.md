# Phylo_Inference

This project uses:
- **Python** 3.12.3 with package versions listed in `requirements.txt`
- **R** version 4.4.3 with packages listed in `r_packages_with_versions.txt`

## Workflow

1. **Generate the phylogeny**

   Run the file `01_generate-phylogeny.R` to generate the phylogeny.  
   You can choose between the **BiSSE** and **CRBD** models.  
   Make sure to adapt the script according to the comments in the file to select the appropriate model.

2. **Convert the phylogeny**

   Use the script `02_convert-phylogeny.R` to convert the generated phylogeny into the desired input format.  
   Additionally, **edit `convert-phylo-to-graph.R`**, specifically within the `get_node_df` function:
   - Change `name.attr` if you are using the **BiSSE** model.

3. **Run neural network inference**

   You will find different architectures in the folders `CRBD/` and `BiSSE/`, depending on the model you used.  
   Load the corresponding **checkpoints** for each architecture as needed.

