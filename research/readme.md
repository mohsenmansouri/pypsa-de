snakemake -call all --configfile config/config.public.yaml --cores all

snakemake -call all --configfile config/config_flex.yaml --cores all

snakemake -call all --configfile config/config_gas.yaml --cores all
snakemake -call all --configfile config/config_flex_limit_h2_40.yaml --cores all
snakemake -call all --configfile config/config_flex_limit_h2_20.yaml --cores all

snakemake -call all --configfile config/config_flex_no_limit_h2.yaml --cores all
conda env list
conda env remove --name pypsa-de
conda env create -f envs/environment.yaml
conda env create -f envs/macos-pinned.yaml
conda activate pypsa-de
conda activate mm2-pypsa-de


jupyter nbconvert ppt.ipynb --to html --TagRemovePreprocessor.remove_cell_tags='{"hide"}'