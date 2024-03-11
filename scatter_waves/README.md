1) load zeus modules:
   - source setup.sh
2) create conda environment:
   - conda create -n {your favourite env name} --file spec-file.txt
3) activate environment:
   - source activate {your favourite env name}

4) compile config.yaml file with your setup:
   The file is organized in 3 blocks:
   - sat_preproc: read by sat_preprocessing.py script
   - model_preproc:: read by model_preprocessing*.py scripts
   - plot: read by validation.py script

5) tool running:
   - run python sat_preprocessing.py to process all years and satellite you need
   - run python mergeSats.py to merge all years and satellites in a unique dataset 
   - run python model_preprocessingNEMObased.py to preprocess the model output
   - run python validation.py to plot 
