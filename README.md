The tool validates WAVEWATCH III model (WW3) in unstructured implementation 
versus multi-satellites mission. Several satellite missions will be merged in one series.
User can subset global multimission satellites by a bounding box, or filtering data according to
distance in space or time between model and observations. A Z-score filter could be applied for outliers.
Satellite can be also filtered by the quality_check variable (if present in the dataset), or providing 
a shapefile of the wanted region.
The final plot include several statistics

# Requirements

**Libraries for Zeus CMCC-HPC**

_utils/setup.sh_

**Python libraries:**

_utils/environment.yml_

You can recreate the conda environment by running:

_conda env create -f environment.yml_

# Required inputs

### **Configuration file** in yaml format
Please see the example file provided (conf_adriatic_iride.yaml)

1) fill the conf.yaml file with your setup:
   The file is organized in 3 blocks:
   - sat_preproc: read by sat_preprocessing.py script
   - model_preproc:: read by model_preprocessing*.py scripts
   - plot: read by validation.py script

2) Submission

_main.py -c <path/to/catalog_file>_
eg: _bsub -P <proj queue> python main.py -c <conf.yaml>_

4) 
5) 
6) submission of the main script
   - bsub -P R000 python main.py -c {your configuration file} 

# Workflow:
   - sat_preprocessing.py to process all years and satellite you need
   - mergeSats.py to merge all years and satellites in a unique dataset 
   - model_preprocessingUnstruct.py to preprocess the model output
   - validation.py to apply some filtering specified in the configuration and plotting the results

The main.py is the master file which drives the submission of the tasks, from satellite preprocessing, to model preprocessing to the plotting.


## Details about configuration file

### sat_preproc:
   
    sat_names: -list-,  name of satellites to process eg ['CFOSAT','Cryosat-2','AltiKa','HY-2B','Jason-3','Sentinel-3A', 'Sentinel-3B']
    years: -list-, list of the year to process

  sat_specifics: 

    lat: -string-, name of latitude coordinate in the satellite dataset, eg 'latitude'
    lon: -string-, name of longitude coordinate in the satellite dataset, eg'longitude'
    time: -string-, name of time coordinate in the satellite dataset, eg'time'
    hs: -string-, name of significant wave height variable in the satellite dataset, eg'VAVH'
    type: -string-, name of processing type of the dataset. This could be used to define the path to the files eg'L3'
 
 paths:

    sat: -string-, path to data. User can use some variable here by the format {variable}. Variable allowed are: satName, year,month,type
    eg '/work/opa/md04916/iride/WAVE_GLO_PHY_SWH_L3_NRT_014_001/{satName}/{year}/{month}/'
    
    output: -string-, output directory

  filenames: For Iride project no need of changing this block

    sat:
      template: 'global_vavh_l3_rt_*_*.nc'
      cicle: '*'
      passage: '*'
      output: '{sat_name}_{year}'
  
  processing:

    boundingBox: this block cut the global dataset in the region
      xmin: 12.40
      xmax: 21.31
      ymin: 38.44
      ymax: 46.34
 
   filters:

      quality_check:
         variable_name: -bool or string-,  Set to False to skip, else set the variable name  eg qual_alt_1hz_swh_ku  
         value: -string or integer-, set the idenfifier for good data
      land_masking:
        variable_name: -bool or string-, Set to False to skip, else set the variable name eg 'surface_type' 
        value: -string or integer-, set the idenfifier for the ocean
        shapefile: -string or bool - shapefile of your valid model domain.  if shapefile: False the tool masks land according to the dataset flag, otherwise it use the ocean shapefile
      threshold:
        min: -float or integer-, discard Hs< min, recommended value 0.25 
        max: -float or integer-,  discard Hs> max
      zscore:
        sigma: 3  -integer-, number of standard deviation to filter out the outliers in the Z-score test


### model_preproc:

  out_dir: -string-, output directory
 
  datasets:

    sat:
     path: -string-, path to the satellite. User can keep the following filename which is parameterized. '/{year}_ALLSAT_threshold_landMasked_qcheck_zscore{sigma}.nc'
    models: # for validation of more exps please repeat the code block 'yourExp1' naming it as you prefer. This will be the name of your outputs
     yourExp1:
       type: -string-, select the model type [unstructured, nemo, cf_compliant ]
       path: -string-, path to the model outputs. you can use * for globbing eg '/work/opa/user/output/iride/adri/ww3*.nc' 
       lat:  -string-, name of latitude coordinate in the model data, eg 'latitude'
       lon: -string-, name of longitude coordinate in the model data, eg 'longitude'
       hs: -string-, name of significant wave height variable in the model data, eg 'hs'
       time: -string-, name of time coordinate in the model data, eg 'time'
 
 filters:

    max_distance_in_space: -float or integer-, (in degree) it discards data if sat is farther this value
    max_distance_in_time: -float or integer- (in hours) it discards data if sat is farther this value
    threshold:
     min: -float or integer-, discard Hs< min, recommended value 0.25 
     max: -float or integer-,  discard Hs> max

### plot:

  title: -string-, plot title eg  'September 2023' 

  experiments: # experiments you want to plot

    yourExp1:

      series: -string-, path to the model_preprocess output. User can use the following name which is parameterized'/{experiment}_{year}_sat_series.nc'

    out_dir: -string-, plot output directory. If the directory does not exists, will be created
  additional_filters:
    ntimes: 1.5 # mask data if satellite hs is > of ntimes * model hs
