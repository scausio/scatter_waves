# scatter_waves

Validation of WAVEWATCH III significant wave height against multi-mission
altimeter L3 data (CMEMS): collocation, statistics and plots for one or more
experiments.

## Run

```bash
python main.py -c conf.yaml -s 20220101 -e 20220131            # whole pipeline
python main.py -c conf.yaml -s 20220101 -e 20220131 --steps scatter map
python main.py -c conf.yaml -s 20220101 -e 20220131 -w 8       # read satellite files with 8 processes
```

Steps: `sat`, `model`, `scatter`, `map`, `timeseries`, `tracks`.
`-w` defaults to the number of cores allocated by LSF (`LSB_DJOB_NUMPROC`), else 1.
On Juno the parallel run needs a parallel queue, e.g.
`bsub -q p_short -n 8 -R "span[hosts=1]" python main.py ... `.

## Pipeline

| step | module | output in `out_dir` |
|------|--------|--------------------|
| sat | `satellite_preprocessing.py` | `{start}_{end}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc` |
| model | `model_preprocessing.py` | `{start}_{end}_{experiment}_series.nc` (cache), `{start}_{end}_series.nc` |
| scatter | `validation.py` | scatter plots, stats tables, `metrics_*.csv`, Taylor diagram, errors by Hs class |
| map | `validation_map.py` | `spatial_metrics_*.nc`, `plots/map_*` |
| timeseries | `validation_timeseries.py` | `plots/timeseries*`, `timeseries_metrics_*.csv` |
| tracks | `validation_tracks.py` | `plots/tracks_*` |

**Satellite preprocessing** reads every L3 file of each mission and day, keeps the
observations inside the bounding box and of that day (the filename glob also
matches files ending at 00:00 of the next day), applies the quality / land
variable filters, the shapefile sea mask and a global z-score, and stores all
missions in one file with a `satellite` variable. The file is reused only if it
was produced with the same settings.

**Model preprocessing** collocates every experiment on the satellite
observations. Spatial weights are computed once per mesh; each daily file is
read once. `model_preproc.interpolation`:

- `space: nearest` nearest node within `max_distance_in_space` (degrees);
  `space: linear` barycentric interpolation in the mesh triangles (unstructured
  models). With `nearest` and a 0.01° threshold, observations far from a node
  (offshore, where the mesh is coarser) are discarded.
- `time: nearest` nearest output within `max_distance_in_time` (hours);
  `time: linear` interpolation between the two bracketing outputs, also across
  daily files (the next day file is read for the last hour of the period).

Without the `interpolation` block the behaviour is nearest/nearest as in the
previous versions.

**Plots** use the same sample, built in `pairs.py`: pairs where the satellite and
all experiments are valid, within `filters.min/max`, and not flagged by
`filters.ntimes` (`|model - sat| > ntimes * model`). `filters.unbias: True`
removes the mean bias of each experiment.

## Other data sources

**Satellites from CMEMS sparse tables** (e.g. the ESA CCI Sea State product
`cci_obs-wave_glo_phy-swh_my_l3_PT1S-i`, downloaded with `copernicusmarine subset
--file-format parquet`):

```yaml
sat_preproc:
  sat_names: ['jason-1', 'envisat']     # platform_id values to keep
  sat_specifics: {type: 'cmems_parquet', hs: 'swh_adjusted'}
  paths: {sat: '/path/cci_med_2007*.parquet', out_dir: ...}
  processing:
    filters:
      min_distance_to_coast_km: 0       # uses the distance_to_coast of the product
```

`distance_to_coast` is kept in the series file and the scatter step adds the
errors by distance-to-coast class.

**Model output as one dataset** (`type: series`, zarr or netCDF with hs(time, node)),
e.g. the WaveGraph test set, and **forecasts** (`type: forecast`, files
hs(init_time, pred_step, node)): one experiment per lead, `<name>_+<lead>h`,
compared with the forecast issued at the latest init with lead in
(lead - init spacing, lead].

```yaml
models:
  testset:
    type: series
    path: /path/processed/fields/swh.zarr
    hs: swh
    time: time
    lat: lat
    lon: lon
    mesh: /path/mesh.gr3                # triangles for linear interpolation
  wavegraph:
    type: forecast
    path: /path/predictions/test_set/wav_*.nc
    hs: swh
    coords: /path/processed/fields/swh.zarr   # float lat/lon of the nodes
    lat: lat
    lon: lon
    mesh: /path/mesh.gr3
    step_hours: 3
    leads: [24, 72, 120, 240]
```

Without `plot.experiments` all the experiments of `plot.series` (default
`{out_dir}/{date}_series.nc`) are validated on their common sample; with forecasts
the scatter step also plots the skill against forecast lead.

## Configuration (plot section)

```yaml
plot:
  title: '{start_date} - {end_date}'
  out_dir: *out
  coast_resolution: 'i'
  binning_resolution: 0.05     # map cell [deg]
  min_obs_per_cell: 10         # map cells with fewer pairs are masked
  timeseries_freq: 'D'         # 'D', 'W', 'MS', ...
  daily_tracks: False
  experiments:
    my_exp:
      series: '{out_dir}/{date}_series.nc'
  filters:
    ntimes: 3
    min: 0.15
    max: 20
    unbias: False
    percentile_thresholds: [90, 95]   # global percentiles of the observations
```

See `conf.yaml` for the full example.

## Metrics

`stats.summary` (scalar) and `stats.grouped_metrics` (per cell / day / Hs class):
bias, RMSE, MAE, NBIAS = Σ(m-o)/Σo, NRMSE = sqrt(Σ(m-o)²/Σo²), scatter index
(unbiased, Mentaschi et al. 2013), correlation, R² (squared correlation), Murphy
skill score, symmetric slope, means and standard deviations.

Percentile plots select pairs by the **observed** value (obs ≥ Pxx): this
conditioning makes the model look negatively biased at high values even for an
unbiased model with random errors; compare with the Q-Q plot and the errors by
Hs class.

## Environment

`setup/environment.yml`; on Juno the `work311dev` environment of `ww3_cst-dev`.
Needs xarray, netCDF4, numpy, scipy, pandas, matplotlib, basemap, geopandas,
shapely ≥ 2, munch, pyyaml.
