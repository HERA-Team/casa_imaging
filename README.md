# casa_imaging

Scripts for calibration, imaging and source modeling in CASA 5.1.0.
A full calibration and imaging pipeline is provided in `pipelines/skycal_pipe.py`.
See `pipelines/skycal_params.yml` for parameter selections.

Note that CASA version 5.3 is known to have a bug in its `ia.modify` task, and will silently error in the `complist_gleam.py` script.

## Dependencies

Depending on the script you want to use, dependencies will vary. Here we list all of them for completeness.

* `numpy >= 1.14.5`
* `astropy >= 2.0`
* `scipy >= 1.1`
* `healpy >= 1.11`
* `pyuvdata >= 1.3`
* `hera_cal >= 2.0`
* `sklearn >= 0.19`
