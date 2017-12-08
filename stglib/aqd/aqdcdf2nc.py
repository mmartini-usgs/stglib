from __future__ import division, print_function

import xarray as xr
from ..core import utils
from . import qaqc

def cdf_to_nc(cdf_filename, atmpres=False):
    """
    Load a "raw" .cdf file and generate a processed .nc file
    """

    # Load raw .cdf data
    VEL = qaqc.load_cdf(cdf_filename, atmpres=atmpres)

    # Clip data to in/out water times or via good_ens
    VEL = utils.clip_ds(VEL)

    # Create water_depth variables
    VEL = utils.create_water_depth(VEL)

    # Create depth variable depending on orientation
    VEL, T = qaqc.set_orientation(VEL, VEL['TransMatrix'].values)

    # Transform coordinates from, most likely, BEAM to ENU
    u, v, w = qaqc.coord_transform(VEL['VEL1'].values, VEL['VEL2'].values, VEL['VEL3'].values,
        VEL['Heading'].values, VEL['Pitch'].values, VEL['Roll'].values, T, VEL.attrs['AQDCoordinateSystem'])

    VEL['U'] = xr.DataArray(u, dims=('time', 'bindist'))
    VEL['V'] = xr.DataArray(v, dims=('time', 'bindist'))
    VEL['W'] = xr.DataArray(w, dims=('time', 'bindist'))

    VEL = qaqc.magvar_correct(VEL)

    VEL['AGC'] = (VEL['AMP1'] + VEL['AMP2'] + VEL['AMP3']) / 3

    VEL = qaqc.trim_vel(VEL)

    VEL = qaqc.make_bin_depth(VEL)

    # Reshape and associate dimensions with lat/lon
    for var in ['U', 'V', 'W', 'AGC', 'Pressure', 'Temperature', 'Heading', 'Pitch', 'Roll']:
        VEL = da_reshape(VEL, var)

    # swap_dims from bindist to depth
    VEL = ds_swap_dims(VEL)

    VEL = qaqc.ds_rename(VEL)

    VEL = ds_drop(VEL)

    VEL = qaqc.ds_add_attrs(VEL)

    VEL = utils.add_min_max(VEL)

    VEL = qaqc.add_delta_t(VEL)

    VEL = utils.add_start_stop_time(VEL)

    VEL = utils.add_epic_history(VEL)

    nc_filename = VEL.attrs['filename'] + '.nc'

    VEL.to_netcdf(nc_filename, unlimited_dims='time')
    print('Done writing netCDF file', nc_filename)

    # rename time variables after the fact to conform with EPIC/CMG standards
    utils.rename_time(nc_filename)

    print('Renamed dimensions')

    return VEL


# TODO: add analog input variables (OBS, NTU, etc)


def ds_swap_dims(ds):

    ds = ds.swap_dims({'bindist': 'depth'})

    # need to swap dims and then reassign bindist to be a normal variable (no longer a coordinate)
    valbak = ds['bindist'].values
    ds = ds.drop('bindist')
    ds['bindist'] = xr.DataArray(valbak, dims='depth')

    return ds


def da_reshape(ds, var, waves=False):
    """
    Add lon and lat dimensions to DataArrays and reshape to conform to our
    standard order
    """

    # Add the dimensions using concat
    ds[var] = xr.concat([ds[var]], dim=ds['lon'])
    ds[var] = xr.concat([ds[var]], dim=ds['lat'])

    # Reshape using transpose depending on shape
    if waves == False:
        if len(ds[var].shape) == 4:
            ds[var] = ds[var].transpose('time', 'lon', 'lat', 'bindist')
        elif len(ds[var].shape) == 3:
            ds[var] = ds[var].transpose('time', 'lon', 'lat')

    return ds


def ds_drop(ds):
    """
    Drop old DataArrays from Dataset that won't make it into the final .nc file
    """

    todrop = ['VEL1',
        'VEL2',
        'VEL3',
        'AMP1',
        'AMP2',
        'AMP3',
        'Battery',
        'TransMatrix',
        'AnalogInput1',
        'AnalogInput2',
        'jd',
        'Depth']

    ds = ds.drop(todrop)

    return ds
