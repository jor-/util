import numpy as np

import util.logging
logger = util.logging.logger


## netcdf

def load_with_scipy(file, data_name):
    import scipy.io

    """
    Loads data from a netcdf file.

    Parameters
    ----------
    file : string or file-like
        The name of the netcdf file to open.
    data_name : string
        The name of the data to extract from the netcdf file.

    Returns
    -------
    data : ndarray
        The desired data from the netcdf file as ndarray with nan for missing values.
    """

    logger.debug('Loading data {} of netcdf file {} with scipy.io.'.format(data_name, file))

    f = scipy.io.netcdf.netcdf_file(file, 'r')
    data_netcdf = f.variables[data_name]
    data = np.array(data_netcdf.data, copy = True)
    data[data == data_netcdf.missing_value] = np.nan
    f.close()

    return data


def load_with_netcdf4(file, data_name):
    import netCDF4

    logger.debug('Loading data {} of netcdf file {} with netCDF4.'.format(data_name, file))

    nc_file = netCDF4.Dataset(file, 'r')
    data = nc_file.variables[data_name][:]
    nc_file.close()

    return data


def load(file, data_name):
    try:
        import netCDF4
    except:
        pass
    else:
        return load_with_netcdf4(file, data_name)
    
    try:
        import scipy.io
    except:
        pass
    else:
        return load_with_scipy(file, data_name)
    
    raise ImportError('Could not import netCDF4 nor scipy.io.')
    