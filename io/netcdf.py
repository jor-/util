import numpy as np

import util.logging
logger = util.logging.logger


## netcdf

def load_with_scipy(file, dataname):
    import scipy.io

    """
    Loads data from a netcdf file.

    Parameters
    ----------
    file : string or file-like
        The name of the netcdf file to open.
    dataname : string
        The name of the data to extract from the netcdf file.

    Returns
    -------
    data : ndarray
        The desired data from the netcdf file as ndarray with nan for missing values.
    """

    logger.debug('Loading data {} of netcdf file {}.'.format(dataname, file))

    f = scipy.io.netcdf.netcdf_file(file, 'r')
    data_netcdf = f.variables[dataname]
    data = np.array(data_netcdf.data, copy = True)
    data[data == data_netcdf.missing_value] = np.nan
    f.close()

    return data


def load_with_netcdf4(file, data_name):
    import netCDF4

    logger.debug('Loading data {} of netcdf file {}.'.format(data_name, file))

    nc_file = netCDF4.Dataset(file, 'r')
    data = nc_file.variables[data_name][:]
    nc_file.close()

    return data