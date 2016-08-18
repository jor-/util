import numpy as np

import logging
logger = logging.getLogger(__name__)


def to_cartesian(points, measure='degree', surface_radius=None):

    logger.debug('Converting {} coordinates to cartesian coordinates with surface radius {}.'.format(measure, surface_radius))

    ## check input
    if measure not in ('degree', 'radian'):
        raise ValueError('Measure has to be "degree" or "radian" but it is {}.'.format(measure))
    if len(points.shape) != 2:
        raise ValueError('Points must have 2 dimensions but its shape is {}.'.format(points.shape))
    if points.shape[1] < 3:
        raise ValueError('Points second dimensions must be at least 3 but it is {}.'.format(points.shape[1]))


    ## unpack points
    (n, m) = points.shape
    longitude = points[:, -3]
    latitude = points[:, -2]
    r = points[:, -1]

    ## check input
    assert np.all(r >= 0)
    if measure == 'degree':
        assert (np.all(longitude >= -180) and np.all(longitude <= 180)) or (np.all(longitude >= 0) and np.all(longitude <= 360))
        assert (np.all(latitude >= -90) and np.all(latitude <= 90))
    else:
        assert (np.all(longitude >= -np.pi) and np.all(longitude <= np.pi)) or (np.all(longitude >= 0) and np.all(longitude <= 2*np.pi))
        assert (np.all(latitude >= -np.pi/2) and np.all(latitude <= np.pi/2))


    ## convert degree to radian
    if measure == 'degree':
        logger.debug('Converting degree to radian.')
        longitude = np.pi * longitude / 180
        latitude = np.pi * latitude / 180

    ## convert radius
    if surface_radius is not None:
        logger.debug('Setting radius to surface radius {} minus radius.'.format(surface_radius))
        r = surface_radius - r

    ## calculate cartesian
    logger.debug('Calulating cartesian coordinates.')

    long_sin = np.sin(longitude)
    long_cos = np.cos(longitude)
    lati_sin = np.sin(latitude)
    lati_cos = np.cos(latitude)

    x = r * lati_cos * long_cos
    y = r * lati_cos * long_sin
    z = r * lati_sin

    ## pack data
    x = x.reshape([n, 1])
    y = y.reshape([n, 1])
    z = z.reshape([n, 1])

    cartesian_points = np.concatenate((points[:,:-3], x, y, z),axis=1)

    return cartesian_points