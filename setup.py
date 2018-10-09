# Copyright (C) 2011-2018  Joscha Reimer jor@informatik.uni-kiel.de
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""A setuptools based setup module.
https://packaging.python.org/en/latest/distributing.html
"""

import setuptools
import os.path
import versioneer

# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(readme_file, mode='r', encoding='utf-8') as f:
    long_description = f.read()

# Setup
setuptools.setup(
    # general informations
    name='utillib',
    description='util functions',
    long_description=long_description,
    keywords='utility auxiliary functions',

    url='https://github.com/jor-/util',
    author='Joscha Reimer',
    author_email='jor@informatik.uni-kiel.de',
    license='AGPLv3+',

    classifiers=[
        # Development Status
        'Development Status :: 3 - Alpha',
        # Intended Audience, Topic
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Licence (should match "license" above)
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        # Supported Python versions
        'Programming Language :: Python :: 3',
    ],

    # version
    version=versioneer.get_version().replace('.dev0+', '+dirty.').replace('.post', '.post0.dev'),
    cmdclass=versioneer.get_cmdclass(),

    # packages to install
    packages=setuptools.find_packages(),

    # dependencies
    python_requires='>=3.6',
    setup_requires=[
        'setuptools>=0.8',
        'pip>=1.4',

    ],
    install_requires=[
        'numpy>=1.15',
        'overrides',
    ],
    extras_require={
        'cache': ['cachetools'],
        'hdf5': ['h5py'],
        'options': ['h5py'],
        'sorted_multi_dict': ['blist'],
        'colored_log': ['colorlog'],
        'petsc': ['petsc4py'],
        'multi_dict_stats': ['scipy'],
        'matlab': ['scipy'],
        'market': ['scipy'],
        'optimize': ['scipy'],
        'interpolate': ['scipy>=0.17'],
        'sparse': ['scipy>=0.19'],
        'netcdf': ['scipy'],
        'plot': ['matplotlib', 'pycairo'],
        'scoop': ['scoop'],
        'deap': ['deap'],
    },
)
