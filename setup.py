"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
"""

import setuptools
import os.path

# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(readme_file, mode='r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    # Name
    name='utillib',

    # Version
    use_scm_version=True,
    
    # Desctiption
    description='util functions',
    long_description=long_description,

    # Homepage
    url='https://github.com/jor-/util',

    # Author
    author='Joscha Reimer',
    author_email='jor@informatik.uni-kiel.de',

    # License
    # license='MIT',

    # Classifiers
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # Development Status
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Intended Audience, Topic
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Licence (should match "license" above)
        #'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.5',scikits
    ],

    # Keywords
    keywords='utility auxiliary functions',

    # Packages to install
    packages=setuptools.find_packages(),

    # Dependencies
    setup_requires=[
        'setuptools>=0.8',
        'pip>=1.4',
        'setuptools_scm',
    ],
    install_requires=[
        'numpy',
    ],
    extras_require={
        'cache': ['cachetools'],
        'options': ['h5py'],
        'sorted_multi_dict': ['blist'],
        'colored_log': ['colorlog'],
        'hdf5': ['h5py'],
        'netcdf': ['netCDF4', 'scipy'],
        'petsc': ['petsc4py'],
        'cholmod': ['scikit-sparse'],
        'scoop': ['scoop'],
        'deap': ['deap'],
        'plot' : ['matplotlib', 'scipy'],
        'sparse' : ['scipy'],
        'interpolate' : ['scipy'],
        'multi_dict_stats' : ['scipy'],
    }
)
