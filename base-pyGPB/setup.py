from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'pyGPB Python Package'
LONG_DESCRIPTION = 'Python package for dealing with Generalised Poisson Binomial random variables'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pyGPB", 
        version=VERSION,
        author="Mark Rotchell",
        author_email="<MarkRotchell@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numba','numpy','scipy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'Generalised Poisson Binomial Distribution', 'GPB'],
        classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Data Scientists",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)