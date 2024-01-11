from setuptools import setup, Extension, find_packages
from codecs import open
from os import path
import warnings

package_name = 'teomim'
example_dir = 'examples/'
bin_dir = 'bin/'
example_data_dir = example_dir + 'examples_data/'

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    author='paraknowledge corp',
    author_email='research@paraknowledge.ai',
    version = str(version['__version__']),
    packages=find_packages(),
    package_data={'teomim': ['assets/*']},
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/teomim',
    license='LICENSE',
    description='Digital twin for generating and analyzing medical histories with deep comorbidity structures',
    keywords=[
        'machine learning', 
        'statistics'],
    download_url='https://github.com/zeroknowledgediscovery/teomim/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "scikit-learn", 
        "scipy", 
        "numpy",  
        "pandas",
        "quasinet",
        "scipy"],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"],
    include_package_data=True,
    )
