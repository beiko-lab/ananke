#Taken from https://wiki.python.org/moin/Distutils/Tutorial
from setuptools import setup
import ananke

setup(name='ananke',
	version=ananke.__version__,
	description='Ananke: Clustering of time-series marker gene data',
	url='https://github.com/beiko-lab/ananke',
	author='Michael Hall',
        author_email='hallm2533@gmail.com',
	classifiers=['Development Status :: 4 - Beta',
	'Environment :: Console',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: MIT License',
	'Natural Language :: English',
	'Operating System :: POSIX :: Linux',
	'Topic :: Scientific/Engineering :: Bio-Informatics'],
        license='GPL3',
	packages=['ananke'],
	install_requires=['h5py>=2.3.1','numpy>=1.6','scipy>=0.16.1','scikit-learn>=0.16','pandas>=0.17'],
	entry_points = {
        'console_scripts': ['ananke=ananke.ananke:main'],
    },
	zip_safe=False)
