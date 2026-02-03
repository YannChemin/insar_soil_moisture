"""
Setup script for InSAR Soil Moisture package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name='insar-soil-moisture',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='InSAR-based soil moisture retrieval from Sentinel-1 SAR data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/insar_soil_moisture',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/insar_soil_moisture/issues',
        'Source': 'https://github.com/yourusername/insar_soil_moisture',
        'Documentation': 'https://github.com/yourusername/insar_soil_moisture#readme',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='insar, soil moisture, sentinel-1, sar, remote sensing, interferometry',
    package_dir={'insar_soil_moisture': 'src'},
    packages=['insar_soil_moisture'],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'cartopy>=0.20.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'insar-sm=insar_soil_moisture.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
