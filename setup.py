

from setuptools import setup,find_packages

setup(
    name='Yes_no_units',
    author='Swadhin Agrawal',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            'my_start=src.start:main',
        ]
    }
)