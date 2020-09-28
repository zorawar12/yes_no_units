

from setuptools import setup,find_packages

setup(
    name='Yes_no_units',
    author='Swadhin Agrawal',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            'my_start=start:main',
        ]
    },
    install_requires=[
        'numpy==1.15',
        'matplotlib>=3.0.3'
    ],
    python_requires='==3.5',
)