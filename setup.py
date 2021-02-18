from setuptools import setup,find_packages

setup(
    name='Yes_no_units',
    author='Swadhin Agrawal',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.15',
        'matplotlib>=3.0'
        'statsmodels==0.9.0'
    ],
    python_requires='==3.5.6',
)
