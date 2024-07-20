# setup.py
from setuptools import setup, find_packages

setup(
    name='mlkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',      # Ensures numpy version 1.18.0 or higher
        'pandas>=1.0.0',      # Ensures pandas version 1.0.0 or higher
        'scikit-learn>=0.22', # Ensures scikit-learn version 0.22 or higher
        'matplotlib>=3.1.0',  # Ensures matplotlib version 3.1.0 or higher
    ],  # Add any dependencies here
    description='A custom machine learning toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    url='',  # Replace with your actual URL
)