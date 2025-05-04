from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), 
          encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="SynthCtrl",
    version="0.1.1",
    description="A library for Synthetic Control methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yaroslav Rogoza",
    author_email="r.yaroslav1w@gmail.com",
    url="https://github.com/123yaroslav/SynthCtrl",
    project_urls={
        "Bug Tracker": "https://github.com/123yaroslav/SynthCtrl/issues",
        "Documentation": "https://github.com/123yaroslav/SynthCtrl",
        "Source Code": "https://github.com/123yaroslav/SynthCtrl",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "joblib>=1.1.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
    license="MIT",
    keywords="synthetic control, causal inference, econometrics, statistics, difference-in-differences",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
) 