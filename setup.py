import os
import json
from setuptools import setup, find_packages

def get_extra_requirements(folder='./extra_requirements'):
    """ Helper function to read in all extra requirement files in the specified
        folder. """
    extra_requirements = {}
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return extra_requirements

    for file in os.listdir(folder):
        if file.endswith('.json'):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as fh:
                requirements = json.load(fh)
                extra_requirements.update(requirements)

    return extra_requirements

req = [
    "scipy>=1.4.1",
    "numpy>=1.18.1",
    "ConfigSpace>=0.4.12",
    "pandas",
    "scikit-learn",
    "openml",
    "matplotlib",
    "pdf2image",
    "torch",
    "torchvision",
    "seaborn",
    "GPy",
    "GPyOpt"
    "Pillow",
    "gym",
    "sobol-seq",
    "xgboost",
    "paramz",
    "matplotlib",
    "emukit",
    "pymoo",
    "jax",
    "tikzplotlib",
    "gplearn",
]

setup(
    name="transopt",
    version="0.0.1",
    author="transopt",
    description="Transfer Optimiztion System",
    long_description="This is a longer description of my package.",
    url="https://github.com/maopl/TransOpt.git",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    license="BSD",
    packages=find_packages(exclude=["hpobench"]),
    install_requires=req,
    extras_require=get_extra_requirements(),
)
