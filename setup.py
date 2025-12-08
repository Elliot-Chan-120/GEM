from setuptools import setup, find_packages

setup(
    name = "GEM",
    version="3.1.0",
    author="Elliot Chan",
    author_email="e224chan@uwaterloo.ca / elliotchan119@gmail.com",
    packages = find_packages(),
    entry_points={
        "console_scripts": [
            "gem=gem_cli:entry_point",
        ],
    },
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "biopython",
        "tqdm",
        "optuna",
        "pyfaidx",
        "pyyaml",
    ]
)
