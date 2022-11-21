"""
Theory aware Machine Learning (TaML)
"""
from setuptools import setup

requirements = [
    "gpflow==2.2.1",
    "matplotlib>=3.5",
    "notebook",
    "numpy",
    "pandas>=1.0",
    "scikit-learn>=1.0",
    "scipy",
    "seaborn",
    "tensorflow==2.9.3",
]

readme_txt = open('README.md').read()

setup(name="taml",
      version="0.1",
      description="This repository contains code to incorporate imperfect theory into machine learning for improved prediction and explainability. Specifically, it focuses on the case study of the dimensions of a polymer chain in different solvent qualities. For machine learning methods, three methods are considered: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest.",
      long_description=readme_txt,
      author="Debra J. Audus",
      author_email="debra.audus@nist.gov",
      packages=["taml"],
      python_requires=">=3.7",
      install_requires=requirements,
      url="https://github.com/usnistgov/TaML"
     )
