# Theory aware Machine Learning (TaML)

This repository contains code to incorporate imperfect theory into machine learning for improved prediction and explainability. Specifically, it focuses on the case study of the dimensions of a polymer chain in different solvent qualities. For machine learning methods, three methods are considered: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest. 

This code is provided to supplement a manuscript to be published. For those wishing to incorporate key ideas from the manuscript including incorporating theory and using Gaussian Process Regression with heteroscedastic noise, we suggest starting with `MethodComparision_GPR_HeteroscedasticNoise` in the notebook folder. However, all the code needed to reproduce all of the figures is provided.

## Running the code

All code Python based and can be used on any operating system.

First clone the code via

```bash
git clone https://github.com/usnistgov/TaML.git
```

Create a Python virtual enviroment

```bash
python3 -m TaML env
```

where `TaML` is the name of the virtual enviroment and `env` in the location of the virtual enviroment

Activate the virtual enviroment

```bash
source env/bin/activate
```

Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Notebooks

Included notebooks include `DataVisualization` for visualizing the input data used for machine learning, `MethodComparision_GPR_HeteroscedasticNoise` for comparing different methods for incorporating theory into machine learning using Gaussian Process Regression with heteroscedastic noise, `MethodComparision_GPR_HomoscedasticNoise` for comparing different methods for incorporating theory into machine learning using Gaussian Process Regression with homoscedastic noise, and `ViewResults` for plotting the relative performance of different methods for incorporating theory into machine learning for three different machine learning models.

For users interested in testing ideas, we recommend focusing on the `MethodComparison_GPR_HeteroscedasticNoise` notebook as it explores the different methods and takes into account the known uncertainities in the input data. 

To run the Juypter notebooks, navigate to the notebook folder and run

```bash
jupyter notebook
```

## Source code

The source code compares a variety of methods for incoporating theory into machine learning for three different machine learning models: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest. The output of the files can be plotted by modifying the notebook title `ViewResults` such that the data files are pulled from a local run as opposed to the stored data.

To run the source code, navigate to the src folder and run

```bash
python3 main.py
```

## Contact

Debra J. Audus, PhD  
Polymer Analytics Project  
Materials Science and Engineering Division  
Material Measurement Laboratory  
National Institute of Standards and Technology  

Email: debra.audus@nist.gov  
GithubID: @debraaudus  
Project website: https://www.nist.gov/programs-projects/polymer-analytics  
Staff website: https://www.nist.gov/people/debra-audus  

## How to cite

Please check back later. This will be updated once the accompanying manuscipt is published.
