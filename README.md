# Theory aware Machine Learning (TaML)

This repository contains code to incorporate imperfect theory into machine learning for improved prediction and explainability. Specifically, it focuses on the case study of the dimensions of a polymer chain in different solvent qualities. For machine learning methods, three methods are considered: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest. 

This code is provided to supplement a manuscript to be published. For those wishing to incorporate key ideas from the manuscript including incorporating theory and using Gaussian Process Regression with heteroscedastic noise, we suggest starting with `MethodComparison_GPR_HeteroscedasticNoise` in the notebook folder. However, all the code needed to reproduce all of the figures is provided.

## Running the code

All code is written in Python and requires Python >= 3.7. It can be used on any operating system. Other requirements are listed in `requirements.txt`.

If you are only interested in running the Jupyter Notebooks in [Google Colab](https://colab.research.google.com/), you can skip ahead to [Notebooks](##Notebooks).

First clone the code via

```bash
git clone https://github.com/usnistgov/TaML.git
```

and navigate to the directory where the repository lives

```bash
cd TaML
```

Next, one needs to create a virtual enviroment. This can be done using Python virtual enviroments or with Anaconda. Both options are listed below.

### Create a Python virtual environment (option 1)

First, make sure you are using Python 3.7 or later.

```bash
python3 -m venv env
```

where `env` is the location of the virtual environment

Activate the virtual environment

```bash
source env/bin/activate
```

Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Create a virtual enviroment with Anaconda (option 2)

First, install [conda](https://www.anaconda.com).

```bash
conda env create -f environment.yml
```

If you are using conda>=4.6, activate the virtual environment via

```bash
conda activate TaML
```

Otherwise, see the [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

GPFlow 2.2.1 is not available on conda channels and must be installed via pip

```bash
pip install gpflow==2.2.1
```

### Install the TaML package

For users who wish to use the source code or import functions, the TaML package can be installed via

```bash
pip install .
```

## Notebooks

Included notebooks include `DataVisualization` for visualizing the input data used for machine learning, `MethodComparison_GPR_HeteroscedasticNoise` for comparing different methods for incorporating theory into machine learning using Gaussian Process Regression with heteroscedastic noise, `MethodComparison_GPR_HomoscedasticNoise` for comparing different methods for incorporating theory into machine learning using Gaussian Process Regression with homoscedastic noise, and `ViewResults` for plotting the relative performance of different methods for incorporating theory into machine learning for three different machine learning models.

### Running notebooks locally (option 1)

For users interested in testing ideas, we recommend focusing on the `MethodComparison_GPR_HeteroscedasticNoise` notebook as it explores the different methods and takes into account the known uncertainties in the input data. 

If you cloned the repository, the Jupyter notebooks can by run by navigating to the notebook folder and using the command

```bash
jupyter notebook
```

### Running notebooks in Google Colab (option 2)

If you are interested in running one or more notebooks in [Google Colab](https://colab.research.google.com/), nativigate to the notebook of interest on the TaML GitHub page, for example, `https://github.com/usnistgov/TaML/blob/main/notebooks/MethodComparison_GPR_HeteroscedasticNoise.ipynb`. Then replace `github.com` with `githubtocolab.com`. This should open the notebook in Google Colab. For the `DataVisualization` and `ViewResults` notebooks, all dependencies are likely available and you should be able to directly run them. For the `MethodComparison_GPR_HeteroscedasticNoise` and `MethodComparison_GPR_HomoscedasticNoise` notebooks, you must install GPFlow. This can be accomplished by 

(1) uncommenting out the code block

```bash
!pip install gpflow==2.2.1
```

(2) executing the code block

(3) restarting the run time enviroment (there should be a button at the bottom of the output for that code block).

Then you can run the notebook as normal.

## Source code

The source code (see the `taml` folder) compares a variety of methods for incorporating theory into machine learning for three different machine learning models: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest. The output of the files can be plotted by modifying the notebook title `ViewResults` such that the data files are pulled from a local run as opposed to the stored data.

To run the source code

```bash
python3 -m taml
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

Please check back later. This will be updated once the accompanying manuscript is published.
