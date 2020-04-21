# A Python Module for Mobility Determination of Parallel Manipulators Using Screw Theory: `pyScrew4Mobility` module


**This repo is use to host materials for the article published in Journal METAL, Dept. of Mechanical Engineering, Andalas University.**

This repo mainly comprises:
* a `pyScrew4Mobility` module which written in python programming language and using Sympy library.
* a notebook that shows the use of `pyScrew4Mobility` module to obtain the mobility of four parallel manipulators (PMs), i.e. 3-PRRR, 3-PR(Pa)R, 4-PRRU, and 6-UPS. The first three PMs is over-constrained PMs.

**Both are release under BSD 3-Clause "New" or "Revised" License**.

If you find that the `pyScrew4Mobility` module is worth for your research or publications later on, please cites the paper.

## How to use:
This is an initial package for mechanical system using screw theory. Author has a plan to create a python package which based on screw theory. Here, it is shown how to use the `pyScrew4Mobility` module to compute mobility of any parallel manipulators. This module is written on top a python programming language and Sympy (Symbolic Python) library.

1. If you do not have a python (python3) installed on your personal computer, you can download a python distribution from [Anaconda](https://www.anaconda.com/distribution/), you have to download Anaconda3 (Anaconda with Python 3+), that depends on your operating system. Anaconda provides all scientific python stacks (Numpy, Scipy, Matplotlib, **Sympy**, Pandas), specific scientific domain packages such as AstroPy, data science packages, machine learning packages (ScikitLearn) and image processing pacakages (Scikit Image).
2. [Install Anaconda Python/R Distribution](https://docs.anaconda.com/anaconda/install/) according to your operating system.
3. Clone or download (as zip) this repo to a directory in your personal computer.
4. After finished, unzip.
5. **Windows user:** Open Anaconda Powershell Prompt (Anaconda3) or Anaconda Prompt (Anaconda3). **Linux or Mac user** can directly use the terminal.
6. Use the Anaconda Powershell Prompt/Anaconda Prompt/Terminal to access the directory where the materials in this repo downloaded in your personal computer.
7. Afterwards, type `jupyter notebook` or `jupyter lab` and press `Enter` key to launch **jupyter notebook** or **jupyter lab**.
8. Open `Screw Theory for Mobility Determination of PMs.ipynb`. This notebook provides an implementation of the `pyScrew4Mobility` module to compute mobility of four parallel manipulators, i.e.:  3-PRRR, 3-PR(Pa)R, 4-PRRU, and 6-UPS.


