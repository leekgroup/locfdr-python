locfdr-python v0.1
==================

Python version of Efron, Turnbull, and Narasimhan's R function locfdr() v1.1-7, which computes local false discovery rates. See http://cran.r-project.org/web/packages/locfdr/ for more information.

This port is very literal: the organization of the code parallels the original where efficient, and variable and function names are the same as in R. Rfunctions.py contains Python functions designed to mimic R's idiosyncratic implementations of spline and polynomial interpolation.

Requirements: latest versions of scipy, numpy, matplotlib, pandas, and statsmodels as of September 14, 2013. The code was tested on Anaconda, which contains these packages: https://store.continuum.io/cshop/anaconda/.
