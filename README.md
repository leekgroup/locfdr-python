locfdr-python v0.1
==================

Python variant of Efron, Turnbull, and Narasimhan's R function locfdr() v1.1-7, which computes local false discovery rates. See http://cran.r-project.org/web/packages/locfdr/ for more information.

This port is very literal interpretation of the augmented version of locfdr at https://github.com/alyssafrazee/derfinder/blob/master/R/locfdrFit.R : the organization of the code parallels the original where efficient, and variable and function names are the same as in R. Rfunctions.py contains Python functions designed to mimic R's idiosyncratic implementations of spline and polynomial interpolation.

Requirements: latest versions of scipy, numpy, matplotlib, pandas, and statsmodels as of September 14, 2013. The code was tested on Anaconda, which contains these packages: https://store.continuum.io/cshop/anaconda/.

Installation: (will use distutils)

Usage:

from locfdr import locfdr

\# Initialize data here, as described in http://cran.r-project.org/web/packages/locfdr/locfdr.pdf .

locfdr(zz, bre=120, df=7, pct=0, pct0=1/4, nulltype=1, type=0, plot=1, mult, mlests, main=" ", sw=0)
