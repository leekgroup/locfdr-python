locfdr-python v0.1a
===================

Python variant of Efron, Turnbull, and Narasimhan's R function locfdr() v1.1-7, which computes local false discovery rates. See http://cran.r-project.org/web/packages/locfdr/ for more information.

This port is very literal interpretation of Frazee, Collado-Torres, and Leek's augmented version of locfdr at https://github.com/alyssafrazee/derfinder/blob/master/R/locfdrFit.R : the organization of the code parallels the original where efficient, and variable and function names are the same as in R. Rfunctions.py contains Python functions designed to mimic R's idiosyncratic implementations of interpolation.

Requirements: Python 2.7.x, scipy, numpy, matplotlib, pandas, and statsmodels. The code was tested on the latest version of Anaconda as of 8/12/2014, which contains Python 2.7.7, scipy 0.14.0, numpy 1.8.1, pandas 0.14.0, statsmodels 0.5.0, and matplotlib 1.3.1. This distribution is available at https://store.continuum.io/cshop/anaconda/.

THIS SOFTWARE IS LICENSED UNDER THE GNU GENERAL PUBLIC LICENSE VERSION 2. See COPYING for more information.

Installation
------------------
Download ZIP (button on right at https://github.com/buci/locfdr-python/), unzip in same directory, and run

    python setup.py install


Usage
-----------------

    from locfdr import locfdr
    # Initialize data here, as described in http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf 
    results = locfdr(zz)

R vs. Python
------------------

The port is relatively faithful. Variable names are almost precisely the same; if the original variable name contained a period, that period is replaced by an underscore in the Python. (So 'Cov2.out' in the R is 'Cov2_out' in the Python.)

To access returned values:

(R)

    results = locfdr(...)
    results$fdr
    results$z.2
	
(Corresponding Python)

    results = locfdr(...)
    results['fdr']
    results['z_2']

Some returned values are pandas Series and DataFrames. An introduction to pandas data structures is available at http://pandas.pydata.org/pandas-docs/dev/dsintro.html .

Example usage
------------------
Start the Python interpreter, and enter

    >>> execfile('example.py')

This is the simulation of Section 4 from http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf . Compare with results in R.

More details
-----------------
See the locfdr pydoc, or view the source of locfdr.py .
