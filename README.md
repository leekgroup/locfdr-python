locfdr-python v0.1
==================

Python variant of Efron, Turnbull, and Narasimhan's R function locfdr() v1.1-7, which computes local false discovery rates. See http://cran.r-project.org/web/packages/locfdr/ for more information.

This port is very literal interpretation of the augmented version of locfdr at https://github.com/alyssafrazee/derfinder/blob/master/R/locfdrFit.R : the organization of the code parallels the original where efficient, and variable and function names are the same as in R. Rfunctions.py contains Python functions designed to mimic R's idiosyncratic implementations of interpolation.

Requirements: Python 2.7.x, scipy, numpy, matplotlib, pandas, and statsmodels. The code was tested on the latest version of Anaconda as of 9/14/2013, which contains Python 2.7.5, scipy 0.12.0, numpy 1.7.1, pandas 0.12.0, statsmodels 0.5.0, and matplotlib 1.3.0. This distribution is available at https://store.continuum.io/cshop/anaconda/.

THIS SOFTWARE IS LICENSED UNDER THE MIT LICENSE. See LICENSE.txt for more information.

Installation
------------------
Download ZIP (button on right at https://github.com/buci/locfdr-python/), unzip in same directory, and run

    python setup.py install


Usage
-----------------

    from locfdr import locfdr
    # Initialize data here, as described in http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf 
    results = locfdr(zz)

R vs. Python usage
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
