import locfns as lf
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.api import families
from statsmodels.formula.api import glm
import Rfunctions as rf
import warnings

def locfdr(zz, bre = 120, df = 7, pct = 0., pct0 = 1./4, nulltype = 1, type = 0, plot = 1, mult = None, mlests = None, main = ' ', sw = 0):
	"""ignore R match.call()"""
	if len(bre) > 1:
		lo = min(bre)
		up = max(bre)
		bre = len(bre)
	else:
		if len(pct) > 1:
			lo = pct[0]
			up = pct[1]
		else:
			if pct == 0:
				lo = min(zz)
				up = max(zz)
			elif pct < 0:
				med = np.median(zz)
				lo = med + (1 - pct) * (min(zz) - med)
				up = med + (1 - pct) * (max(zz) - med)
			elif pct > 0:
				lo = np.percentile(zz, pct * 100)
				up = np.percentile(zz, (1 - pct) * 100)
	zzz = np.array([max(min(el, up), lo) for el in zz])
	breaks = np.linspace(lo, up, bre)
	x = (breaks[1:] + breaks[0:]) / 2.
	y = np.histogram(zzz, bins = len(breaks) - 1)[0])
	yall = y
	K = len(y)
	N = len(zz)
	if pct > 0:
		y[0] = min(y[0], 1.)
		y[K-1] = min(y[K-1], 1)
	if not type:
		basismatrix = rf.ns(x, df)
		X = np.ones((basismatrix.shape[0], basismatrix.shape[1]+1), dtype=np.float64)
		X[:, 1:] = basismatrix
		f = glm("y ~ basismatrix", data = dict(y=np.matrix(y).transpose(), basismatrix=basismatrix, family=families.Poisson()).fit().fittedvalues
	else:
		basismatrix = rf.poly(x, df)
		X = np.ones((basismatrix.shape[0], basismatrix.shape[1]+1), dtype=np.float64)
		X[:, 1:] = basismatrix
		f = glm("y ~ basismatrix", data = dict(y=np.matrix(y).transpose(), basismatrix=basismatrix, family=families.Poisson()).fit().fittedvalues
	l = np.log(f)
	Fl = f.cumsum()
	Fr = f[::-1].cumsum()
	D = ((y - f) / np.sqrt((f + 1)))[0,:]
	D = sum(power(D[1:(K-2)], 2)) / (K - 2 - df)
	if D > 1.5:
		warnings.warn("f(z) misfit = " + str(round(D,1)) + ". Rerun with larger df.")
	if nulltype == 3:
		fp0 = pd.DataFrame(np.zeros((6,4)), index=['thest', 'theSD', 'mlest', 'mleSD', 'cmest', 'cmeSD'], columns=['delta', 'sigleft', 'p0', 'sigright'])
	else:
		fp0 = pd.DataFrame(np.zeros((6,3)), index=['thest', 'theSD', 'mlest', 'mleSD', 'cmest', 'cmeSD'], columns=['delta', 'sigma', 'p0'])
	fp0.loc['thest'][1] = 1
	xmax = max(l)
	imax = np.where(l==xmax)
	if len(pct) == 1:
		pctup = 1 - pct0
		pctlo = pct0
	else:
		pctlo = pct0[0]
		pctup = pct0[1]
	lo0 = np.percentile(zz, pctlo*100)
	hi0 = np.percentile(zz, pctup*100)
	nx = len(x)
	x0 = np.array([el for el in x if el > lo0 and el < hi0])
	y0 = np.array([el for i,el in enumerate(l) if x[i] > l0 and x[i] < hi0])
	xsubtract = x0 - xmax
	X00 = np.zeros((2, len(xsubtract)))
	if nulltype == 3:
		X00[0,:] = np.power(xsubtract, 2)
		X00[1,:] = [el**el for el in]
