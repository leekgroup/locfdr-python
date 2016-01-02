# locfdr.py: computes local false discovery rates
# Part of locfdr-python, http://www.github.com/buci/locfdr-python/
#
# Copyright (C) 2013 Abhinav Nellore (anellore@gmail.com)
# Copyright (C) 2013 Alyssa Frazee, Leonardo Collado-Torres, and Jeffrey Leek
# Copyright (C) 2011 Bradley Efron, Brit B. Turnbull, and Balasubramanian Narasimhan
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License v2 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details
#
# You should have received a copy of the GNU General Public License
# along with this program in the file COPYING. If not, write to 
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA

try:
	import numpy as np
except ImportError:
	print 'numpy is required, but it was not found. locfdr-python was tested on numpy 1.7.1.'
	raise
try:
	from scipy import stats
except ImportError:
	print 'scipy is required, but it was not found. locfdr-python was tested on scipy 0.12.0.'
	raise
try:
	import pandas as pd
except ImportError:
	print 'pandas is required, but it was not found. locfdr-python was tested on pandas 0.12.0.'
	raise
try:
	from statsmodels.api import families
	from statsmodels.formula.api import glm
except ImportError:
	print 'statsmodels is required, but it was not found. locfdr-python was tested on statsmodels 0.5.0.'
	raise
import locfns as lf
import Rfunctions as rf
import warnings as wa
import inspect as it

class Error(Exception):
	"""Base class for exceptions."""
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

class EstimationError(Error):
	"""Exception raised for errors in estimations."""
	pass

class InputError(Error):
	"""Exception raised for errors in input."""
	pass

def locfdr(zz, bre = 120, df = 7, pct = 0., pct0 = 1./4, nulltype = 1, type = 0, plot = 1, mult = None, mlests = None, 
		main = ' ', sw = 0, verbose = True, showplot = True, saveplot = False, saveroot = 'locfdr', saveext = 'pdf', savestamp = False):
	"""Computes local false discovery rates.

	This is Abhinav Nellore's Python implementation of the R function locfdr() v1.1.7, originally written by Bradley Efron, 
	Brit B. Turnbull, and Balasubramanian Narasimhan; and later enhanced by Alyssa Frazee, Leonardo Collado-Torres, and Jeffrey Leek 
	(see https://github.com/alyssafrazee/derfinder/blob/master/R/locfdrFit.R ). It is licensed under the GNU GPL v2.
	See COPYING for more information.
	
	The port is relatively faithful. Variable names are almost precisely the same; if the original variable name contained a period, that
	period is replaced by an underscore here. (So 'Cov2.out' in the R is 'Cov2_out' in the Python.)
	To access returned values:
	(in R)        --- results = locfdr(...)
					  results$fdr
			          results$z.2
	(in Python)   --- results = locfdr(...)
					  results['fdr']
					  results['z_2']
	Some returned values are pandas Series and DataFrames. An introduction to pandas data structures is available at
	http://pandas.pydata.org/pandas-docs/dev/dsintro.html .

	A nearly complete description of arguments and returned values may be found at 
	http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf .

	Additional arguments in this version:
		 verbose: (True or False) --- If True, outputs warnings.
	     showplot: (True or False) --- If True, displays plot. Ignored if plot = 0.
	     saveplot: (True or False) --- If True, saves plot according to constraints specified by saveroot, saveext, and savestamp.
	     							   Ignored if plot = 0.
	     saveroot: (Any string that constitutes a valid filename.) --- Specifies prefix of file to save. Ignored if saveplot = False.
	     saveext: (Most valid image file extensions work here. Try 'png', 'pdf', 'ps', 'eps', or 'svg'.) --- Selects file format and extension.
	     	Ignored if saveplot = False.
	     savestamp: (True or False) --- If True, date/timestamp is appended to filename prefix; this helps prevent overwriting old saves.
	     	Ignored if saveplot = False.

	 Additional returned values in this version:
		yt: Heights of pink histogram bars that appear on the plots (i.e., heights of alt. density's histogram).
		x: Locations of pinkfl histogram bars that appear on the plots (locations of alt. density's histogram).
		mlest_lo AND mlest_hi: If the function outputs a warning message that reads "please rerun with mlest parameters = ...",
			these parameters are contained in mlest_lo and mlest_hi .
		needsfix: 1 if a rerun warning is output; otherwise 0.
		nulldens: y-values of estimated null distribution density.
		nulldens: y-values of estimated full (mixture) density."""
	call = it.stack()
	zz = np.array(zz)
	mlest_lo = None
	mlest_hi = None
	yt = None
	x = None
	needsfix = 0
	try:
		brelength = len(bre)
		lo = min(bre)
		up = max(bre)
		bre = brelength
	except TypeError:
		try:
			len(pct)
			lo = pct[0]
			up = pct[1]
			# the following line is present to mimic how R handles [if (pct > 0)] (see code below) when pct is an array
			pct = pct[0]
		except TypeError:
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
	x = (breaks[1:] + breaks[0:-1]) / 2.
	y = np.histogram(zzz, bins = len(breaks) - 1)[0]
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
		f = glm("y ~ basismatrix", data = dict(y=np.matrix(y).transpose(), basismatrix=basismatrix), 
				family=families.Poisson()).fit().fittedvalues
	else:
		basismatrix = rf.poly(x, df)
		X = np.ones((basismatrix.shape[0], basismatrix.shape[1]+1), dtype=np.float64)
		X[:, 1:] = basismatrix
		f = glm("y ~ basismatrix", data = dict(y=np.matrix(y).transpose(), basismatrix=basismatrix), 
			family=families.Poisson()).fit().fittedvalues
	fulldens = f
	l = np.log(f)
	Fl = f.cumsum()
	Fr = f[::-1].cumsum()
	D = ((y - f) / np.sqrt((f + 1)))
	D = sum(np.power(D[1:(K-1)], 2)) / (K - 2 - df)
	if D > 1.5:
		wa.warn("f(z) misfit = " + str(round(D,1)) + ". Rerun with larger df.")
	if nulltype == 3:
		fp0 = pd.DataFrame(np.zeros((6,4)).fill(np.nan), index=['thest', 'theSD', 'mlest', 'mleSD', 'cmest', 'cmeSD'], 
			columns=['delta', 'sigleft', 'p0', 'sigright'])
	else:
		fp0 = pd.DataFrame(np.zeros((6,3)).fill(np.nan), index=['thest', 'theSD', 'mlest', 'mleSD', 'cmest', 'cmeSD'], 
			columns=['delta', 'sigma', 'p0'])
	fp0.loc['thest'][0:2] = np.array([0,1])
	fp0.loc['theSD'][0:2] = 0
	imax = l.argmax()
	xmax = x[imax]
	try:
		len(pct)
		pctlo = pct0[0]
		pctup = pct0[1]
	except TypeError:
		pctup = 1 - pct0
		pctlo = pct0
	lo0 = np.percentile(zz, pctlo*100)
	hi0 = np.percentile(zz, pctup*100)
	nx = len(x)
	i0 = np.array([i for i, el in enumerate(x) if el > lo0 and el < hi0])
	x0 = np.array([el for el in x if el > lo0 and el < hi0])
	y0 = np.array([el for i,el in enumerate(l) if x[i] > lo0 and x[i] < hi0])
	xsubtract = x0 - xmax
	X00 = np.zeros((2, len(xsubtract)))
	if nulltype == 3:
		X00[0, :] = np.power(xsubtract, 2)
		X00[1, :] = [max(el, 0)*max(el, 0) for el in xsubtract]
	else:
		X00[0, :] = xsubtract
		X00[1, :] = np.power(xsubtract, 2)
	X00 = X00.transpose()
	co = glm("y0 ~ X00", data = dict(y0=y0, X00=X00)).fit().params
	# these errors may not be necessary
	if nulltype == 3 and ((pd.isnull(co[1]) or pd.isnull(co[2])) or (co[1] >= 0 or co[1] + co[2] >= 0)):
			raise EstimationError('CM estimation failed. Rerun with nulltype = 1 or 2.')
	elif pd.isnull(co[2]) or co[2] >= 0:
		if nulltype == 2:
			raise EstimationError('CM estimation failed. Rerun with nulltype = 1.')
		elif nulltype != 3:
			xsubtract2 = x - xmax
			X0 = np.ones((3, len(xsubtract2)))
			X0[1, :] = xsubtract2
			X0[2, :] = np.power(xsubtract2, 2)
			X0 = X0.transpose()
			wa.warn('CM estimation failed; middle of histogram nonnormal')
	else:
		xsubtract2 = x - xmax
		X0 = np.ones((3, len(xsubtract2)))
		if nulltype == 3:
			X0[1, :] = np.power(xsubtract2, 2)
			X0[2, :] = [max(el, 0)*max(el, 0) for el in xsubtract2]
			sigs = np.array([1/np.sqrt(-2*co[1]), 1/np.sqrt(-2*(co[1]+co[2]))])
			fp0.loc['cmest'][0] = xmax
			fp0.loc['cmest'][1] = sigs[0]
			fp0.loc['cmest'][3] = sigs[1]
		else:
			X0[1, :] = xsubtract2
			X0[2, :] = np.power(xsubtract2, 2)
			xmaxx = -co[1] / (2 * co[2]) + xmax
			sighat = 1 / np.sqrt(-2 * co[2])
			fp0.loc['cmest'][[0,1]] = [xmaxx, sighat]
		X0 = X0.transpose()
		l0 = np.array((X0 * np.matrix(co).transpose()).transpose())[0]
		f0 = np.exp(l0)
		p0 = sum(f0) / float(sum(f))
		f0 = f0 / p0
		fp0.loc['cmest'][2] = p0
	b = 4.3 * np.exp(-0.26 * np.log10(N))
	if mlests == None:
		med = np.median(zz)
		sc = (np.percentile(zz, 75) - np.percentile(zz, 25)) / (2 * stats.norm.ppf(.75))
		mlests = lf.locmle(zz, xlim = np.array([med, b * sc]))
		if N > 5e05:
			if verbose:
				wa.warn('length(zz) > 500,000: an interval wider than the optimal one was used for maximum likelihood estimation. To use the optimal interval, rerun with mlests = [' + str(mlests[0]) + ', ' + str(b * mlests[1]) + '].')
			mlest_lo = mlests[0]
			mlest_hi = b * mlests[1]
			needsfix = 1
			mlests = lf.locmle(zz, xlim = [med, sc])
	if not pd.isnull(mlests[0]):
		if N > 5e05:
			b = 1
		if nulltype == 1:
			Cov_in = {'x' : x, 'X' : X, 'f' : f, 'sw' : sw}
			ml_out = lf.locmle(zz, xlim = [mlests[0], b * mlests[1]], d = mlests[0], s = mlests[1], Cov_in = Cov_in)
			mlests = ml_out['mle']
		else:
			mlests = lf.locmle(zz, xlim = [mlests[0], b * mlests[1]], d = mlests[0], s = mlests[1])
		fp0.loc['mlest'][0:3] = mlests[0:3]
		fp0.loc['mleSD'][0:3] = mlests[3:6]
	if (not (pd.isnull(fp0.loc['mlest'][0]) or pd.isnull(fp0.loc['mlest'][1]) or pd.isnull(fp0.loc['cmest'][0]) or pd.isnull(fp0.loc['cmest'][1]))) and nulltype > 1:
		if abs(fp0.loc['cmest'][0] - mlests[0]) > 0.05 or abs(np.log(fp0.loc['cmest'][1] / mlests[1])) > 0.05:
			wa.warn('Discrepancy between central matching and maximum likelihood estimates. Consider rerunning with nulltype = 1.')
	if pd.isnull(mlests[0]):
		if nulltype == 1:
			if pd.isnull(fp0.loc['cmest'][1]):
				raise EstimationError('CM and ML estimation failed; middle of histogram is nonnormal.')
			else:
				raise EstimationError('ML estimation failed. Rerun with nulltype = 2.')
		else:
			wa.warn('ML estimation failed.')
	if nulltype < 2:
		xmaxx = mlests[0]
		xmax = mlests[0]
		delhat = mlests[0]
		sighat = mlests[1]
		p0 = mlests[2]
		f0 = np.array([stats.norm.pdf(el, delhat, sighat) for el in x])
		f0 = (sum(f) * f0) / sum(f0)
	fdr = np.array([min(el, 1) for el in (p0 * (f0 / f))])
	f00 = np.exp(-np.power(x, 2) / 2)
	f00 = (f00 * sum(f)) / sum(f00)
	p0theo = sum(f[i0]) / sum(f00[i0])
	fp0.loc['thest'][2] = p0theo
	fdr0 = np.array([min(el, 1) for el in ((p0theo * f00) / f)])
	f0p = p0 * f0
	if nulltype == 0:
		f0p = p0theo * f00
	F0l = f0p.cumsum()
	F0r = f0p[::-1].cumsum()
	Fdrl = F0l / Fl
	Fdrr = (F0r / Fr)[::-1]
	Int = (1 - fdr) * f * (fdr < 0.9)
	if np.any([x[i] <= xmax and fdr[i] == 1 for i in xrange(len(fdr))]):
		xxlo = min([el for i,el in enumerate(x) if el <= xmax and fdr[i] == 1])
	else:
		xxlo = xmax
	if np.any([x[i] >= xmax and fdr[i] == 1 for i in xrange(len(fdr))]):
		xxhi = max([el for i,el in enumerate(x) if el >= xmax and fdr[i] == 1])
	else:
		xxhi = xmax
	indextest = [i for i,el in enumerate(x) if el >= xxlo and el <= xxhi]
	if len(indextest) > 0:
		fdr[indextest] = 1
	indextest = [i for i,el in enumerate(x) if el <= xmax and fdr0[i] == 1]
	if len(indextest) > 0:
		xxlo = min(x[indextest])
	else:
		xxlo = xmax
	indextest = [i for i,el in enumerate(x) if el >= xmax and fdr0[i] == 1]
	if len(indextest) > 0:
		xxhi = max(x[indextest])
	else:
		xxhi = xmax
	indextest = [i for i,el in enumerate(x) if el >= xxlo and el <= xxhi]
	if len(indextest) > 0:
		fdr0[indextest] = 1
	if nulltype == 1:
		indextest = [i for i,el in enumerate(x) if el >= mlests[0] - mlests[1] and el <= mlests[0] + mlests[1]]
		fdr[indextest] = 1
		fdr0[indextest] = 1
	p1 = sum((1 - fdr) * f) / N
	p1theo = sum((1 - fdr0) * f) / N
	fall = f + (yall - y)
	Efdr = sum((1 - fdr) * fdr * fall) / sum((1 - fdr) * fall)
	Efdrtheo = sum((1 - fdr0) * fdr0 * fall) / sum((1 - fdr0) * fall)
	iup = [i for i,el in enumerate(x) if el >= xmax]
	ido = [i for i,el in enumerate(x) if el <= xmax]
	Eleft = sum((1 - fdr[ido]) * fdr[ido] * fall[ido]) / sum((1 - fdr[ido]) * fall[ido])
	Eleft0 = sum((1 - fdr0[ido]) * fdr0[ido] * fall[ido])/sum((1 - fdr0[ido]) * fall[ido])
	Eright = sum((1 - fdr[iup]) * fdr[iup] * fall[iup])/sum((1 - fdr[iup]) * fall[iup])
	Eright0 = sum((1 - fdr0[iup]) * fdr0[iup] * fall[iup])/sum((1 - fdr0[iup]) * fall[iup])
	Efdr = np.array([Efdr, Eleft, Eright, Efdrtheo, Eleft0, Eright0])
	for i,el in enumerate(Efdr):
		if pd.isnull(el):
			Efdr[i] = 1
	Efdr = pd.Series(Efdr, index=['Efdr', 'Eleft', 'Eright', 'Efdrtheo', 'Eleft0', 'Eright0'])
	if nulltype == 0:
		f1 = (1 - fdr0) * fall
	else:
		f1 = (1 - fdr) * fall
	if mult != None:
		try:
			mul = np.ones(len(mult) + 1)
			mul[1:] = mult
		except TypeError:
			mul = np.array([1, mult])
		EE = np.zeros(len(mul))
		for m in xrange(len(EE)):
			xe = np.sqrt(mul[m]) * x
			f1e = rf.approx(xe, f1, x, rule = 2, ties = 'mean')
			f1e = (f1e * sum(f1)) / sum(f1e)
			f0e = f0
			p0e = p0
			if nulltype == 0:
				f0e = f00
				p0e = p0theo
			fdre = (p0e * f0e) / (p0e * f0e + f1e)
			EE[m] = sum(f1e * fdre) / sum(f1e)
		EE = EE / EE[0]
		EE = pd.Series(EE, index=mult)
	Cov2_out = lf.loccov2(X, X0, i0, f, fp0.loc['cmest'], N)
	Cov0_out = lf.loccov2(X, np.ones((len(x), 1)), i0, f, fp0.loc['thest'], N)
	if sw == 3:
		if nulltype == 0:
			Ilfdr = Cov0_out['Ilfdr']
		elif nulltype == 1:
			Ilfdr = ml_out['Ilfdr']
		elif nulltype == 2:
			Ilfdr = Cov2_out['Ilfdr']
		else:
			raise InputError('When sw = 3, nulltype must be 0, 1, or 2.')
		return Ilfdr
	if nulltype == 0:
		Cov = Cov0_out['Cov']
	elif nulltype == 1:
		Cov = ml_out['Cov_lfdr']
	else:
		Cov = Cov2_out['Cov']
	lfdrse = np.sqrt(np.diag(Cov))
	fp0.loc['cmeSD'][0:3] = Cov2_out.loc['stdev'][[1,2,0]]
	if nulltype == 3:
		fp0.loc['cmeSD'][3] = fp0['cmeSD'][1]
	fp0.loc['theSD'][2] = Cov0_out['stdev'][0]
	if sw == 2:
		if nulltype == 0:
			pds = fp0.loc['thest'][[2, 0, 1]]
			stdev = fp0.loc['theSD'][[2, 0, 1]]
			pds_ = Cov0_out['pds_'].transpose()
		elif nulltype == 1:
			pds = fp0.loc['mlest'][[2, 0, 1]]
			stdev = fp0.loc['mleSD'][[2, 0, 1]]
			pds_ = ml_out['pds_'].transpose()
		elif nulltype == 2:
			pds = fp0.loc['cmest'][[2, 0, 1]]
			stdev = fp0.loc['cmeSD'][[2, 0, 1]]
			pds_ = Cov2_out['pds_'].transpose()
		else:
			raise InputError('When sw = 2, nulltype must equal 0, 1, or 2.')
		pds_ = pd.DataFrame(pds_, columns=['p0', 'delhat', 'sighat'])
		pds = pd.Series(pds, index=['p0', 'delhat', 'sighat'])
 		stdev = pd.Series(stdev, index=['sdp0', 'sddelhat', 'sdsighat'])
		return pd.Series({'pds': pds, 'x': x, 'f': f, 'pds_' : pds_, 'stdev' : stdev})
	p1 = np.arange(0.01, 1, 0.01)
	cdf1 = np.zeros((2,99))
	cdf1[0, :] = p1
	if nulltype == 0:
		fd = fdr0
	else:
		fd = fdr
	for i in xrange(99):
		cdf1[1, i] = sum([el for j,el in enumerate(f1) if fd[j] <= p1[i]])
	cdf1[1, :] = cdf1[1, :] / cdf1[1, -1]
	cdf1 = cdf1.transpose()
	if nulltype != 0:
		mat = pd.DataFrame(np.vstack((x, fdr, Fdrl, Fdrr, f, f0, f00, fdr0, yall, lfdrse, f1)), 
			index=['x', 'fdr', 'Fdrleft', 'Fdrright', 'f', 'f0', 'f0theo', 'fdrtheo', 'counts', 'lfdrse', 'p1f1'])
	else:
		mat = pd.DataFrame(np.vstack((x, fdr, Fdrl, Fdrr, f, f0, f00, fdr0, yall, lfdrse, f1)), 
			index=['x', 'fdr', 'Fdrltheo', 'Fdrrtheo', 'f', 'f0', 'f0theo', 'fdrtheo', 'counts', 'lfdrsetheo', 'p1f1'])
	z_2 = np.array([np.nan, np.nan])
	m = sorted([(i, el) for i, el in enumerate(fd)], key=lambda nn: nn[1])[-1][0]
	if fd[-1] < 0.2:
		z_2[1] = rf.approx(fd[m:], x[m:], 0.2, ties = 'mean')
	if fd[0] < 0.2:
		z_2[0] = rf.approx(fd[0:m], x[0:m], 0.2, ties = 'mean')
	if nulltype == 0:
		nulldens = p0theo * f00
	else:
		nulldens = p0 * f0
	yt = np.array([max(el, 0) for el in (yall * (1 - fd))])
	# construct plots
	if plot > 0:
		try:
			import matplotlib.pyplot as plt
			import matplotlib.patches as patches
			import matplotlib.path as path
		except ImportError:
			print 'matplotlib is required for plotting, but it was not found. Rerun with plot = 0 to turn off plots.'
			print 'locfdr-python was tested on matplotlib 1.3.0.'
			raise
		fig = plt.figure(figsize=(14, 8))
		if plot == 4:
			histplot = fig.add_subplot(131)
			fdrFdrplot = fig.add_subplot(132)
			f1cdfplot = fig.add_subplot(133)
		elif plot == 2 or plot == 3:
			histplot = fig.add_subplot(121)
			if plot == 2:
				fdrFdrplot = fig.add_subplot(122)
			else:
				f1cdfplot = fig.add_subplot(122)
		elif plot == 1:
			histplot = fig.add_subplot(111)
		# construct histogram
		leftplt = breaks[:-1]
		rightplt = breaks[1:]
		bottomplt = np.zeros(len(leftplt))
		topplt = bottomplt + y
		XYplt = np.array([[leftplt,leftplt,rightplt,rightplt], [bottomplt,topplt,topplt,bottomplt]]).transpose()
		barpath = path.Path.make_compound_path_from_polys(XYplt)
		patch = patches.PathPatch(barpath, facecolor='white', edgecolor='#302f2f')
		histplot.add_patch(patch)
		histplot.set_xlim(leftplt[0], rightplt[-1])
		histplot.set_ylim(-1.5, (topplt.max()+1.5) * 0.1 + topplt.max())
		histplot.set_title(main)
		for k in xrange(K):
			histplot.plot([x[k], x[k]], [0, yt[k]], color='#e31d76', linewidth = 2)
		if nulltype == 3:
			histplot.set_xlabel('delta = ' + str(round(xmax, 3)) + ', sigleft = ' + str(round(sigs[0], 3))  
				+ ', sigright = ' + str(round(sigs[1], 3)) + ', p0 = ' + str(round(fp0.loc['cmest'][2], 3)))
		if nulltype == 1 or nulltype == 2:
			histplot.set_xlabel('MLE: delta = ' + str(round(mlests[0], 3)) + ', sigma = ' + str(round(mlests[1], 3))  
				+ ', p0 = ' + str(round(mlests[2], 3)) + '\nCME: delta = ' + str(round(fp0.loc['cmest'][0], 3)) 
				+  ', sigma = ' + str(round(fp0.loc['cmest'][1], 3)) + ', p0 = ' + str(round(fp0.loc['cmest'][2], 3)))
		histplot.set_ylabel('Frequency')
		histplot.plot(x, f, color='#3bbf53', linewidth = 3)
		if nulltype == 0:
			histplot.plot(x, p0theo * f00, linewidth = 3, linestyle = 'dashed', color = 'blue')
		else:
			histplot.plot(x, p0 * f0, linewidth = 3, linestyle = 'dashed', color = 'blue')
		if not pd.isnull(z_2[1]): 
			histplot.plot([z_2[1]], [-0.5], marker = '^', markersize = 16, markeredgecolor = 'red', markeredgewidth = 1.3, color = 'yellow')
		if not pd.isnull(z_2[0]): 
			histplot.plot([z_2[0]], [-0.5], marker = '^', markersize = 16, markeredgecolor = 'red', markeredgewidth = 1.3, color = 'yellow')
		if nulltype == 1 or nulltype == 2:
			Ef = Efdr[0]
		elif nulltype == 0: 
			Ef = Efdr[3]
		# construct fdr + Fdr plot
		if plot == 2 or plot == 4:
			if nulltype == 0:
				fdd = fdr0
			else:
				fdd = fdr
			fdrFdrplot.plot(x, fdd, linewidth = 3, color = 'black')
			fdrFdrplot.plot(x, Fdrl, linewidth = 3, color = 'red', linestyle = 'dashed')
			fdrFdrplot.plot(x, Fdrr, linewidth = 3, color = 'green', linestyle = 'dashed')
			fdrFdrplot.set_ylim(-0.05, 1.1)
			fdrFdrplot.set_title('fdr (solid); Fdr\'s (dashed)')
			fdrFdrplot.set_xlabel('Efdr = ' + str(round(Ef, 3)))
			fdrFdrplot.set_ylabel('fdd (black), Fdrl (red), and Fdrr (green)')
			fdrFdrplot.plot([0, 0], [0, 1], linestyle = 'dotted', color = 'red')
			fdrFdrplot.axhline(linestyle = 'dotted', color = 'red')
		# construct plot of f1 cdf of estimated fdr curve
		if plot == 3 or plot == 4:
			if sum([1 for el in cdf1[:, 1] if pd.isnull(el)]) == cdf1.shape[0]:
				wa.warning('cdf1 is not available.')
			else:
				f1cdfplot.plot(cdf1[:, 0], cdf1[:, 1], linewidth = 3, color = 'black')
				f1cdfplot.set_xlabel('fdr level\nEfdr = ' + str(round(Ef, 3)))
				f1cdfplot.set_ylabel('f1 proportion < fdr level')
				f1cdfplot.set_title('f1 cdf of estimated fdr')
				f1cdfplot.set_ylim(0, 1)
				f1cdfplot.plot([0.2, 0.2], [0, cdf1[19, 1]], color = 'blue', linestyle = 'dashed')
				f1cdfplot.plot([0, 0.2], [cdf1[19, 1], cdf1[19, 1]], color = 'blue', linestyle = 'dashed')
				f1cdfplot.text(0.05, cdf1[19, 1], str(round(cdf1[19, 1], 2)))
		if saveplot:
			if savestamp:
				import time, datetime
				plt.savefig(saveroot + '_' + '-'.join(str(el) for el in list(tuple(datetime.datetime.now().timetuple())[:6])) + '.' + saveext)
			else:
				plt.savefig(saveroot + '.' + saveext)
		if showplot:
			plt.show()
	if nulltype == 0:
		ffdr = rf.approx(x, fdr0, zz, rule = 2, ties = 'ordered')
	else:
		ffdr = rf.approx(x, fdr, zz, rule = 2, ties = 'ordered')
	if mult != None:
		return {'fdr' : ffdr, 'fp0' : fp0, 'Efdr' : Efdr, 'cdf1' : cdf1, 'mat' : mat, 'z_2' : z_2, 'yt' : yt, 'call' : call, 'x' : x, 'mlest_lo' : mlest_lo, 'mlest_hi' : mlest_hi, 'needsfix' : needsfix, 'nulldens' : nulldens, 'fulldens' : fulldens, 'mult' : EE}
	return {'fdr' : ffdr, 'fp0' : fp0, 'Efdr' : Efdr, 'cdf1' : cdf1, 'mat' : mat, 'z_2' : z_2, 'yt' : yt, 'call' : call, 'x' : x, 'mlest_lo' : mlest_lo, 'mlest_hi' : mlest_hi, 'needsfix' : needsfix, 'nulldens' : nulldens, 'fulldens' : fulldens}
