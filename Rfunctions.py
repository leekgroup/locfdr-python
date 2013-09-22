try:
	import numpy as np
except ImportError:
	print 'numpy is required, but it was not found. Rfunctions was tested on numpy 1.7.1.'
	raise
try:
	import scipy as sp
	from scipy.linalg import qr
	from scipy.interpolate import splev
except ImportError:
	print 'scipy is required, but it was not found. Rfunctions was tested on scipy 0.12.0.'
	raise

class InputError(Exception):
    """Exception raised for errors in the input."""
    def __init__(self, value):
        self.value = value
	def __str__(self):
		return repr(self.value)

def splineDesign(knots, x, ord = 4, der = 0):
	"""Reproduces behavior of R function splineDesign() for use by ns(). See R documentation for more information.
	
	Python code uses scipy.interpolate.splev to get B-spline basis functions, while R code calls C.
	Note that der is the same across x."""
	knots = np.array(knots, dtype=np.float64)
	x = np.array(x, dtype=np.float64)
	knots.sort()
	assert max(x) <= max(knots) and min(x) >= min(knots)
	m = len(knots) - ord
	v = np.zeros((m, len(x)))
	d = np.eye(m, len(knots))
	for i in range(m):
		v[i] = splev(x, (knots, d[i], ord-1), der = der)
	return v.transpose()

def ns(x, df = 1):
	"""Reproduces only that output of the R function ns() necessary for locfdr(). See R documentation for more information."""
	x = np.array(x)
	assert df >= 1
	knots = np.arange(0, 1, 1./df)[1:]
	knots = [np.percentile(x,el*100) for el in knots]
	knots = np.array([min(x)]*4 + knots + [max(x)]*4)
	out = np.matrix(splineDesign(knots, x, 4)[:, 1:])
	const = np.matrix(splineDesign(knots, [min(x), max(x)], 4, 2)[:, 1:])
	QRdec = qr(const.transpose(), mode='full')
	# note that basis may be different from that reported by R because the QR
	# decomposition algo used by scipy is different.
	# of course, it gives same GLM fit.
	return (out * QRdec[0])[:, 2:]

def poly(x, df = 1):
	"""Reproduces only that output of R function poly() necessary for locfdr(). See R documentation for more information."""
	x = np.array(x)
	if df < 1:
		raise InputError('df must be > 0')
	if df >= len(set(x)):
		raise InputError('df must be >= bre')
	xbar = np.mean(x)
	x = x - xbar
	X = np.matrix([[float(x[i])**j for i in xrange(len(x))] for j in xrange(df+1)]).transpose()
	# this is required in R function; not sure it's necessary to check
	assert np.linalg.matrix_rank(X) >= df
	QRdec = qr(X, mode='full')
	z = np.zeros(QRdec[1].shape)
	for i in range(min(z.shape)):
		z[i,i] = QRdec[1][i,i]
	raw = QRdec[0] * sp.matrix(z)
	norm2 = np.sum(np.power(raw,2), axis=0)
	Z = raw / np.sqrt(norm2).repeat(len(x)).reshape(raw.shape[1], raw.shape[0]).transpose()
	return Z[:,1:]

def approx(x, y, xout, rule = 1, ties = 'ordered', tieprecision = 10):
	"""Mimics R function approx(). ties can be either 'ordered' or 'mean'.
	
	'ordered' breaks ties by distributing tied x values at intervals of 1e(-tieprecision).
	This way, given some x-value X with ties, the interpolating function approaches the max y-value
	at x from the right and the min y value at x from the left. tieprecision is ignored if ties = 'mean.'"""
	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	xout = np.array(xout, dtype=np.float64)
	holder = {}
	for i,el in enumerate(x):
		if not holder.has_key(el):
			holder[el] = []
		holder[el].append(y[i])
	together = []
	if ties != 'ordered':
		for key in holder:
			together.append([key, np.mean(holder[key])])
	else:
		for key in holder:
			reps = len(holder[key])
			holder[key].sort()
			for i,el in enumerate(holder[key]):
				together.append([key-(reps-i-1)*np.power(10., -tieprecision), el])
	together.sort(key=lambda lam: lam[0])
	together = np.array(together).transpose()
	vals = sp.interpolate.interp1d(together[0, :], together[1, :], bounds_error = False)(xout)
	if rule == 2:
		try:
			for i,el in enumerate(vals):
				if el > together[0, -1]:
					vals[i] = together[1, -1]
				elif el < together[0, 0]:
					vals[i] = together[1, 0]
		except TypeError:
			if vals > together[0, -1]:
				vals = together[1, -1]
			elif vals < together[0, 0]:
				vals = together[0, 0]
	return vals