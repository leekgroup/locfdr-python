import numpy as np
import scipy as sp
from scipy.linalg import qr
from scipy.interpolate import splev

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.
	Attributes:
        expr---input expression in which the error occurred
        msg---explanation of the error"""
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

def splineDesign(knots, x, ord = 4, der = 0):
	"""Reproduce behavior of R function splineDesign() for use by ns().
	See R documentation for more information.
	Python code uses scipy.interpolate.splev to get B-spline basis functions.
	R code calls some C function.
	Note that der is the same across x; can be hacked so it's a vector, but not necessary here."""
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
	"""Reproduce only that output of the R function ns() necessary for locfdr().
	See R documentation for more information."""
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
	"""Reproduce only that output of R function poly() necessary for locfdr().
	See R documentation for more information.
	Python code mirrors R code where efficient."""
	x = np.array(x)
	if df < 1:
		raise InputError('if df < 1', 'df must be > 0')
	if df >= len(set(x)):
		raise InputError('elif df >= uniques', 'df must be >= bre')
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