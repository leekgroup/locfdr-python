# locfns.py: computes mles of p0, sig0, and del0 for locfdr.py
# Part of locfdr-python, http://www.github.com/buci/locfdr-python/
#
# Copyright (C) 2013 Abhinav Nellore (anellore@gmail.com)
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
	print 'numpy is required, but it was not found. locfns was tested on numpy 1.7.1.'
	raise
try:
	from scipy import stats
	from scipy.linalg import pascal
except ImportError:
	print 'scipy is required, but it was not found. locfns was tested on scipy 0.12.0.'
	raise
try:
	import pandas as pd
except ImportError:
	print 'pandas is required, but it was not found. locfns was tested on pandas 0.12.0.'
	raise

def locmle(z, xlim = None, Jmle = 35, d = 0., s = 1., ep = 1/100000., sw = 0, Cov_in = None):
	"""Uses z-values in [-xlim,xlim] to find mles for p0, del0, sig0 .

	Jmle is the number of iterations, beginning at (del0, sig0) = (d, s).
	sw = 1 returns the correlation matrix.
	z can be a numpy/scipy array or an ordinary Python array.
	Note that this function returns pandas Series."""
	N = len(z)
	if xlim is None:
		if N > 500000:
			b = 1
		else:
			b = 4.3 * np.exp(-0.26*np.log10(N))
		xlim = np.array([np.median(z), b*(np.percentile(z, 75)-np.percentile(z, 25))/(2*stats.norm.ppf(.75))])
	aorig = xlim[0] - xlim[1]
	borig = xlim[0] + xlim[1]
	z0 = np.array([el for el in z if el >= aorig and el <= borig])
	N0 = len(z0)
	Y = np.array([np.mean(z0), np.mean(np.power(z0, 2))])
	that = float(N0) / N
	# find MLE estimates
	for j in xrange(Jmle):
		bet = np.array([d/(s*s), -1/(2*s*s)])
		aa = (aorig - float(d)) / s
		bb = (borig - float(d)) / s
		H0 = stats.norm.cdf(bb) - stats.norm.cdf(aa)
		fa = stats.norm.pdf(aa)
		fb = stats.norm.pdf(bb)
		H1 = fa - fb
		H2 = H0 + aa * fa - bb * fb
		H3 = (2 + aa*aa) * fa - (2 + bb*bb) * fb
		H4 = 3 * H0 + (3 * aa + np.power(aa, 3)) * fa - (3 * bb + np.power(bb, 3)) * fb
		H = np.array([H0, H1, H2, H3, H4])
		r = float(d) / s
		I = pascal(5, kind = 'lower', exact = False)
		u1hold = np.power(s, range(5))
		u1 = np.matrix([u1hold for k in range(5)])
		II = np.power(r, np.matrix([[max(k-i, 0) for i in range(5)] for k in range(5)]))
		I = np.multiply(np.multiply(I, II), u1.transpose())
		E = np.array(I * np.matrix(H).transpose()).transpose()[0]/H0
		mu = np.array([E[1], E[2]])
		V = np.matrix([[E[2] - E[1]*E[1], E[3] - E[1] * E[2]],[E[3] - E[1] * E[2], E[4] - E[2]*E[2]]])
		addbet = np.linalg.solve(V, (Y - mu).transpose()).transpose()/(1+1./((j+1)*(j+1)))
		bett = bet + addbet
		if bett[1] > 0:
			bett = bet + .1 * addbet
		if pd.isnull(bett[1]) or bett[1] >= 0:
			break
		d = -bett[0]/(2 * bett[1])
		s = 1 / np.sqrt(-2. * bett[1])
		if np.sqrt(sum(np.array(np.power(bett - bet, 2)))) < ep:
			break
	if pd.isnull(bett[1]) or bett[1] >= 0:
		mle = np.array([np.nan for k in xrange(6)])
		Cov_lfdr = np.nan
		if pd.isnull(bett[1]):
			Cov_out = np.nan
		Cor = np.matrix([[np.nan]*3]*3)
	else:
		aa = (aorig - d) / s
		bb = (borig - d) / s
		H0 = stats.norm.cdf(bb) - stats.norm.cdf(aa)
		p0 = that / H0
		# sd calcs
		J = s*s * np.matrix([[1, 2 * d],[0, s]])
		JV = J * np.linalg.inv(V)
		JVJ = JV * J.transpose()
		mat = np.zeros((3,3))
		mat[1:,1:] = JVJ/N0
		mat[0,0] = (p0 * H0 * (1 - p0 * H0)) / N
		h = np.array([H1/H0, (H2 - H0)/H0])
		matt = np.eye(3)
		matt[0,:] = np.array([1/H0] + (-(p0/s) * h).tolist())
		matt = np.matrix(matt)
		C = matt * (mat * matt.transpose())
		mle = np.array([p0, d, s] + np.sqrt(np.diagonal(C)).tolist())
		if sw == 1:
			sd = mle[3:]
			Co = C/np.outer(sd, sd)
			Cor = Co[:,[1,2,0]][[1,2,0]]
			# switch to pandas dataframe for labeling
			Cor = pd.DataFrame(Cor, index=['d', 's','p0'], columns=['d','s','p0'])
		if Cov_in is not None:
			i0 = [i for i,x in enumerate(Cov_in['x']) if x > aa and x < bb]
			Cov_out = loccov(N, N0, p0, d, s, Cov_in['x'], Cov_in['X'], Cov_in['f'], JV, Y, i0, H, h, Cov_in['sw'])
	#label with pandas Series
	mle = pd.Series(mle[[1,2,0,4,5,3]], index=['del0', 'sig0', 'p0', 'sd_del0', 'sd_sig0', 'sd_p0'])
	out = {}
	out['mle'] = mle
	if sw == 1:
		out['Cor'] = Cor
	if Cov_in is not None:
		if Cov_in['sw'] == 2:
			out['pds_'] = Cov_out
		elif Cov_in['sw'] == 3:
			out['Ilfdr'] = Cov_out
		else:
			out['Cov_lfdr'] = Cov_out
	if sw == 1 or Cov_in is not None:
		return pd.Series(out)
	return mle

def loccov(N, N0, p0, d, s, x, X, f, JV, Y, i0, H, h, sw):
	M = np.ones((3, len(x)))
	M[1, :] = x - Y[0]
	M[2, :] = np.power(x,2) - Y[1]
	if sw == 2:
		K = len(x)
		K0 = len(i0)
		mat = np.zeros((3, 3))
		mat[0, 1:] = (-np.matrix(h) * (JV / s))[0]
		mat[0,0] = 1 - float(N0)/N
		mat[1:, 1:] = JV / p0
		M0 = M[:, i0]
		dpds_dy0 = mat * np.matrix((M0 / N) / H[0])
		dy0_dy = np.zeros((K0, K))
		dy0_dy[:, i0] = np.eye(K0)
		dpds_dy = dpds_dy0 * dy0_dy
		dpds_dy = pd.DataFrame(dpds_dy, index=['p', 'd', 's'])
		return dpds_dy
	xstd = (x - d) / s
	U = np.zeros((2, len(xstd)))
	U[0, :] = xstd - H[1]/H[0]
	U[1, :] = np.power(xstd, 2) - H[2]/H[0]
	U = U.transpose()
	for i in range(M.shape[1]):
		if i not in i0:
				M[:, i] = 0
	dl0plus_dy = np.zeros((U.shape[0], JV.shape[1]+1))
	dl0plus_dy[:,1:] = U * (JV / s)
	dl0plus_dy[:,0] = 1 - float(N0)/N
	dl0plus_dy = dl0plus_dy * np.matrix(M / N / H[0] / p0)
	X = np.matrix(X)
	fholder = np.matrix([f for k in xrange(X.shape[1])]).transpose()
	G = X.transpose() * (np.multiply(fholder, X))
	dl_dy = X * np.linalg.inv(G) * X.transpose()
	dlfdr_dy = dl0plus_dy - dl_dy
	if sw == 3:
		return dlfdr_dy
	fholder = np.matrix([f for k in xrange(dlfdr_dy.shape[1])]).transpose()
	Cov_lfdr = dlfdr_dy * (np.multiply(fholder, dlfdr_dy.transpose()))
	return Cov_lfdr

def loccov2(X, X0, i0, f, ests, N):
	d = ests[0]
	s = ests[1]
	p0 = ests[2]
	theo = (len(X0[0]) == 1)
	Xtil = X[i0,:]
	X0til = X0[i0,:]
	X = np.matrix(X)
	fholder = np.matrix([f for k in xrange(X.shape[1])]).transpose()
	G = X.transpose() * np.multiply(fholder, X)
	X0til = np.matrix(X0til)
	G0 = X0til.transpose() * X0til
	B0 = X0 * (np.linalg.inv(G0) * X0til.transpose()) * Xtil
	C = B0 - X
	Ilfdr = C * np.linalg.solve(G, X.transpose())
	Cov = C * np.linalg.inv(G) * C.transpose()
	if theo:
		D = np.ones((1,1))
	else:
		D = np.matrix([[1, d, s*s+d*d], [0, s*s, 2*d*s*s], [0, 0, s*s*s]], dtype=np.float64)
	gam_ = np.linalg.solve(G0, X0til.transpose()) * (Xtil * np.linalg.solve(G, X.transpose()))
	pds_ = D * gam_
	if theo:
		pds_ = np.append(pds_, np.zeros((2, X.shape[0])), axis=0)
	pds_[0,:] = pds_[0,:] - 1./N
	f = np.matrix(f)
	m1 = pds_ * f.transpose()
	m2 = np.power(pds_, 2) * f.transpose()
	stdev = np.sqrt(m2 - np.power(m1, 2) / float(N))
	stdev[0] = p0 * stdev[0]
	pds_[0,:] = p0 * pds_[0,:]
	# pandas dataframe for labeling
	pds_ = pd.DataFrame(pds_, index=['p', 'd','s'])
	return pd.Series({'Ilfdr' : Ilfdr, 'pds_' : pds_, 'stdev' : np.array(stdev.transpose())[0], 'Cov' : Cov})
