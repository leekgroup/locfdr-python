# example.py: executes simulation of Section 4 from
# http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf
#
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

from locfdr import locfdr

f = open('samplez.tsv')
zz = []

for line in f:
    zz.append(float(line.strip()))

results = locfdr(zz)
print 'Executed simulation described in Section 4 of the locfdr() R vignette at http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf . See results in variable \'results\'.'
