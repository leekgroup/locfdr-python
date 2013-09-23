# setup.py: for installing locfdr-python
# Part of locfdr-python, http://www.github.com/buci/locfdr-python/
#
# Copyright (C) 2013 Abhinav Nellore (anellore@gmail.com)
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

#!/usr/bin/env python

from distutils.core import setup

setup(name = "locfdr-python",
    version = "0.1",
    description = "Computes local false discovery rates.",
    long_description = "A Python port of R function locfdr(), written by Efron, Turnbull, and Narasimhan and enhanced by Frazee, Collado-Torres, and Leek.",
    author = "Abhinav Nellore",
    author_email = 'anellore@gmail.com',
    url = "http://www.github.com/buci",
    download_url = "https://github.com/buci/locfdr-python",
    platforms = ['any'],
    license = "GPLv2",
    py_modules = ['locfdr', 'locfns', 'Rfunctions'],
    classifiers = [
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Development Status :: 5 - Production/Stable',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python']
)
