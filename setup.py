#!/usr/bin/env python

from distutils.core import setup

setup(name = "locfdr-python",
    version = "0.1",
    description = "Computes local false discovery rates.",
    long_description = "A Python port of R function locfdr(), written by Efron, Turnbull, and Narasimhan.",
    author = "Abhinav Nellore",
    author_email = 'anellore@gmail.com',
    url = "http://www.github.com/buci",
    download_url = "https://github.com/buci/locfdr-python",
    platforms = ['any'],
    license = "MIT",
    py_modules = ['locfdr', 'locfns', 'Rfunctions'],
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Development Status :: 5 - Production/Stable',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python']
)
