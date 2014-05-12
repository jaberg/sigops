#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


setup(
    name="sigops",
    version="0.1.0",
    author="",
    author_email="",
    packages=['sigops', 'sigops.tests'],
    scripts=[],
    url="",
    license="BSD",
    description="",
    long_description="",
    requires=[
        "numpy (>=1.5.0)",
        "networkx",
    ],
)
