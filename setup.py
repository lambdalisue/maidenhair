# coding=utf-8
import sys
from setuptools import setup, find_packages

NAME = 'maidenhair'
VERSION = '0.3.5'

def read(filename):
    import os
    BASE_DIR = os.path.dirname(__file__)
    filename = os.path.join(BASE_DIR, filename)
    fi = open(filename, 'r')
    return fi.read()

def readlist(filename):
    rows = read(filename).split("\n")
    rows = [x.strip() for x in rows if x.strip()]
    return list(rows)

extras = {}
if sys.version_info >= (3,):
    extras['use_2to3'] = True

setup(
    name = NAME,
    version = VERSION,
    description = 'Convert raw text data files into a single excel file.',
    long_description = read('README.rst'),
    classifiers = (
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ),
    keywords = 'data python loader parser statistics',
    author = 'Alisue',
    author_email = 'lambdalisue@hashnote.net',
    url = 'https://github.com/lambdalisue/%s' % NAME,
    download_url = 'https://github.com/lambdalisue/%s/tarball/master' % NAME,
    license = 'MIT',
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    include_package_data = True,
    package_data = {
        '': ['LICENSE', 'README.rst',
             'requirements.txt',
             'requirements-test.txt',
             'requirements-docs.txt'],
    },
    zip_safe=True,
    install_requires=readlist('requirements.txt'),
    entry_points={
        'maidenhair.plugins': [
            'parsers.PlainParser = maidenhair.parsers.plain:PlainParser',
            'parsers.CSVParser = maidenhair.parsers.plain:CSVParser',
            'loaders.PlainLoader = maidenhair.loaders.plain:PlainLoader',
        ],
    },
    **extras
)
