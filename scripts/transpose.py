#!/usr/bin/env python

import sys
import pandas
import numpy


mat = pandas.read_csv(sys.argv[1], sep="\t", index_col=0)
mat[~numpy.isnan(mat).any(axis=1)].transpose().to_csv(sys.argv[2], sep="\t")
