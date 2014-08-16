#!/usr/bin/env python

import sys
import os
import json
from glob import glob

def recall(stats):
    return float(stats['tp']) / float(stats['tp'] + stats['fn'])

def precision(stats):
    return float(stats['tp']) / float(stats['tp'] + stats['fp'])

def true_count(stats):
    return stats['fn'] + stats['tp']

for a in glob(os.path.join(sys.argv[1], "*.model")):
    with open(a) as handle:
        txt = handle.read()
        data = json.loads(txt)
        thresholds = dict(list( (float(i),data['stats'][i])  for i in data['stats']))
        mthresh = min(thresholds.keys())
        #if mthresh > 0.01:
        if true_count(thresholds[mthresh]) > 0:
            #print data['gene'], precision(thresholds[mthresh]), recall(thresholds[mthresh]), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)
            #print data['gene'], list( (recall(thresholds[i]), i) for i in thresholds), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)
            print data['gene'], list( (i, thresholds[i]) for i in thresholds), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)
