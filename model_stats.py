#!/usr/bin/env python

import os
import sys
import json
import argparse
from glob import glob
from sklearn.metrics import auc
import numpy
import pandas
import multiprocessing

def recall(stats):
    return float(stats['tp']) / float(stats['tp'] + stats['fn'])

def precision(stats):
    return float(stats['tp']) / float(stats['tp'] + stats['fp'])

def true_count(stats):
    return stats['fn'] + stats['tp']


def run_scan(args):
    for a in glob(os.path.join(args.models, "*.model")):
        with open(a) as handle:
            txt = handle.read()
            data = json.loads(txt)
            thresholds = dict(list( (float(i),data['stats'][i])  for i in data['stats']))
            mthresh = min(thresholds.keys())
            #if mthresh > 0.01:
            if true_count(thresholds[mthresh]) > 0:
                x = list( (i, precision(thresholds[i]), recall(thresholds[i])) for i in sorted(thresholds.keys(), reverse=True))
                print data['gene'], auc( [0.0] + list(i[2] for i in x),  [1.0] + list(i[1] for i in x), True ), true_count(thresholds.values()[0]), list( (i, precision(thresholds[i]), recall(thresholds[i])) for i in thresholds)
                #print data['gene'], precision(thresholds[mthresh]), recall(thresholds[mthresh]), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)
                #print data['gene'], list( (recall(thresholds[i]), i) for i in thresholds), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)
                #print data['gene'], list( (i, precision(thresholds[i]), recall(thresholds[i])) for i in thresholds), true_count(thresholds[mthresh]) #, min(thresholds), max(thresholds)

def run_prscore(args):
    
    gene_scores = {}
    
    for a in glob(os.path.join(args.models, "*.model")):
        with open(a) as handle:
            txt = handle.read()
            data = json.loads(txt)
            thresholds = dict(list( (float(i),data['stats'][i])  for i in data['stats']))
            mthresh = min(thresholds.keys())
            gene = data['gene']
            if gene not in gene_scores:
                gene_scores[gene] = []
            if true_count(thresholds[mthresh]) > 0:
                x = list( (i, precision(thresholds[i]), recall(thresholds[i])) for i in sorted(thresholds.keys(), reverse=True))
                pr_vals = list(i[2] for i in x)
                rc_vals = list(i[1] for i in x)
                a = auc( [0.0] + rc_vals, [1.0] + pr_vals, True )
                #a = auc( [1.0] + pr_vals, [0.0] + rc_vals, True )
                gene_scores[gene].append(a)

    matrix = None
    if args.original is not None:
        matrix = pandas.read_csv(args.original, sep="\t", index_col=0)

    for gene in gene_scores:
        vals = gene_scores[gene] + [0.0] * (args.folds-len(gene_scores[gene]))
        if matrix is None or gene not in matrix.columns :
            print gene, numpy.mean(vals), numpy.std(vals)
        else:
            base = float(sum(matrix[gene] == 1)) / float(len(matrix[gene]))
            print gene, numpy.mean(vals), numpy.std(vals), base, (numpy.mean(vals) - base) / numpy.std(vals)


def run_ratios(args):
    matrix = pandas.read_csv(args.labels, sep="\t", index_col=0)
    for i in matrix.columns:
        if args.mode == "ratio":
            print i, float(sum(matrix[i] == 1)) / float(len(matrix[i]))
        if args.mode == "count":
            print i, sum(matrix[i] == 1)



class Model:

    def __init__(self, path):
        with open(path) as handle:
            data = handle.read()
            self.model_data = json.loads(data)
        self.weights =  pandas.Series(self.model_data['weights'])
        self.intercept = self.model_data['intercept']
        self.gene = self.model_data['gene']

    def predict(self, sample):
        common = self.weights.index.union(sample.index)
        if len(common) == 0:
            raise Exception("No model overlap")
        left = self.weights.reindex(index=common, copy=False, fill_value=0.0)
        right = sample.reindex(index=common, copy=False, fill_value=0.0)

        margin = left.dot(right) + self.intercept
        try:
            score = 1.0/ (1.0 + numpy.exp(-margin))
            return score
        except OverflowError:
            return numpy.nan


def model_predict(args):
    model_path, matrix = args
    print "Predicting", model_path
    model = Model(model_path)
    predict = {}
    for c in matrix.columns:
        sample = matrix[c]
        p = model.predict(sample)
        predict[c] = p
    return model.gene, predict


def run_model(args):

    matrix = pandas.read_csv(args.matrix, sep="\t", index_col=0).replace(numpy.nan, 0.0)

    gene_set = {}
    for model_path in glob(os.path.join(args.models, "*")):
        with open(model_path) as handle:
            txt = handle.read()
            data = json.loads(txt)
            gene_set[data['gene']] = model_path

    out = pandas.DataFrame(columns=matrix.columns, index=gene_set.keys())

    pool = multiprocessing.Pool(8)
    for value in pool.map(model_predict, ( (i, matrix) for i in gene_set.values() )):
        print value
        
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommand")

    parser_scan = subparsers.add_parser('scan')
    parser_scan.add_argument("models")
    parser_scan.set_defaults(func=run_scan)

    parser_prscore = subparsers.add_parser('prscore')
    parser_prscore.add_argument("models")
    parser_prscore.add_argument("--folds", default=1)
    parser_prscore.add_argument("--original", default=None)
    parser_prscore.set_defaults(func=run_prscore)

    parser_ratios = subparsers.add_parser('labels')
    parser_ratios.add_argument("labels")
    parser_ratios.add_argument("mode")
    parser_ratios.set_defaults(func=run_ratios)  
    
    parser_model = subparsers.add_parser('model')
    parser_model.add_argument("models")
    parser_model.add_argument("matrix")
    parser_model.set_defaults(func=run_model)  
    
    
    args = parser.parse_args()
    sys.exit(args.func(args))


