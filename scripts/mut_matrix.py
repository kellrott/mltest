#!/usr/bin/env python

import sys
import csv

reader = csv.reader(open(sys.argv[1]), delimiter="\t")
header = None

writer = csv.writer(open(sys.argv[2], "w"), delimiter="\t")

for row in reader:
    if header is None:
        header = row
        writer.writerow( row )
    else:
        o = []
        for a in row[1:]:
            v = list(b for b in a.split(",") if len(b) and b != "Silent")
            o.append(len(v) > 0)
        if sum(o) > 20:
            writer.writerow( [row[0]] + list(1 if a else 0 for a in o) )
