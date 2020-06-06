#!/bin/sh

for i in 2010,2015 2011,2016 2012,2017 2013,2018;
do IFS=",";
set $i;
    for trans in box_cox in_between untransformed
    do
        python3 pca_analysis.py -b $1 -e $2 -t $trans
    done
done