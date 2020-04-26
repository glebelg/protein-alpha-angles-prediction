#!/bin/bash

mkdir data
wget https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp12.tar.gz -O data/casp12.tar.gz
tar zxvf data/casp12.tar.gz -C data

# files data/casp12/training_{30, 50, 70, 90, 95, 100} include 25299, 34039, 41522, 49600, 50914, 104059 proteins


pip install torch
pip install biopython
