# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse


def predict(path, thresh):
    test_feature = pd.read_csv(path, header=None)
    gbm = lgb.Booster(model_file='lgb_model.txt')
    results = np.where(gbm.predict(test_feature) > thresh, 1, 0)
    np.savetxt('results.csv', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Please write the csv-file path")
    parser.add_argument("thresh", help="the threshold to predict pos or neg", type=float)
    args = parser.parse_args()
    predict(args.path, args.thresh)

