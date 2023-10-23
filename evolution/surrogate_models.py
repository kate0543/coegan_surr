from __future__ import division

import os
import shutil
import numpy as np
from time import time
from pdb import set_trace
from copy import deepcopy
from os.path import join as pjoin
import pandas as pd

from scipy.stats import rankdata, kendalltau
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans

from smt.surrogate_models import RBF, IDW, RMTB, KRG, LS, KPLS, KPLSK, \
    GEKPLS, QP, MGP

import time
import numpy as np
import itertools
import scipy.stats as spstats
from sklearn.base import BaseEstimator

class SMTWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = sm.name

    def train(self, xt, yt):
        self.xt = xt.copy()
        self.yt = yt.copy()
        self.sm.set_training_values(xt, yt)
        self.sm.train()

    def update(self, xt_inst, yt_inst):
        self.xt = np.vstack((self.xt, xt_inst))
        self.yt = np.append(self.yt, yt_inst)
#        df_xt = pd.DataFrame(self.xt)
#        df_yt = pd.DataFrame(self.yt)
#        df_xt.drop_duplicates(keep=False)
#        df_yt.drop_duplicates(keep=False)
#        self.xt = df_xt.to_numpy()
#        self.yt = df_yt.to_numpy()

        self.train(self.xt, self.yt)

    def predict_values(self, xt_inst):
        try:
            result = self.sm.predict_values(xt_inst.reshape(1, -1))[0,0]
        except:
            try:
                result = self.sm.predict_values(xt_inst.reshape(1, -1))[0]
            except:
                set_trace()
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SKLearnIncrementalRegressorWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = type(sm).__name__

    def train(self, xt, yt):
        self.sm = self.sm.fit(xt, yt)

    def update(self, xt_inst, yt_inst):
        self.sm.partial_fit(xt_inst.reshape(1, -1), yt_inst.reshape(1, ))

    def predict_values(self, xt_inst):
        result = self.sm.predict(xt_inst.reshape(1, -1))
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SKLearnRegressorWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = type(sm).__name__

    def train(self, xt, yt):
        self.xt = xt.copy()
        self.yt = yt.copy()
        self.sm = self.sm.fit(xt, yt)

    def update(self, xt_inst, yt_inst):
        self.xt = np.vstack((self.xt, xt_inst))
        try:
            self.yt = np.append(self.yt, yt_inst)
        except:
            set_trace()
        self.train(self.xt, self.yt)

    def predict_values(self, xt_inst):
        result = self.sm.predict(xt_inst.reshape(1, -1))
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SurrogateEnsembleWrapper:
    def __init__(self, rule='sum'):
        sm_list = []
        sm_list.append(SMTWrapper(RBF(d0=5)))
        sm_list.append(SMTWrapper(IDW(p=2)))
        sm_list.append(SMTWrapper(LS()))
        sm_list.append(SKLearnIncrementalRegressorWrapper(MLPRegressor()))
        sm_list.append(SKLearnRegressorWrapper(KNeighborsRegressor(n_neighbors=5)))
        sm_list.append(SKLearnRegressorWrapper(SVR(C=1.0, epsilon=0.2)))

        self.sm_list = sm_list
        self.rule = rule
        self.name = 'Ensemble_%s_rule' % (rule)

    def train(self, xt, yt):
        sm_list = self.sm_list
        for sm_idx in range(len(sm_list)):
            sm_list[sm_idx].train(xt, yt)
        del sm_idx
        self.sm_list = sm_list

    def update(self, xt_inst, yt_inst):
        sm_list = self.sm_list
        for sm_idx in range(len(sm_list)):
            sm_list[sm_idx].update(xt_inst, yt_inst)
        del sm_idx
        self.sm_list = sm_list

    def predict_values(self, xt_inst):
        sm_list = self.sm_list
        rule = self.rule

        if rule == 'sum':
            pred_ensemble = 0
            for sm in sm_list:
                pred = sm.predict_values(xt_inst)
                pred_ensemble += pred
            pred_ensemble /= (1.0 * len(sm_list))
            return pred_ensemble
        else:
            raise NotImplementedError('The ensemble rule "%s" is not implemented' % (rule))

        self.sm_list = sm_list

def main():
    pass

if __name__ == '__main__':
    main()
