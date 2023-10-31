from __future__ import division

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('INFO')
print(tf.__version__)

#import logging as lg
#lg.basicConfig(level=lg.DEBUG)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
tf.get_logger().setLevel(logging.ERROR)

import shutil
from time import time
from pdb import set_trace
from copy import deepcopy
from os.path import join as pjoin
import pandas as pd
import numpy as np
from numpy import array
from pprint import pprint
import copy
import random
from time import time
from pdb import set_trace

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scipy.stats import rankdata, kendalltau
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from smt.surrogate_models import RBF, IDW, RMTB, KRG, LS, KPLS, KPLSK, \
    GEKPLS, QP, MGP

import time
import numpy as np
import itertools
import scipy.stats as spstats
from sklearn.base import BaseEstimator

from evolution.surrogate_models import SurrogateEnsembleWrapper 
# from evolution.gan_train_surr import logger

from .config import config


import torch
from torch.autograd import Variable
from evolution.phenotype import Phenotype
from evolution.discriminator import Discriminator
from evolution.generator import Generator
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import util.tools as tools
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

FIXED_LENGTH_REPRESENTATION_SIZE = 100
PAD_VALUE = -100

def create_variable_length_data(n_inst):
    data = []
    for inst_idx in range(n_inst):
        n_timestep = np.random.randint(1, high=10)
        seq = np.random.rand(n_timestep).tolist()
        data.append(seq)
    return data

def build_base_lstm_model(n_timestep):   
    model = Sequential()
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', input_shape=(n_timestep,1)))
#    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='tanh', input_shape=(n_timestep,1), recurrent_dropout=0.5))
    model.add(RepeatVector(n_timestep))
    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='relu', return_sequences=True,))
#    model.add(LSTM(FIXED_LENGTH_REPRESENTATION_SIZE, activation='tanh', return_sequences=True, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(1)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer='adam', loss='mse')
#    model.compile(optimizer=optimizer, loss='mse')
    return model

class VariableLengthAutoencoderLSTM:
    def __init__(self):
        pass

    def train(self, var_len_seq, n_epoch):
        print("Train the variable-length LSTM autoencoder!")
        shuffled_var_len_seq = copy.deepcopy(var_len_seq)
        curr_weights = None
        for epoch_idx in range(n_epoch):
            print("Epoch: %d" % (epoch_idx+1))
            random.shuffle(shuffled_var_len_seq)
            for seq in shuffled_var_len_seq:
                # reshape input into [samples, timesteps, features]
                n_timestep = len(seq)
                seq = np.array(seq)
                seq = seq.reshape((1,  n_timestep, 1))
                if curr_weights is None:
                    model = build_base_lstm_model(n_timestep)
                    model.fit(seq, seq, epochs=2)

                    # First data instance, we train from scratch then save weights to curr_weights
                    curr_weights = model.get_weights()

                else:
                    model = build_base_lstm_model(n_timestep)
                    t1 = time()
                    # Get weights from current model trained in previous instance.
                    # This helps with ensuring that our model can handle variable-length inputs
                    model.set_weights(curr_weights)
                    t2 = time()
                    print("ABC")
                    print(t2-t1)

                    model.fit(seq, seq, epochs=2)

                    t1 = time()
                    # Save the current weights
                    curr_weights = model.get_weights()
                    t2 = time()
                    print("DEF")
                    print(t2-t1)

        self.model = model

    def train_efficient(self, var_len_seq, n_epoch, batch_size=8):
        padded_seq = pad_sequences(var_len_seq, padding='post', value=PAD_VALUE)
        n_timestep = padded_seq.shape[-1]
        padded_seq = padded_seq.astype(np.float32)
        padded_seq /= 100.0
        padded_seq = padded_seq.reshape(padded_seq.shape[0], padded_seq.shape[1], 1)
        model = build_base_lstm_model(n_timestep)
        model.fit(padded_seq, padded_seq, epochs=n_epoch, batch_size=batch_size)
        curr_weights = model.get_weights()
        self.curr_weights = curr_weights
        self.model = model

    def encode(self, var_len_seq):
        print("Encode into fixed length representation after training!")
        print(var_len_seq)
        fixed_length_outputs  = []
#        set_trace()
        for seq in var_len_seq:
            n_timestep = len(seq)
            seq = np.array(seq)
            seq = seq.astype(np.float32)
            seq /= 100.0
            seq = seq.reshape((1,  n_timestep, 1))

            model = build_base_lstm_model(n_timestep)
#            curr_weights = self.model.get_weights()
#            model.set_weights(curr_weights)
            model.set_weights(self.curr_weights)

            encoder_only_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
            try:
                fixed_length_output = encoder_only_model.predict(seq)
            except:
                print("IT SUCKS!")
                set_trace()
            fixed_length_output = fixed_length_output.tolist()
            fixed_length_output = fixed_length_output[0]

            fixed_length_outputs.append(fixed_length_output)

#        tf.keras.backend.clear_session()
        # print(fixed_length_outputs)
        return fixed_length_outputs

class SMTWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = sm.name

    def train(self, xt, yt):
#        lg.debug("SMTWrapper: Begin training the surrogate model")
        self.xt = xt.copy()
        self.yt = yt.copy()
        self.sm.set_training_values(xt, yt)
        self.sm.train()
#        lg.debug("SMTWrapper: Finished training")

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


class SurrogateManager:
    def __init__(self, surrogate_config, population_size,train_loader):
        print("SURROGATE")
        self.train_loader=train_loader
        self.n_gen_cold_start = surrogate_config["n_gen_cold_start"]
        self.surrogate_name = surrogate_config["surrogate_name"]
        self.n_update = surrogate_config["n_update"]
        # self.dataset_name = surrogate_config["dataset_name"]
        self.population_size = population_size
        self.cold_start_x = []
        self.cold_start_y = []

        if self.surrogate_name == 'IDW':
            sm = SMTWrapper(IDW())
        elif self.surrogate_name == 'RBF':
            sm = SMTWrapper(RBF())
        elif self.surrogate_name == 'LS':
            sm = SMTWrapper(LS())
        elif self.surrogate_name == 'KNN':
            sm = SKLearnRegressorWrapper(KNeighborsRegressor(n_neighbors=5))
        elif self.surrogate_name == 'SVR':
            sm = SKLearnRegressorWrapper(SVR(C=1.0, epsilon=0.2))
        elif self.surrogate_name == 'MLP':
            sm = SKLearnIncrementalRegressorWrapper(MLPRegressor())
        elif self.surrogate_name == 'ensemble_sum_rule':
            sm = SurrogateEnsembleWrapper(rule='sum')
        else:
            raise Exception("In class %s: Surrogate name: %s not defined" % (self.__name__, surrogate_name))
        self.sm = sm

#        set_trace()
        pass

    def set_generation(self, generation_idx):
        self.generation_idx = generation_idx
        if self.generation_idx == self.n_gen_cold_start:
#            lg.debug("Train the variable-length LSTM encoder-decoder to encode the candidates into a fixed-length representation for used by a surrogate model")
            self.var_len_autoencoder_lstm_model = VariableLengthAutoencoderLSTM()
#            self.var_len_autoencoder_lstm_model.train(self.cold_start_x, 5)
            n_epoch = 50
            self.var_len_autoencoder_lstm_model.train_efficient(self.cold_start_x, n_epoch)
            self.cold_start_x_fixed_len = self.var_len_autoencoder_lstm_model.encode(self.cold_start_x)

#            lg.debug("Train the surrogate model")
            self.sm.train(np.array(self.cold_start_x_fixed_len), np.array(self.cold_start_y))


    def evaluate_individual(self, pair_idx,
                            G:Generator, D:Discriminator,train_generator=True, train_discriminator=True, norm_g=1, norm_d=4):
        """
        Manages the evaluation of each individual, whether to use the real fitness function or the surrogate.

        """
        evaluation_type=config.evolution.evaluation.type
        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return
        if self.generation_idx < self.n_gen_cold_start:
            # Use the real fitness function
            torch.cuda.empty_cache()
            print("Cold start at generation ",str(self.generation_idx),' pair_idx:', str(pair_idx))
            self.train_evaluate(G,D,train_generator,train_discriminator,norm_g,norm_d)

            self.cold_start_x.append(G.get_model_vect()+D.get_model_vect())
            self.cold_start_y.append(G.fitness())
            
            self.cold_start_x.append(D.get_model_vect()+G.get_model_vect())
            self.cold_start_y.append(D.fitness())
            # set_trace()
        else:
            # Check if we use the real fitness function or the surrogate model
            if (((self.generation_idx-self.n_gen_cold_start)%self.n_update)==0) and (self.generation_idx>self.n_gen_cold_start):
                # Train the networks in this generation then update
                print("using real fitness at generation ",str(self.generation_idx),' pair_idx:', str(pair_idx))
                self.train_evaluate(G,D,train_generator,train_discriminator,norm_g,norm_d)
                print("update surrogate model at generation ",str(self.generation_idx),' pair_idx:', str(pair_idx))

                model_vect = G.get_model_vect()+D.get_model_vect()
                fixed_model_vect = self.var_len_autoencoder_lstm_model.encode([model_vect])
                fixed_model_vect = fixed_model_vect[0]
                fixed_model_vect_np = np.array(fixed_model_vect)
                self.sm.update(fixed_model_vect_np, G.fitness())

                model_vect = D.get_model_vect()+G.get_model_vect()
                fixed_model_vect = self.var_len_autoencoder_lstm_model.encode([model_vect])
                fixed_model_vect = fixed_model_vect[0]
                fixed_model_vect_np = np.array(fixed_model_vect)
                self.sm.update(fixed_model_vect_np, D.fitness())

                # update_x_fixed_len = self.var_len_autoencoder_lstm_model.encode( [G.get_model_vect()+D.get_model_vect()])
                # self.sm.update(update_x_fixed_len, G.fitness())
                # update_x_fixed_len =  self.var_len_autoencoder_lstm_model.encode([D.get_model_vect()+G.get_model_vect()])
                # self.sm.update(update_x_fixed_len, D.fitness())
                
            else:
                # Use the surrogate model
#                lg.debug("Use the surrogate model")
                print("using surrogate model at generation ",str(self.generation_idx),' pair_idx:', str(pair_idx))

                estimate_fitness=self.estimate_evaluate(G,D)
                if config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score:
                    G.fid_score = estimate_fitness
                elif config.evolution.fitness.generator == "AUC":
                    G.fitness_value = estimate_fitness
                elif config.evolution.fitness.generator == "loss":
                    G.error = estimate_fitness

                estimate_fitness=self.estimate_evaluate(D,G)
                if config.evolution.fitness.discriminator == "AUC":
                    D.fitness_value = estimate_fitness
                elif config.evolution.fitness.discriminator == "loss":
                    D.error = estimate_fitness



    def train_evaluate(self, G:Generator, D:Discriminator,train_generator=True, train_discriminator=True, norm_g=1, norm_d=4):
        """
        Manages the evaluation of each individual, whether to use the real fitness function or the surrogate.

        """
        # Use the real fitness function
        torch.cuda.empty_cache()
        n, ng = 0, 0
        G.error = G.error or 0
        D.error = D.error or 0
        g_error = G.error
        d_error = D.error
        d_fitness_value, g_fitness_value = D.fitness_value, G.fitness_value
        G, D = tools.cuda(G), tools.cuda(D)  # load everything on gpu (cuda)

        G.train()
        D.train()
        while n < config.gan.batches_limit:
            for images, _ in self.train_loader:
                # if n==0: print(images[0].mean())
                n += 1
                if n > config.gan.batches_limit:
                    break
                images = tools.cuda(Variable(images))
                if train_discriminator:
                    D.do_train(G, images)
                if train_generator and n % config.gan.critic_iterations == 0:
                    ng += 1
                    G.do_train(D, images)
        if train_discriminator:
            D.error = d_error + (D.error - d_error)/(n*norm_d)
            D.fitness_value = d_fitness_value + (D.fitness_value - d_fitness_value) / (n * norm_d)
            G.fitness_value = g_fitness_value + (G.fitness_value - g_fitness_value) / (n * norm_g)
        if train_generator:
            G.error = g_error + (G.error - g_error)/(ng*norm_g)
        G, D = G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()
        if config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score:
                G.calc_fid()  

#       set_trace() 
    def estimate_evaluate(self, pheno_1:Phenotype, pheno_2:Phenotype):
        # Use the surrogate model
#                lg.debug("Use the surrogate model")
        model_vect = pheno_1.get_model_vect()+pheno_2.get_model_vect()

        try:
            fixed_model_vect = self.var_len_autoencoder_lstm_model.encode([model_vect]) 
        except:
            print("FIXED_MODEL_VECT ERROR!!!")
            set_trace()
        fixed_model_vect = fixed_model_vect[0]
        fixed_model_vect_np = np.array(fixed_model_vect)
        fitness = self.sm.predict_values(fixed_model_vect_np)
        return fitness
