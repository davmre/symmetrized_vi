import numpy as np
import tensorflow as tf

import matplotlib.pylab as plt

import scipy.stats
import seaborn as sns

import elbow.util as util

from elbow import Gaussian, Model
from elbow.models.factorizations import NoisyGaussianMatrixProduct
from elbow.models.symmetry_qs import SignFlipGaussian, DiagonalRotationMixture, GaussianMonteCarlo

from elbow.structure import split_at_row

def sample_mf_data(n, m, k, sigma_n, sigma_prior=1.0):

    U = np.random.randn(n, k) * sigma_prior
    V = np.random.randn(m, k)  * sigma_prior
    R = np.dot(U, V.T) / np.sqrt(k)
    X = R + np.random.randn(*R.shape) * sigma_n
    return X

def build_simple_mf_model(n, m, model_k, sigma_n, sigma_prior=1.0):
    A = Gaussian(mean=0.0, std=sigma_prior, shape=(n, model_k), name="A")
    B = Gaussian(mean=0.0, std=sigma_prior, shape=(m, model_k), name="B")
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=sigma_n, name="C", rescale=True)

    q_C = C.observe_placeholder()

    jm = Model(C)
    return jm, q_C

def build_sparse_mf_model(n, m, model_k, row_idxs, col_idxs)
    A = Gaussian(mean=0.0, std=sigma_prior, shape=(n, model_k), name="A")
    B = Gaussian(mean=0.0, std=sigma_prior, shape=(m, model_k), name="B")
    C = NoisySparseGaussianMatrixProduct(A=A, B=B, std=sigma_n,
                                         row_idxs=row_idxs,
                                         col_idxs=col_idxs,
                                         name="C",
                                         rescale=True)

    q_C = C.observe_placeholder()

    jm = Model(C)

                          
def mf_analytic_inference(jm):
    A, B = jm["A"], jm["B"]
    
    q_A = GaussianMonteCarlo(shape=A.shape, name="q_A")
    q_B = GaussianMonteCarlo(shape=B.shape, name="q_B")
    A.attach_q(q_A)
    B.attach_q(q_B)


def mf_map_inference(jm):
    A, B = jm["A"], jm["B"]

    A.attach_map_q()
    B.attach_map_q()

def mf_rot_inference(jm):
    A, B = jm["A"], jm["B"]
    n, model_k = A.shape
    m, model_k2 = B.shape
    assert(model_k==model_k2)
    
    sA = tf.tile( tf.exp(tf.Variable(np.asarray(-2).reshape(1,1), dtype=tf.float32)), (n+m, model_k))
    q_AB = DiagonalRotationMixture(shape=(n+m, model_k), std=sA, name="q_AB")
    q_A, q_B = split_at_row(q_AB, n)
    A.attach_q(q_A)
    B.attach_q(q_B)

def mf_isotropic_inference(jm):
    A, B = jm["A"], jm["B"]
    n, model_k = A.shape
    m, model_k2 = B.shape
    assert(model_k==model_k2)
    
    sA =tf.tile( tf.Variable(np.asarray(1e-6).reshape(1,1), dtype=tf.float32), (n, model_k))
    sB = tf.tile(tf.Variable(np.asarray(1e-6).reshape(1,1), dtype=tf.float32), (m, model_k))
    q_A = GaussianMonteCarlo(shape=A.shape, std=sA, name="q_A")
    q_B = GaussianMonteCarlo(shape=B.shape, std=sB, name="q_B")
    A.attach_q(q_A)
    B.attach_q(q_B)
