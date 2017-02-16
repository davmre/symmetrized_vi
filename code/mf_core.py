import numpy as np
import tensorflow as tf

import matplotlib.pylab as plt

import scipy.stats
import seaborn as sns

import elbow.util as util

from elbow import Gaussian, Model
from elbow.models.factorizations import NoisyGaussianMatrixProduct, BatchDenseGeneratorByUser
from elbow.models.symmetry_qs import SignFlipGaussian, DiagonalRotationMixture, GaussianMonteCarlo, DiagonalRotationMixtureJensen, general_orthog_correction, LargeInitGaussian

from elbow.structure import split_at_row

def sample_mf_data(n, m, k, sigma_n, sigma_prior=1.0, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n, k) * sigma_prior
    V = np.random.randn(m, k)  * sigma_prior
    R = np.dot(U, V.T) / np.sqrt(k)
    X = R + np.random.randn(*R.shape) * sigma_n
    return X, R



def build_simple_mf_model(n, m, model_k, sigma_n, sigma_prior=1.0):
    A = Gaussian(mean=0.0, std=sigma_prior, shape=(n, model_k), name="A")
    B = Gaussian(mean=0.0, std=sigma_prior, shape=(m, model_k), name="B")
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=sigma_n, name="C", rescale=True)

    q_C = C.observe_placeholder()

    jm = Model(C)
    return jm, q_C

def build_sparse_mf_model(n, m, model_k, row_idxs, col_idxs):
    A = Gaussian(mean=0.0, std=sigma_prior, shape=(n, model_k), name="A")
    B = Gaussian(mean=0.0, std=sigma_prior, shape=(m, model_k), name="B")
    C = NoisySparseGaussianMatrixProduct(A=A, B=B, std=sigma_n,
                                         row_idxs=row_idxs,
                                         col_idxs=col_idxs,
                                         name="C",
                                         rescale=True)

    q_C = C.observe_placeholder()

    jm = Model(C)


def build_local_mf_model(n, m, model_k, batch_size, sigma_n,
                         sigma_prior=1.0, rotational=False):

    batch_ratings = tf.placeholder(shape=(batch_size,m), dtype=np.float32)
    batch_mask = tf.placeholder(shape=(batch_size,m), dtype=np.float32)

    A = Gaussian(mean=0.0, std=sigma_prior, shape=(batch_size, model_k), name="A", local=True)
    B = Gaussian(mean=0.0, std=sigma_prior, shape=(m, model_k), name="B")
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=sigma_n,
                                   name="C", rescale=True,
                                   mask=batch_mask, local=True)

    q_C = C.observe(batch_ratings)

    if rotational:
        qB = DiagonalRotationMixtureJensen(shape=B.shape)
    else:
        qB = LargeInitGaussian(shape=B.shape)
    B.attach_q(qB)
        
    jm = Model(C, minibatch_ratio = float(n)/batch_size)
        
    return jm, batch_ratings, batch_mask
    
def add_rotational_encoder_term(jm, include_weights=False):

    jm.get_variational_nodes()

    qB = jm["B"].q_distribution()
    k = qB.shape[1]

    if include_weights:
        qA = jm["A"].q_distribution()
        #wmean = weights["W_means"]
        #wstd = weights["W_stds"]

        
        
        M = tf.concat(0, (qA.mean*np.sqrt(jm.minibatch_ratio), qB.mean))
        S = tf.concat(0, (qA.std, qB.std ))
        
    else:
        M, S = qB.mean, qB.std
    
    new_term = general_orthog_correction(M, S, k)
    jm.add_elbo_term(new_term)

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

def mf_rot_inference_jensen(jm):
    A, B = jm["A"], jm["B"]
    n, model_k = A.shape
    m, model_k2 = B.shape
    assert(model_k==model_k2)
    
    sA = tf.exp(tf.Variable(-2 * np.ones((n+m, model_k)), dtype=tf.float32))
    q_AB = DiagonalRotationMixtureJensen(shape=(n+m, model_k), std=sA, name="q_AB")
    q_A, q_B = split_at_row(q_AB, n)
    A.attach_q(q_A)
    B.attach_q(q_B)

def mf_halfrot_inference_jensen(jm):
    A, B = jm["A"], jm["B"]
    n, model_k = A.shape
    m, model_k2 = B.shape
    assert(model_k==model_k2)

    sB = tf.exp(tf.Variable(-2 * np.ones((m, model_k)), dtype=tf.float32))
    qB = DiagonalRotationMixtureJensen(shape=(m, model_k), std=sB, name="q_B")
    B.attach_q(qB)

    
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

def global_posterior(jm):
    post = jm.posterior()
    n, k = jm["A"].shape
    if "q_AB" in post:
        qAB = post["q_AB"]["mean"]
        qA = qAB[:n, :]
        qB = qAB[n:, :]
    else:
        try:
            qA = post["q_A"]["mean"]
        except:
            qA = post["q_A"]["tf_value"]

        try:
            qB = post["q_B"]["mean"]
        except:
            qB = post["q_B"]["tf_value"]

    _, model_k = qA.shape
    qX = np.dot(qA, qB.T) / np.sqrt(model_k)
    return qA, qB, qX

def local_posterior(lmodel, user_rows):
    jm, ratings_feed, mask_feed = lmodel

    n = len(user_rows)
    batch_size, m = jm["C"].shape
    b = BatchDenseGeneratorByUser(user_rows, n_items=m,
                                  batch_size_users=batch_size,
                                  shuffle=False)
    

    qA = jm["A"].q_distribution()
    qB = jm["B"].q_distribution()
    qBmean = jm.session.run(qB.mean)

    
    qAmeans = []
    qAstds = []    
    for i in range(0, n, batch_size):
        batch_ratings, batch_mask = b.next_batch()
        fd = {ratings_feed: batch_ratings, mask_feed: batch_mask}
        
        local_qAmean, local_qAstd = jm.session.run((qA.mean, qA.std), feed_dict=fd)
        qAmeans.append(local_qAmean.copy())
        qAstds.append(local_qAstd.copy())
        
    qAmean = np.vstack(qAmeans)[:n]
    qAstds = np.vstack(qAstds)[:n]

    k = qBmean.shape[1]
    qRmean = np.dot(qAmean, qBmean.T) / np.sqrt(k)
    return qAmean, qBmean, qRmean

