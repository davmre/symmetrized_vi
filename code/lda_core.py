import numpy as np
import tensorflow as tf

import matplotlib.pylab as plt

import scipy.stats
import seaborn as sns

import elbow.util as util
from elbow import Model, WrapperNode, ConditionalDistribution
from elbow.joint_model import BatchGenerator

from elbow.util import extract_shape
from elbow.elementary import Gaussian, DirichletMatrix
from elbow.transforms import DeterministicTransform, TransformedDistribution, Simplex, Simplex1, \
    RowNormalize1, UnaryTransform, Transpose, Simplex1Col, SimplexCol
from elbow.binops import BinaryTransform, CombinedDistribution, concat_binop, VStack
from elbow.structure import PackRVs, split_at_row
from elbow.models.symmetry_qs import ExplicitPermutationWrapper, GaussianMonteCarlo

# topics is an (n_topics, n_types) row-stochastic matrix
# doc_dists is an (n_docs, n_topics) row-stochastic matrix
# we want the product, an (n_docs, n_types) row-stochastic matrix
# and then to return an (n_docs, n_types) integer matrix of counts, which will also depend on document sizes

def sample_multinomial_counts(logits, num_words, num_samples):
    m = tf.multinomial(logits=logits, num_samples=num_samples)
    mhot = tf.one_hot(m, depth=num_words, axis=1)
    return tf.reduce_sum(mhot, axis=-1)

def approx_log_factorial(X):
    # use Stirling's approximation to approximate log X! (entrywise),
    # with some hacking to enforce 0! = 1 and 1! = 1 as special cases
    logx = tf.log(tf.cast(X, tf.float32))
    logx = tf.select(tf.is_finite(logx), logx, tf.zeros_like(logx));
    stirling = X * logx - X + .5 * logx + 0.9189385332 # (final const = .5 log(2pi) )
    stirling = tf.select(tf.equal(logx, 0), tf.zeros_like(stirling), stirling)
    return stirling

class CollapsedLDA(ConditionalDistribution):
        
    def __init__(self, topics, document_dists, n_tokens=None,
                 inference_weights=None, **kwargs):
        self.n_tokens = n_tokens
        self.inference_weights = inference_weights
        super(CollapsedLDA, self).__init__(topics=topics, document_dists=document_dists,
                                           **kwargs)

    def inputs(self):
        return {"topics": None, "document_dists": None}
    
    def _compute_shape(self, topics_shape, document_dists_shape,  **kwargs):
        n_topics, n_types = topics_shape
        n_docs, n_topics2 = document_dists_shape
        assert(n_topics2 == n_topics)

        return (n_docs, n_types)
    
    def _sample(self, topics, document_dists):

        if self.n_tokens is None:
            raise Exception("cannot sample from CollapsedLDA without n_tokens specified")
        
        n_docs, n_types = self.shape
        document_probs = tf.matmul(document_dists, topics)
        logits = tf.log(document_probs)
        n_tokens = tf.cast(self.n_tokens, tf.int32)
        return sample_multinomial_counts(logits = logits, num_words=n_types, num_samples=n_tokens)
    
    def _logp(self, result, topics, document_dists):
        n_docs, n_types = self.shape
        document_probs = tf.matmul(document_dists, topics)
        logits = tf.log(document_probs)

        lp = tf.reduce_sum(logits * result)

        n_tokens_by_doc = tf.reduce_sum(result, axis=1)        
        # each row has the normalizing constant n! / (x1! x2! ...)
        # which we approximate with Stirling's formula log x! ~= x log x - x + .5 * log(2 pi x)
        log_normalizer = tf.reduce_sum(approx_log_factorial(n_tokens_by_doc)) - tf.reduce_sum(approx_log_factorial(result))
        
        return lp + log_normalizer

    def _inference_networks(self, q_result):
        # want a network that operates row-wise on a matrix of word
        # counts, and produces a topic dist matrix


        n_docs, n_types = q_result.shape
        n_topics, n_types1 = self.inputs_random["topics"].shape
        assert(n_types==n_types1)

        with tf.name_scope("network"):
            means, stds, weights = build_topic_network(q_result._sampled,
                                                       n_topics=n_topics-1,
                                                       weights = self.inference_weights)
        self.inference_weights = weights

        with tf.name_scope("sb"):
            softmax_basis = Gaussian(mean=means, std=stds, shape=(n_docs, n_topics-1), name="q_neural_pre_logit")
        with tf.name_scope("td"):
            qD = TransformedDistribution(softmax_basis, Simplex1, name="q_neural_"+self.name)
        
        return {"document_dists": qD}
    
def build_topic_network(docs, n_topics, weights=None):
    # docs is a TF variable with shape n_docs, n_words

    n_docs, n_types = util.extract_shape(docs)
    n_hidden1 = 128
    n_hidden2 = 128

    from elbow.models.neural import layer, init_weights, init_biases

    if weights is None:
        weights = {}
        weights["W1"] = init_weights((n_types, n_hidden1), stddev=1e-4)
        weights["b1"] = init_biases((n_hidden1,))
        weights["W2"] = init_weights((n_hidden1, n_hidden2), stddev=1e-4)
        weights["b2"] = init_biases((n_hidden2,))

        weights["W_means"] = init_weights((n_hidden2, n_topics), stddev=1e-4)
        weights["b_means"] = init_biases((n_topics,))
        weights["W_stds"] = init_weights((n_hidden2, n_topics), stddev=1e-4)
        weights["b_stds"] = init_biases((n_topics,))

    def build_network(W1, W2, b1, b2, W_means, b_means, W_stds, b_stds):

        h1 = tf.nn.relu(layer(docs, W1, b1))
        h2 = tf.nn.relu(layer(h1, W2, b2))
        means = layer(h2, W_means, b_means)
        stds = tf.exp(layer(h2, W_stds, b_stds)) 
        return means, stds
    
    means, stds = build_network(**weights)
    
    return means, stds, weights


"""
def deterministic_recognizer(q_topics_base, subtract_entropy=False):
    n_topics, n_types1 = q_topics_base.shape

    row_means = tf.unpack(q_topics_base.mean)
    row_stds = tf.unpack(q_topics_base.std)

    row_lps = []
    for i in range(n_topics):
        entry_lps = []
        for j in range(i, n_topics):
            mi, mj = row_means[i], row_means[j]
            si, sj = row_stds[i], row_stds[j]
            entry_lp = tf.reduce_sum(util.dists.gaussian_log_density(mi, mean=mj, variance = si**2 + sj**2))
            entry_lps.append(entry_lp)
            
        if len(entry_lps) > 0:
            row_lp = util.reduce_logsumexp(tf.pack(entry_lps)) # log sum_j>=i N_ij
            if subtract_entropy:
                row_lps.append(row_lp - entry_lps[0])
            else:
                row_lps.append(row_lp)
                
    recognition_lp = tf.reduce_sum(tf.pack(row_lps))
    return approx_log_factorial(n_topics) - recognition_lp

def dr2(q_topics_base, subtract_entropy=False):
"""
def deterministic_recognizer(q_topics_base, subtract_entropy=False):
    n_topics, n_types1 = q_topics_base.shape

    # q has shape k x m
    # we want something with shape k x k x m
    
    qvariances = tf.square(q_topics_base.std)
    vi = tf.expand_dims(qvariances, axis=1)
    vj = tf.expand_dims(qvariances, axis=0)
    vsums = vi + vj

    qmeans = q_topics_base.mean
    mi = tf.expand_dims(qmeans, axis=1)
    mj = tf.expand_dims(qmeans, axis=0)

    r = -.5 * tf.reduce_sum(tf.square(mi - mj) / vsums, axis=2)
    logz = -.5 * tf.reduce_sum(tf.log(2*np.pi*vsums), axis=2)
    tri_mask = np.float32(np.tri(n_topics).T)
    
    gaussians = tri_mask * (r + logz)

    row_lps = util.reduce_logsumexp(gaussians, axis=1)
    if subtract_entropy:
        row_lps -= tf.diag_part(gaussians)
    recognition_lp = tf.reduce_sum(row_lps)

    """
    row_means = tf.unpack(q_topics_base.mean)
    row_stds = tf.unpack(q_topics_base.std)

    row_lps = []
    all_entry_lps = []
    for i in range(n_topics):
        entry_lps = []
        for j in range(i, n_topics):
            mi, mj = row_means[i], row_means[j]
            si, sj = row_stds[i], row_stds[j]
            entry_lp = tf.reduce_sum(util.dists.gaussian_log_density(mi, mean=mj, variance = si**2 + sj**2))
            entry_lps.append(entry_lp)
            
        if len(entry_lps) > 0:
            row_lp = util.reduce_logsumexp(tf.pack(entry_lps)) # log sum_j>=i N_ij
            if subtract_entropy:
                row_lps.append(row_lp - entry_lps[0])
            else:
                row_lps.append(row_lp)
        all_entry_lps.append(entry_lps)
        
    recognition_lp2 = tf.reduce_sum(tf.pack(row_lps))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    import pdb; pdb.set_trace()
    """
    return approx_log_factorial(n_topics) - recognition_lp


def lda_model(n_docs, n_types, n_topics,
              alpha_topics=None,
              alpha_docs=None,
              minibatch_size=None,
              inference_weights=None,
              n_tokens=None):

    if alpha_topics is None:
        alpha_topics = 0.4
    if alpha_docs is None:
        alpha_docs = 50./n_topics

    local_docs = minibatch_size is not None
    n_batch = n_docs if minibatch_size is None else minibatch_size
    
    topic_matrix = DirichletMatrix(shape=(n_topics, n_types), alpha=alpha_topics, name="topics")
    doc_matrix = DirichletMatrix(shape=(n_batch, n_topics), alpha=alpha_docs, name="docs", local=local_docs)
    doc_words = CollapsedLDA(topics=topic_matrix, document_dists=doc_matrix,
                             n_tokens=n_tokens, name="words",
                             local=local_docs, inference_weights=inference_weights)

    jm = Model(doc_words, minibatch_ratio = n_docs/float(n_batch))
    return jm

def setup_lda_inference(jm, word_counts,
                        q_topics_base=None,
                        recognition_model=True):
    
    n_docs, n_types = word_counts.shape
    n_topics, n_types2 = jm["topics"].shape
    assert(n_types==n_types2)

    if q_topics_base is None:
        q_topics_base = Gaussian( shape=(n_topics, n_types-1), name="q_topics_base")
    q_topics = TransformedDistribution(q_topics_base, Simplex1, name="q_topics")
    jm["topics"].attach_q(q_topics)
    
    if not jm["docs"].local:
        q_docs_base = Gaussian( shape=(n_docs, n_topics-1), name="q_docs_base")
        q_docs = TransformedDistribution(q_docs_base, Simplex1, name="q_docs")
        jm["docs"].attach_q(q_docs)

    q_words = jm["words"].observe_placeholder()
    batches = BatchGenerator(np.float32(word_counts), batch_size=jm["words"].shape[0])
    jm.register_feed(lambda : {q_words: batches.next_batch()})
    
    if recognition_model:
        recognition_term = deterministic_recognizer(q_topics_base, subtract_entropy=True)
        jm.add_elbo_term(recognition_term)
    else:
        recognition_term = None
