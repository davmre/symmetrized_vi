import numpy as np
import tensorflow as tf

from elbow.elementary import Gaussian

from lda_core import lda_model, setup_lda_inference

def synthetic_training_data(n_docs, n_types, n_topics, n_tokens=500,
                            alpha_topics=None, alpha_docs=None):
    model = lda_model(n_docs, n_types, n_topics,
                      n_tokens=n_tokens,
                      alpha_topics=alpha_topics,
                      alpha_docs=alpha_docs)
    sampled = model.sample()
    model.session.close()
    return sampled

def run_inference(wordcounts, n_topics,
                  model_kwargs = {},
                  minibatch_size=None,
                  recognition_model=True,
                  adam_rate=0.05,
                  steps=5000):
    n_docs, n_types = wordcounts.shape
    jm = lda_model(n_docs, n_types, n_topics,
                   minibatch_size=minibatch_size,
                   **model_kwargs)
    setup_lda_inference(jm, wordcounts,
                        recognition_model=recognition_model)
    jm.train(adam_rate=adam_rate, steps=steps)

    weights = jm["words"].inference_weights
    if weights is not None:
        weight_vals = {key : jm.session.run(v) for key, v in weights.items()}
    else:
        weight_vals = None
        
    post = jm.posterior()
    jm.session.close()
    return post, weight_vals


def optimize_test_model(test_counts,
                        posterior,
                        adam_rate=0.05,
                        steps=1000,
                        weights=None,
                        alpha_docs=None):

    n_test, n_types = test_counts.shape

    topic_means = posterior["q_topics"]["mean"]
    topic_stds =  posterior["q_topics"]["std"]

    n_topics, n_types1 = topic_means.shape
    assert(n_types==n_types1 + 1)

    if weights is None:
        nbatch = None
    else:
        nbatch = n_test
    
    with tf.name_scope("ldamodel"):
        test_jm = lda_model(n_test,
                            n_types,
                            n_topics,
                            minibatch_size=nbatch,
                            inference_weights=weights,
                            alpha_docs=alpha_docs)

    with tf.name_scope("tbase"):
        q_topics_base = Gaussian(mean=tf.constant(topic_means),
                                 std=tf.constant(topic_stds))

    with tf.name_scope("ldainference"):

        setup_lda_inference(test_jm, test_counts,
                            q_topics_base=q_topics_base,
                            recognition_model=False)

    with tf.name_scope("elbo"):
        elbo = test_jm.construct_elbo()

    if weights is None:
        test_jm.train(adam_rate=adam_rate, steps=steps)


        
    return test_jm

def elbo_log_perplexity(test_counts,
                        test_jm,
                        n_mc = 1000):
    n_test, n_types = test_counts.shape
    
    words_lp = test_jm["words"].expected_logp()
    docs_lp = test_jm["docs"].expected_logp()
    q_entropy = test_jm["docs"].q_distribution().entropy()

    test_bounds = []
    sess = test_jm.get_session()
    words_placeholder = test_jm["words"].q_distribution().tf_value
    fd = {words_placeholder: test_counts}
    for i  in range(n_mc):
        lpw, lpd, qe = sess.run((words_lp, docs_lp, q_entropy), feed_dict=fd)
        test_bounds.append(lpw+lpd+qe)
    test_bound = np.mean(test_bounds)

    n_tokens = np.sum(test_counts)
    
    return - test_bound / n_tokens
    

def main():

    #n_docs_train = 1000
    #n_docs_test = 200
    #n_types = 3000
    #n_tokens = 800
    #n_topics = 20

    n_docs_train = 300
    n_docs_test = 20
    n_types = 60
    n_tokens = 200
    n_topics = 20

    alpha_topics = 0.4
    alpha_docs = 50./n_topics

    """
    words  = np.load("newsgroups_stupid.npz")["arr_0"].item().todense()
    n_docs, n_types = words.shape
    n_docs_train = n_docs-200

    minibatch_size = 128
    """

    minibatch_size = n_docs_train
    data = synthetic_training_data(n_docs=n_docs_train+n_docs_test,
                                   n_types=n_types,
                                   n_topics=n_topics,
                                   n_tokens=n_tokens,
                                   alpha_topics = alpha_topics,
                                   alpha_docs = alpha_docs)
    words = data["words"]

    
    train_wordcounts = words[:n_docs_train]
    test_wordcounts = words[n_docs_train:]

    
    posterior1, weights1 = run_inference(train_wordcounts,
                                         n_topics,
                                         recognition_model=False,
                                         minibatch_size=64,
                                         adam_rate=0.02,
                                         steps=1500)


    posterior2, weights2 = run_inference(train_wordcounts,
                                         n_topics,
                                         recognition_model=False,
                                         minibatch_size=minibatch_size,
                                         adam_rate=0.02,
                                         steps=1500)


    test_jm1 = optimize_test_model(test_wordcounts,
                                   posterior1,
                                   adam_rate=0.01,
                                   steps=1000,
                                   weights=weights1,
                                   alpha_docs=alpha_docs)
    log_perplexity1 = elbo_log_perplexity(test_wordcounts,
                                          test_jm1)

    test_jm2 = optimize_test_model(test_wordcounts,
                                   posterior2,
                                   adam_rate=0.05,
                                   steps=1000,
                                   weights=weights2,
                                   alpha_docs=alpha_docs)

    
    
    log_perplexity2 = elbo_log_perplexity(test_wordcounts,
                                          test_jm2)

    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
