import numpy as np
import tensorflow as tf

from elbow.elementary import Gaussian

from mf_core import *


def run_inference(X, k,
                  sigma_n = 2.0,
                  posterior="analytic",
                  adam_rate=0.05,
                  steps=5000):
    n, m = X.shape
    jm, q_C = build_simple_mf_model(n, m, k, sigma_n=sigma_n)

    if posterior=="analytic":
        mf_analytic_inference(jm)
    elif posterior=="rot":
        mf_rot_inference(jm)
    elif posterior=="map":
        mf_map_inference(jm)
    elif posterior=="iso":
        mf_isotropic_inference(jm)
    else:
        raise Exception("unrecognized inference mode %s" % posterior) 

    jm.register_feed(lambda : {qC : np.float32(X)})
    jm.train(adam_rate=adam_rate, steps=steps)

    post = jm.posterior()
    jm.session.close()
    return post

def elbo_log_perplexity(test_ratings,
                        posterior,
                        adam_rate,
                        steps,
                        n_mc = 1000):
    # TODO implement evaluation metric(s) for matrix factorization
    pass    

def main():

    # todo set up sparse-observation matrix factorization
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
