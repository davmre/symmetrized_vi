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

def elbo_log_perplexity(lmodel,
                        test_rows,
                        n_mc = 1000):

    jm, ratings_feed, mask_feed = lmodel

    # lower bound on p(row) for each test row
    # is given by the expected 

    batch_size = jm["X"].shape[0]
    
    A_lp = jm["A"].expected_logp()
    X_lp = jm["X"].expected_logp()
    q_entropy = jm["A"].q_distribution().entropy()

    test_bounds = []
    sess = test_jm.get_session()
    fd = {words_placeholder: test_counts}
    for i  in range(n_mc):
        lpw, lpd, qe = sess.run((words_lp, docs_lp, q_entropy), feed_dict=fd)
        test_bounds.append(lpw+lpd+qe)
    test_bound = np.mean(test_bounds)

    n_tokens = np.sum(test_counts)

    
    # TODO implement evaluation metric(s) for matrix factorization
    pass    


def holdout_items(test_rows, heldout_items):

    holdout_set = set(heldout_items)
    
    obs_rows = []
    heldout_rows = []
    for (row_mids, row_ratings) in test_rows:

        obs_row = []
        heldout_row = []
        for i in range(len(row_mids)):
            mid, r = row_mids[i], row_ratings[i]
            if mid in holdout_set:
                heldout_row.append((mid, r))
            else:
                obs_row.append((mid, r))
        obs_mids, obs_ratings = zip(*obs_row)
        heldout_mids, heldout_ratings = zip(*heldout_row)
                
        obs_rows.append((np.asarray(obs_mids),
                         np.asarray(obs_ratings)))
        heldout_rows.append((np.asarray(heldout_mids),
                             np.asarray(heldout_ratings)))
        
    return obs_rows, heldout_rows
        
def heldout_predictions(lmodel, obs_rows, heldout_rows):

    # construct as inputs the set of test rows without heldout items
    # feed these to the jm and get out predictions.
    # evaluate
    #  - mean standardized likelihood
    #  - rmse
    # on the heldout predictions

    post = local_posterior(lmodel, obs_rows)

    qRmean, qRstd = post["qRmean"], post["qRstd"]

    pred_ll = 0.0
    pred_sqerr = 0.0
    n = 0
    
    for i, (heldout_mids, heldout_ratings) in enumerate(heldout_rows):
        pred_ratings = qRmean[i, heldout_mids]
        pred_rating_stds = qRstd[i, heldout_mids]
        pred_rv = scipy.stats.norm(loc=pred_ratings, scale=pred_rating_stds)

        pred_ll += pred_rv.logpdf(heldout_ratings)
        pred_sqerr += np.sum((heldout_ratings - pred_ratings)**2)
        n += len(heldout_ratings)

    mse = pred_sqerr / n
    rmse = np.sqrt(mse)

    mean_ll = pred_ll / n

    return rmse, mean_ll
    
    
def group_users(sorted_uids, mids, ratings):
    n_users = np.max(sorted_uids)
    user_rows = []
    for uid in range(n_users):
        matching_idxs = sorted_uids==uid
        user_rows.append((mids[matching_idxs], ratings[matching_idxs]))
    return user_rows



def main():

    # build model w/ movie and batch size
    # build feeder that pulls a set of users at a time
    # problem: either we don't have a full set of users or we have a shape-changing nnz?
    #  - or we pad the nnz somehow. define a fixed-shape nnz, then load with as many users as we can fit, and pad the rest with something
    #  - really the answer here is we should support TF flexible shapes. why wouldn't we? probably it's slow though. 
    import pdb; pdb.set_trace()


    
if __name__ == "__main__":
    main()
