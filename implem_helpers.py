import numpy as np
from scipy.special import softmax

np.random.seed(509)


def loss_x(x_new, x_initial):
    """
    Constrains the mapping to Z to be good description of X.
    Prototpyes should retain as much initial info as possible.

    difference is measured by squared sum of difference


    ARGS:
    x_new - Prototypes
    x_initial - raw data
    """
    return np.mean(np.sum(np.square((x_new - x_initial))))


def loss_y(y_true, y_predicted):
    """
    This loss term requires that the prediction of y is as accurate as possible:

    Computes log loss

    ARGS:
    y_true - (num_examples, )
    y_predicted - (num_examples, )
    """
    # logarithm is undefined in 0 which means y cant be 0 or 1 => we clip it
    y_true = np.clip(y_true, 1e-6, 0.999)
    y_predicted = np.clip(y_predicted, 1e-6, 0.999)

    log_loss = np.sum(y_true * np.log(y_predicted) +
                      (1. - y_true) * np.log(1. - y_predicted)) / len(y_true)

    return -log_loss


def loss_z(M_k_sensitive, M_k_non_sensitive):
    """
    Ensures statistical parity

    Calculates L1 distance

    Args:
    M_k_sensitive - (num_prototypes, )
    M_k_non_sensitive - (num_prototypes, )
    """
    return np.sum(np.abs(M_k_sensitive - M_k_non_sensitive))


def distances(X, v, alpha):
    """
    Calculates distance between initial data and each of the prototypes 
    Formula -> euclidean(x, v * alpha) (alpha is weight for each feature)

    ARGS:
    X - (num_examples, num_features)
    v - (num_prototypes, num_features)
    alpha - (num_features, 1)

    returns:
    dists - (num_examples, num_prototypes)
    """
    num_examples = X.shape[0]
    num_prototypes = v.shape[0]
    dists = np.zeros(shape=(num_examples, num_prototypes))

    # X = X.values  # converting to NumPy, this is needed in case you pass dataframe
    for i in range(num_examples):
        dist = np.square(X[i] - v)  # squarred distance
        dist_alpha = np.multiply(dist, alpha)  # multiplying by weights
        sum_ = np.sum(dist_alpha, axis=1)
        dists[i] = sum_

    return dists


def M_nk(dists):
    """
    define Mn,k as the probability that x maps to v

    Given the definitions of the prototypes as points in
    the input space, a set of prototypes induces a natural
    probabilistic mapping from X to Z via the softmax

    Since we already have distances calcutated we just map them to probabilities

    NOTE:
    minus distance because smaller the distance better the mapping

    ARGS:
    dists - (num_examples, num_prototypes)

    Return :
    mappings - (num_examples, num_prototypes)
    """
    return softmax(-dists, axis=1)  # specifying axis is important


def M_k(M_nk):
    """
    Calculate mean of the mapping for each prototype

    ARGS:
    M_nk - (num_examples, num_prototypes)

    Returns:
    M_k - mean of the mappings (num_prototypes, )
    """
    return np.mean(M_nk, axis=0)


def x_n_hat(M_nk, v):
    """
    Gets new representation of the data, 
    Performs simple dot product

    ARGS:
    M_nk - (num_examples, num_prototypes)
    v - (num_prototypes, num_features)

    Returns:
    x_n_hat - (num_examples, num_features)
    """
    return M_nk @ v


def y_hat(M_nk, w):
    """
    Function calculates labels in the new representation space
    Performs simple dot product

    ARGS:
    M_nk - (num_examples, num_prototypes)
    w - (num_prototypes, )

    returns:
    y_hat - (num_examples, )
    """
    return M_nk @ w


def optim_objective(params, data_sensitive, data_non_sensitive, y_sensitive,
                    y_non_sensitive,  inference=False, NUM_PROTOTYPES=10, A_x=0.01, A_y=0.1, A_z=0.5,
                    print_every=100):
    """
    Function gathers all the helper functions to calculate overall loss

    This is further passed to l-bfgs optimizer 

    ARGS:
    params - vector of length (2 * num_features + NUM_PROTOTYPES + NUM_PROTOTYPES * num_features)
    data_sensitive - instances belonging to senstive group (num_sensitive_examples, num_features)
    data_non_sensitive - similar to data_sensitive (num_non_senitive_examplesm num_features)
    y_sensitive - labels for sensitive group (num_sensitive_examples, )
    y_non_sensitive - similar to y_sensitive
    inference - (optional) if True than will return new dataset instead of loss
    NUM_PROTOTYPES - (optional), default 10
    A_x - (optional) hyperparameters for loss_X, default 0.01
    A_y - (optional) hyperparameters for loss_Y, default 1
    A_z - (optional) hyperparameters for loss_Z, default 0.5
    print_every - (optional) how often to print loss, default 100
    returns:
    if inference - False :
    float - A_x * L_x + A_y * L_y + A_z * L_z 
    if inference - True:
    x_hat_sensitive, x_hat_non_sensitive, y_hat_sensitive, y_hat_non_sensitive
    """
    optim_objective.iters += 1

    num_features = data_sensitive.shape[1]
    # extract values for each variable from params vector
    alpha_non_sensitive = params[:num_features]
    alpha_sensitive = params[num_features:2 * num_features]
    w = params[2 * num_features:2 * num_features + NUM_PROTOTYPES]
    v = params[2 * num_features + NUM_PROTOTYPES:].reshape(NUM_PROTOTYPES, num_features)

    dists_sensitive = distances(data_sensitive, v, alpha_sensitive)
    dists_non_sensitive = distances(data_non_sensitive, v, alpha_non_sensitive)

    # get probabilities of mappings
    M_nk_sensitive = M_nk(dists_sensitive)
    M_nk_non_sensitive = M_nk(dists_non_sensitive)

    # M_k only used for calcilating loss_y(statistical parity)
    M_k_sensitive = M_k(M_nk_sensitive)
    M_k_non_sensitive = M_k(M_nk_non_sensitive)
    L_z = loss_z(M_k_sensitive, M_k_non_sensitive)  # stat parity

    # get new representation of data
    x_hat_sensitive = x_n_hat(M_nk_sensitive, v)
    x_hat_non_sensitive = x_n_hat(M_nk_non_sensitive, v)
    # calculates how close new representation is to original data
    L_x_sensitive = loss_x(data_sensitive, x_hat_sensitive)
    L_x_non_sensitive = loss_x(data_non_sensitive, x_hat_non_sensitive)

    # get new values for labels
    y_hat_sensitive = y_hat(M_nk_sensitive, w)
    y_hat_non_sensitive = y_hat(M_nk_non_sensitive, w)
    # ensure how good new predictions are(log_loss)
    L_y_sensitive = loss_y(y_sensitive, y_hat_sensitive)
    L_y_non_sensitive = loss_y(y_non_sensitive, y_hat_non_sensitive)

    L_x = L_x_sensitive + L_x_non_sensitive
    L_y = L_y_sensitive + L_y_non_sensitive

    loss = A_x * L_x + A_y * L_y + A_z * L_z

    if optim_objective.iters % print_every == 0:
        print(f'loss on iteration {optim_objective.iters} : {loss}, L_x - {L_x * A_x} L_y - {L_y * A_y} L_z - {L_z * A_z}')
    
    if not inference:
        return loss
    if inference:
        return x_hat_sensitive, x_hat_non_sensitive, y_hat_sensitive, y_hat_non_sensitive

optim_objective.iters = 0
