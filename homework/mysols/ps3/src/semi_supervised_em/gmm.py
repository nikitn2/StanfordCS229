import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)
alpha = 20.     # Weight for the labeled examples

def computeMu(x, w):
    
    numerator   = w @ x
    denominator = np.sum(w,axis=1)
    return numerator/denominator[:,np.newaxis]

def computeSigma(x, w, mu):
    
    x_st        = x[:,np.newaxis,:] - mu[np.newaxis,:, :]
    numerator   = np.einsum('ji,ijk,ijl->jkl', w, x_st, x_st)
    denominator = w.sum(axis=1)
    return numerator/denominator[:, np.newaxis, np.newaxis]

def updateEgetLL(x, mu, sigma, phi):
    
    # Initialise
    n             = x.shape[0]
    
    # First get p_xz_given_theta
    p_xz_given_theta = np.zeros((n,K))
    for i in range(0,K): p_xz_given_theta[:,i] = multivariate_normal.pdf(x, mu[i,:], sigma[i,...])*phi[i]
    
    # Now p_x_given_theta
    p_x_given_theta = p_xz_given_theta.sum(axis=1)
    
    # Now get one of the returns, p_z_given_xTheta
    p_z_given_xTheta = p_xz_given_theta/p_x_given_theta[:,np.newaxis]
    
    # And then for the other, log-likelihood
    ll = np.log(p_xz_given_theta.sum(axis=1)).sum(axis=0)
    
    return p_z_given_xTheta.T, ll

def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # # Split into labeled and unlabeled examples
    labeled_idxs   = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    
    # Make sure unabelled data is shuffled and set z_tilde to int
    x = x[np.random.permutation(len(x))]
    z_tilde = z_tilde.astype(int, copy=False)
    
    # Send labeled examples to the back
    d            = x.shape[1]
    n_unlabelled = len(x)
    n_labelled   = len(x_tilde)
    x = np.append(x, x_tilde, axis=0)
    
    # Create the w 
    w       = np.ones((K,n_unlabelled))/K
    w_tilde = np.zeros((K, n_labelled)); w_tilde[z_tilde[:,0], np.arange(0,n_labelled)] = alpha
    w       = np.append(w, w_tilde, axis = 1)
    
    # And randomly initialise z
    if is_semi_supervised:
        n = n_unlabelled + n_labelled
        z = np.random.randint(0, K, n)
    else: 
        n = n_unlabelled
        z = np.random.randint(0, K, n_unlabelled )
    
    # And compute mu and sigma
    mu = np.zeros((K,d)); sigma = np.zeros((K,d,d))
    for i in range(0,K): 
        mu[i,:] = np.mean(x[:n][z==i], axis = 0)
        sigma[i,...] = np.cov(x[:n][z==i].T)
    
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K)/K
    
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    # ALREADY DONE.
    
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_em(x, w, phi, mu, sigma, n_unlabelled)
    else:
        w = run_em(x[:n_unlabelled,...], w[...,:n_unlabelled], phi, mu, sigma, n_unlabelled)
    
    # Plot your predictions
    z_pred = w.argmax(axis=0)
    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma, n_unlabelled):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-4  # Convergence threshold
    max_iter = 2000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):

        # *** START CODE HERE
        # (1) E-step: Update your estimates in w (and compute ll while at it)
        prev_ll = ll
        w[:,:n_unlabelled], ll = updateEgetLL(x[:n_unlabelled,:], mu, sigma, phi)
        
        # (2) M-step: Update the model parameters phi, mu, and sigma
        mu    = computeMu(x, w)
        sigma = computeSigma(x, w, mu)
        phi   = w.sum(axis=1)/w.sum()

        # Iterate and repeat until convergence
        print(ll, it)
        it+=1
        # *** END CODE HERE ***

    return w

def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
