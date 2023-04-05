import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False) # <-- x_train ~ 800,2 & y_train ~ 800,
    
    # *** START CODE HERE ***
    
    # First reshape x_train and y_train into dims,800 and 1,800 arrays.
    x_train = x_train.transpose(); y_train.shape = (1,y_train.shape[0])
    
    # #!!
    # x_train[1,:] = np.log(x_train[1,:])
    
    
    # Train the GDA classifier model
    clf = GDA()
    clf.fit(x_train, y_train)
    
    # Load validation dataset and reshape appropriately
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    x_valid = x_valid.transpose(); y_valid.shape = (1,y_valid.shape[0])
    
    
    # #!!
    # x_valid[1,:] = np.log(x_valid[1,:])
    
    
    # Create predictions and save them
    probs_valid = clf.predict(x_valid)
    np.savetxt(save_path, probs_valid)
    
    # Plot the validation dataset with decision boundary of p = 0.5
    theta_plot = np.concatenate( (clf.theta[0].flatten(), clf.theta[1].flatten() ) )    
    x_valid_plot = np.concatenate( (np.ones([1,100]), x_valid) )
    util.plot(x_valid_plot.transpose(), y_valid.flatten(), theta_plot, save_path[:-3]+'jpg')
    
    print(theta_plot)
    
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
    # Define sigmoid function
    def sigmoid(self, z): return 1/(1+np.exp(-z))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (dim, n_examples).
            y: Training example labels. Shape (1, n_examples).
        """
        # *** START CODE HERE ***
        phi = np.mean(y)
        mu0 = np.sum((1-y)*x,axis=1)/np.sum(1-y,axis=1); mu0.shape=(mu0.shape[0],1)
        mu1 = np.sum(    y*x,axis=1)/np.sum(  y,axis=1); mu1.shape=(mu1.shape[0],1)
        muy = mu0*(1-y) + mu1*y
        sigma = ((x-muy) @ (x-muy).transpose())/y.shape[1]
        
        sigmaInv= np.linalg.inv(sigma)
        thetaTrans  = (-mu0+mu1).transpose() @ sigmaInv
        theta0 = np.log(phi/(1-phi)) + 0.5*( mu0.transpose() @ sigmaInv @ mu0 - mu1.transpose() @ sigmaInv @ mu1 )
        
        self.theta = (theta0, thetaTrans)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (dim, n_examples).

        Returns:
            Outputs of shape (1, n_examples).
        """
        # *** START CODE HERE ***
        z = self.theta[1] @ x + self.theta[0]        
        return self.sigmoid(z)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
          valid_path='ds2_valid.csv',
          save_path='gda_pred_2.txt')
