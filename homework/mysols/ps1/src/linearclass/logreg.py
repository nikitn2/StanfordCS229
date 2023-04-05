import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***    
    
    # Train the logistic regression classifier model
    theta_0 = np.array([1e-14]*x_train.shape[1])
    # First train with rough step_size
    nikClassifier = LogisticRegression(theta_0, step_size=0.05, max_iter=int(1e5), eps=1e-5, verbose=True)
    nikClassifier.fit(x_train, y_train)
    
    # Create predictions and save them
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    probs_valid = nikClassifier.predict(x_valid)
    np.savetxt(save_path, probs_valid)
    
    # Plot the validation dataset with decision boundary of p = 0.5
    util.plot(x_valid, y_valid, nikClassifier.theta.transpose(), save_path[:-3]+'jpg')
    
    print(nikClassifier.theta)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0, step_size=0.01, max_iter=1000000, eps=1e-5,
                     verbose=True):
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        # Define cost function
        def cost(theta, x, y):
            
            z = x @ theta
            summand = y*np.log(self.sigmoid(z)) + (1-y)*np.log(1-self.sigmoid(z))
            
            return - np.mean(summand)
        
        # Define gradient finding function
        def grad(theta, x, y): # <--- theta is (dim,)
            
            z = x @ theta #<-- (n,)
            summand = x*(self.sigmoid(z) - y).reshape(-1,1)
            gradient = np.mean(summand,axis=0); gradient.shape=(theta.shape)
            
            return gradient #<-- (dim,1)
        
        # Define Hessian finding function
        def hessian(theta, x, y):
                        
            z = x @ theta #<-- (n,1)
            summand = (1-self.sigmoid(z))*self.sigmoid(z)
            hessian = x.transpose() @ x*np.mean(summand)
            
            return hessian
        
        # Do Newton's update        
        for i in range(0, self.max_iter):
            if self.verbose is True and i % (self.max_iter//10) == 0: print("cost_{} = {}".format(i,cost(self.theta, x, y)))
            delta = self.step_size*np.linalg.inv(hessian(self.theta, x, y)) @ grad(self.theta,x,y)
            if np.linalg.norm(delta) > self.eps: self.theta -= delta
            else: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        return self.sigmoid(z)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
