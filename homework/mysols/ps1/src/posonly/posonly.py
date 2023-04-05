import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')
    
    # *** START CODE HERE ***
    
    # Basic code that is rerun
    def trainPredictSaveAndPlot(theta0, features_train, labels_train, features_test, labels_test, labels_test_plot = None, savepath = None):
        
        # Train
        clf = LogisticRegression(theta_0, step_size=0.05, max_iter=int(1e5), eps=1e-7, verbose=True)
        clf.fit(features_train, labels_train); clf.step_size=clf.step_size**2; clf.fit(features_train, labels_train);
        
        # Predict
        probs_test = clf.predict(features_test)
        
        # Plot and save
        if labels_test_plot is None: labels_test_plot = labels_test
        if savepath is not None: 
            util.plot(features_test, labels_test_plot, clf.theta, savepath[:-3]+'jpg')
            np.savetxt(savepath, probs_test)
        
        return clf
    
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path, 't', add_intercept=True)
    x_test,  t_test =  util.load_dataset(test_path,  't', add_intercept=True)
    theta_0 = np.array([1e-14]*x_train.shape[1])
    trainPredictSaveAndPlot(theta_0, x_train, t_train, x_test, t_test, savepath = output_path_true)
    
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, 'y', add_intercept=True)
    x_test,  y_test =  util.load_dataset(test_path,  'y', add_intercept=True)
    clf = trainPredictSaveAndPlot(theta_0, x_train, y_train, x_test, y_test, labels_test_plot =  t_test, savepath = output_path_naive)
    
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    
    # First get corrected labels
    x_valid,  y_valid =  util.load_dataset(valid_path,  'y', add_intercept=True)
    hx_valid = clf.predict(x_valid)
    alphaCorrFactr = np.mean(hx_valid[y_valid == 1]); print(r"Found that $\alpha={}$.".format(alphaCorrFactr))
    hx_test_corrected = clf.predict(x_test)/alphaCorrFactr
    labels_test_corrected = np.round(hx_test_corrected)
    theta_corrected = clf.theta + [np.log(2/alphaCorrFactr -1),0,0]
    
    # Now plot and save
    util.plot(x_test, labels_test_corrected, theta_corrected, output_path_adjusted[:-3]+'jpg')
    np.savetxt(output_path_adjusted, hx_test_corrected)
    

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
