#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import logging
import multiprocessing
import os
from collections import Counter, defaultdict
from datetime import datetime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import mlrose
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import isspmatrix
from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             make_scorer, plot_confusion_matrix)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     learning_curve, train_test_split,
                                     validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import ignore_warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(relativeCreated)6d [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Configure output directory
OUTPUT_DIRECTORY = './output/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
IMAGES_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'images')
LOGS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'logs')
os.makedirs(IMAGES_DIRECTORY)
os.makedirs(LOGS_DIRECTORY)

# Variables to control algorithm runs
# LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Number of inputs in a state
# NUM_RUNS = 10           # Number of times to run experiment to get an average

LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Number of inputs in a state
NUM_RUNS = 10           # Number of times to run experiment to get an average


# https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#threadpool-configuration
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(multiprocessing.cpu_count())

# QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-cyc'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-cyc'

# libEGL warning: DRI2: failed to open swrast (search paths /usr/lib64/dri)
# -> sudo yum install mesa-libGL.x86_64

np.set_printoptions(precision=2)        # Trim output to 2 decimal places so output is pretty


class Expr(object):
    def __init__(self, path, threads, random_state, cv=5, verbose=False):
        self.dataset = Dataset(path=path, threads=threads, verbose=verbose)
        self.threads = threads
        self.random_state = random_state
        self.cv = cv        # Number of cross validation folds
        self.verbose = verbose
        self.scorer = make_scorer(balanced_accuracy_score)

    def do_prepare_dataset(self, test_size=0.2, shuffle=True):
        """Preprocess dataset for experiment"""
        logging.info('Load, Split and Preprocess {} Dataset for Experiment'.format(self.dataset.name))
        self.dataset.load()
        self.dataset.split(test_size=test_size, random_state=self.random_state, shuffle=shuffle)
        self.dataset.scale()
        self.test_size = test_size
        self.shuffle = shuffle

    def do_decision_tree(self):
        start = timer()
        logging.info('Start DecisionTree algorithm with Dataset {}'.format(self.dataset.name))

        # max_depths = np.arange(1, 31, 1)
        # param_grid = {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': max_depths,
        #     'class_weight': ['balanced', None]
        # }

        # Best Params from Previous GSCV
        param_grid = {}
        if self.dataset.name == 'CreditDefaults':
            param_grid = {
                'criterion': ['gini'],
                'max_depth': [5],
                'class_weight': ['balanced']
            }
        elif self.dataset.name == 'PenDigits':
            param_grid = {
                'criterion': ['entropy'],
                'max_depth': [11],
                'class_weight': [None]
            }

        estimator = DecisionTreeClassifier(random_state=self.random_state)
        gscv = self.do_grid_search(estimator=estimator, param_grid=param_grid)

        duration = timer() - start
        logging.info('DecisionTree algorithm with Dataset {}. GSCV Duration: {:.4f} seconds'.format(self.dataset.name, duration))

        self.save_grid_search(gscv)
        self.save_classification_report(gscv)
        self.plot_confusion_matrix(gscv)
        self.plot_learning_curve(gscv)

        # validation_params = {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': max_depths,
        #     'class_weight': ['balanced', None]
        # }
        # for param_name, param_range in validation_params.items():
        #     train_scores, test_scores = self.do_validation_curve(gscv, param_name=param_name, param_range=param_range)
        #     self.save_validation_curve(gscv, train_scores, test_scores, param_name=param_name)
        #     self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='linear')

        duration = timer() - start
        logging.info('Finish DecisionTree algorithm with Dataset {}. Duration: {:.4f} seconds'.format(self.dataset.name, duration))

    def do_k_nearest_neighbor(self):
        start = timer()
        logging.info('Start k-NearestNeighbor algorithm with Dataset {}'.format(self.dataset.name))

        # n_neighbors = np.arange(1, 21, 1)
        # param_grid = {
        #     'metric': ['manhattan', 'euclidean', 'minkowski', 'chebyshev'],
        #     'n_neighbors': n_neighbors,
        #     'weights': ['uniform', 'distance']
        # }

        # Best Params from Previous GSCV
        param_grid = {}
        if self.dataset.name == 'CreditDefaults':
            param_grid = {
                'metric': ['euclidean'],
                'n_neighbors': [10],
                'weights': ['distance']
            }
        elif self.dataset.name == 'PenDigits':
            param_grid = {
                'metric': ['manhattan'],
                'n_neighbors': [1],
                'weights': ['uniform']
            }

        estimator = KNeighborsClassifier(n_jobs=self.threads, weights='distance')
        gscv = self.do_grid_search(estimator=estimator, param_grid=param_grid)

        duration = timer() - start
        logging.info('k-NearestNeighbor algorithm with Dataset {}. GSCV Duration: {:.4f} seconds'.format(self.dataset.name, duration))

        self.save_grid_search(gscv)
        self.save_classification_report(gscv)
        self.plot_confusion_matrix(gscv)
        self.plot_learning_curve(gscv)

        # validation_params = {
        #     'metric': ['manhattan', 'euclidean', 'minkowski', 'chebyshev'],
        #     'n_neighbors': n_neighbors,
        #     'weights': ['uniform', 'distance']
        # }
        # for param_name, param_range in validation_params.items():
        #     train_scores, test_scores = self.do_validation_curve(gscv, param_name=param_name, param_range=param_range)
        #     self.save_validation_curve(gscv, train_scores, test_scores, param_name=param_name)
        #     self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='linear')

        duration = timer() - start
        logging.info('Finish k-NearestNeighbor algorithm with Dataset {}. Duration: {:.4f} seconds'.format(self.dataset.name, duration))

    def do_boosting(self):
        start = timer()
        logging.info('Start Boosting algorithm with Dataset {}'.format(self.dataset.name))

        # n_estimators = [int(v) for v in np.logspace(0, 9, 10, base=2)]
        # algorithms = ['SAMME', 'SAMME.R']
        # learning_rates = np.logspace(-7, 0, 8, base=2)
        # base_estimators = [DecisionTreeClassifier(max_depth=n) for n in range(1, 12)]
        # param_grid = {
            # 'n_estimators': n_estimators,
            # 'learning_rate': learning_rates,
            # 'base_estimator': base_estimators,
            # 'algorithm': algorithms
        # }

        # Best Params from Previous GSCV
        param_grid = {}
        if self.dataset.name == 'CreditDefaults':
            param_grid = {
                'n_estimators': [4],
                'learning_rate': [1],
                'base_estimator': [DecisionTreeClassifier(max_depth=2, random_state=self.random_state)],
                'algorithm': ['SAMME.R']
            }
        elif self.dataset.name == 'PenDigits':
            param_grid = {
                'n_estimators': [128],
                'learning_rate': [1],
                'base_estimator': [DecisionTreeClassifier(max_depth=10, random_state=self.random_state)],
                'algorithm': ['SAMME']
            }

        estimator = AdaBoostClassifier(random_state=self.random_state)
        gscv = self.do_grid_search(estimator=estimator, param_grid=param_grid)

        duration = timer() - start
        logging.info('Boosting algorithm with Dataset {}. GSCV Duration: {:.4f} seconds'.format(self.dataset.name, duration))

        self.save_grid_search(gscv)
        self.save_classification_report(gscv)
        self.plot_confusion_matrix(gscv)
        self.plot_learning_curve(gscv)

        # validation_params = {
        #     'n_estimators': n_estimators,
        #     'learning_rate': learning_rates,
        #     'base_estimator': base_estimators,
        #     'algorithm': algorithms
        # }
        # for param_name, param_range in validation_params.items():
        #     train_scores, test_scores = self.do_validation_curve(gscv, param_name=param_name, param_range=param_range)
        #     self.save_validation_curve(gscv, train_scores, test_scores, param_name=param_name)
        #     self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='log')

        duration = timer() - start
        logging.info('Finish Boosting algorithm with Dataset {}. Duration: {:.4f} seconds'.format(self.dataset.name, duration))

    def do_neural_network(self, estimator=None, param_grid=None):
        start = timer()
        logging.info('Start Neural Network algorithm with Dataset {}'.format(self.dataset.name))

        do_validation = True
        if param_grid is None:      # Search for hypeparameters
            alphas = np.logspace(-5, -3, 3)
            hiddens = [(h,) * l for l in [1, 2] for h in [2, 4, 8, 16, 32, 64, 128, 256]]      # [(2,), (64,), (256,), (2, 2), (64, 64), (256, 256)]
            max_iters = [int(v) for v in np.logspace(0, 8, 9, base=2)]     # 1, 2, 4, 8, ..., 2048

            param_grid = {
                'alpha': alphas,
                'hidden_layer_sizes': hiddens,
                'max_iter': max_iters,
            }
        elif param_grid == 'best':  # Best Params from Previous GSCV
            do_validation = False
            if self.dataset.name == 'CreditDefaults':
                param_grid = {
                    'alpha': [0.001],
                    'hidden_layer_sizes': [(64, 64)],
                    'max_iter': [16],
                }
            elif self.dataset.name == 'PenDigits':
                param_grid = {
                    'alpha': [0.0001],
                    'hidden_layer_sizes': [(128, 128)],
                    'max_iter': [128],
                }
            else:
                param_grid = {}

        if estimator is None:
            estimator = MLPClassifier(random_state=self.random_state)

        gscv = self.do_grid_search(estimator=estimator, param_grid=param_grid)

        duration = timer() - start
        logging.info('Neural Network algorithm with Dataset {}. GSCV Duration: {:.4f} seconds'.format(self.dataset.name, duration))

        self.save_grid_search(gscv)
        self.save_classification_report(gscv)
        self.plot_confusion_matrix(gscv)
        self.plot_learning_curve(gscv)

        if do_validation:
            validation_params = param_grid
            for param_name, param_range in validation_params.items():
                train_scores, test_scores = self.do_validation_curve(gscv, param_name=param_name, param_range=param_range)
                self.save_validation_curve(gscv, train_scores, test_scores, param_name=param_name)
                if param_name in ['hidden_layer_sizes', 'hidden_nodes', 'algorithm', 'early_stopping']:
                    self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='linear')
                else:
                    self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='log')

        duration = timer() - start
        logging.info('Finish Neural Network algorithm with Dataset {}. Duration: {:.4f} seconds'.format(self.dataset.name, duration))
        return gscv

    def do_support_vector_machine(self):
        start = timer()
        logging.info('Start Support Vector Machine algorithm with Dataset {}'.format(self.dataset.name))

        # Cs = np.logspace(-7, 2, 10, base=2)
        # kernels = ['rbf', 'linear']
        # gammas = ['scale', 'auto']

        # param_grid = {
            # 'C': Cs,
            # 'kernel': kernels,
            # 'gamma': gammas,
        # }

        # Best Params from Previous GSCV
        param_grid = {}
        if self.dataset.name == 'CreditDefaults':
            param_grid = {
                'C': [4],
                'kernel': ['rbf'],
                'gamma': ['scale'],
            }
        elif self.dataset.name == 'PenDigits':
            param_grid = {
                'C': [4],
                'kernel': ['rbf'],
                'gamma': ['scale'],
            }

        estimator = SVC(kernel='rbf', random_state=self.random_state)
        gscv = self.do_grid_search(estimator=estimator, param_grid=param_grid)

        duration = timer() - start
        logging.info('Support Vector Machine algorithm with Dataset {}. GSCV Duration: {:.4f} seconds'.format(self.dataset.name, duration))

        self.save_grid_search(gscv)
        self.save_classification_report(gscv)
        self.plot_confusion_matrix(gscv)
        self.plot_learning_curve(gscv)

        # validation_params = {
        #     'C': Cs,
        #     'kernel': kernels,
        #     'gamma': gammas,
        # }
        # for param_name, param_range in validation_params.items():
        #     train_scores, test_scores = self.do_validation_curve(gscv, param_name=param_name, param_range=param_range)
        #     self.save_validation_curve(gscv, train_scores, test_scores, param_name=param_name)
        #     if param_name == 'C':
        #         self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='log')
        #     else:
        #         self.plot_validation_curve(gscv, train_scores, test_scores, param_name=param_name, param_range=param_range, xscale='linear')

        duration = timer() - start
        logging.info('Finish Support Vector Machine algorithm with Dataset {}. Duration: {:.4f} seconds'.format(self.dataset.name, duration))

    def get_estimator_name(self, estimator):
        """Get a nice label from estimator class name"""
        name = str(estimator).split('(',)[0] or 'UnnamedEstimator'
        return name

    def do_grid_search(self, estimator, param_grid, refit=True):
        gscv = GridSearchCV(estimator=estimator, n_jobs=self.threads, param_grid=param_grid, scoring=self.scorer, refit=refit, cv=self.cv)
        gscv.fit(self.dataset.X_train, self.dataset.Y_train)
        return gscv

    def save_grid_search(self, gscv):
        """Save GridSearch CrossValidation results"""
        # cv_cv_results_
        estimator_name = self.get_estimator_name(gscv.estimator)
        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-cv_results.csv'.format(self.dataset.name, estimator_name))
        pd.DataFrame(gscv.cv_results_).to_csv(path, index=False)

        # best_params_
        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-best_params.csv'.format(self.dataset.name, estimator_name))
        pd.DataFrame([gscv.best_params_]).to_csv(path, index=False)

        # best_estimator_: All params of best_estimator, including default params
        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-best_estiamtor_.csv'.format(self.dataset.name, estimator_name))
        pd.DataFrame([gscv.best_estimator_.get_params()]).to_csv(path, index=False)

        # Algorithm Summary: Combine with other algorithms
        path = os.path.join(LOGS_DIRECTORY, 'Summary.csv'.format())
        row = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Estimator': estimator_name,
            'Dataset': self.dataset.name,
            'BestParams': gscv.best_params_,            
            'TestScore': gscv.score(self.dataset.X_test, self.dataset.Y_test),
        }
        pd.DataFrame([row]).to_csv(path, index=False, mode='a')

    def plot_confusion_matrix(self, gscv):
        """Plot Normalized Confusion Matrix"""
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Plot Normalized Confusion Matrix for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        path = os.path.join(IMAGES_DIRECTORY, '{}-{}-gscv-confusion_matrix.png'.format(self.dataset.name, estimator_name))
        plot = plot_confusion_matrix(gscv, self.dataset.X_test, self.dataset.Y_test, cmap=plt.cm.Blues, normalize='true', values_format='.2f')
        plot.ax_.set_title('Confusion Matrix - {} - {}'.format(self.dataset.name, estimator_name))
        plt.savefig(path)
        return plt

    def plot_learning_curve(self, gscv):
        """Plot Learning Curve
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
        https://docs.w3cub.com/scikit_learn/auto_examples/model_selection/plot_learning_curve/#sphx-glr-download-auto-examples-model-selection-plot-learning-curve-py
        """
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Plot Learning, Scalability, and Performance Curves for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        train_sizes = np.linspace(0.1, 1, 10)

        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            gscv.estimator, self.dataset.X_train, self.dataset.Y_train, cv=self.cv, n_jobs=self.threads, 
            train_sizes=train_sizes, return_times=True, scoring=self.scorer)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        score_times_std = np.std(score_times, axis=1)

        # Learning Curves
        # plt.figure(1)
        plt.clf()
        plt.title('Learning Curves - {} - {}'.format(self.dataset.name, estimator_name))
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        plt.legend(loc='best')
        path = os.path.join(IMAGES_DIRECTORY, '{}-{}-gscv-learning_curve.png'.format(self.dataset.name, estimator_name))
        plt.savefig(path)
        # plt.close(1)

        # Scalability - Number of Samples vs Fit Times
        # plt.figure(2)
        plt.clf()
        plt.title('Model Scalability - {} - {}'.format(self.dataset.name, estimator_name))
        plt.xlabel('Training Samples')
        plt.ylabel('Fit Times')
        plt.grid()
        plt.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1, color='r')
        plt.plot(train_sizes, fit_times_mean, 'o-', color='r', label='Fit Times')
        plt.legend(loc='best')
        path = os.path.join(IMAGES_DIRECTORY, '{}-{}-gscv-learning_curve-scalability.png'.format(self.dataset.name, estimator_name))
        plt.savefig(path)
        # plt.close(2)

        # Performance - Fit Times vs Score
        # plt.figure(3)
        plt.clf()
        plt.title('Model Performance - {} - {}'.format(self.dataset.name, estimator_name))
        plt.xlabel('Fit Times')
        plt.ylabel('Score')
        plt.grid()
        plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')
        plt.plot(fit_times_mean, test_scores_mean, 'o-', color='r', label='Test Score')
        plt.legend(loc='best')
        path = os.path.join(IMAGES_DIRECTORY, '{}-{}-gscv-learning_curve-performance.png'.format(self.dataset.name, estimator_name))
        plt.savefig(path)
        # plt.close(3)

    def do_validation_curve(self, gscv, param_name, param_range):
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Computing Validation Curves Scores for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        train_scores, test_scores = validation_curve(gscv.estimator, self.dataset.X_train, self.dataset.Y_train,
            param_name=param_name, param_range=param_range, cv=self.cv, scoring=self.scorer, n_jobs=self.threads)
        return train_scores, test_scores

    def save_validation_curve(self, gscv, train_scores, test_scores, param_name):
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Save Validation Curves Scores for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-validation_curve-{}-train_scores.csv'.format(self.dataset.name, estimator_name, param_name))
        pd.DataFrame(train_scores).to_csv(path)

        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-validation_curve-{}-test_scores.csv'.format(self.dataset.name, estimator_name, param_name))
        pd.DataFrame(test_scores).to_csv(path)

    def plot_validation_curve(self, gscv, train_scores, test_scores, param_name, param_range, xscale='log'):
        """Plot Validation Curve
        References:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
        """
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Plot Validation Curves for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Hook to enable base_estimator param in Boosting. Convert from object to integer
        if isinstance(param_range[0], DecisionTreeClassifier):
            param_range = [v.max_depth for v in param_range]

        # Hook to enable class_weight param in Decision Trees. Convert None to string
        if param_name == 'class_weight':
            param_range = [str(v) for v in param_range]

        # Hook to enable hidden_nodes param in mlrose.NueralNetwork. Convert array to string
        if param_name == 'hidden_nodes':
            param_range = [str(v) for v in param_range]

        # plt.figure(4)        # Create figure here so we can set ticks in hidden_layer_sizes hook
        plt.clf()

        # Hook to enable hidden_layer_sizes in Neural Network. Convert from tuple to string
        if isinstance(param_range[0], tuple):
            param_range = [str(v) for v in param_range]
            plt.xticks(rotation=90)     # Tuples in hidden sizes do not show legibly by default. Rotate to stop overlap

        plt.title('Validation Curve - {} - {}'.format(self.dataset.name, estimator_name))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.grid()
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

        if xscale == 'log':
            plt.semilogx(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
            plt.semilogx(param_range, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        elif xscale == 'linear':
            plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
            plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        else:
            raise ValueError('Unknown xscale: {}'.format(xscale))

        plt.legend(loc='best')
        path = os.path.join(IMAGES_DIRECTORY, '{}-{}-gscv-validation_curve-{}.png'.format(self.dataset.name, estimator_name, param_name))
        plt.savefig(path)
        # plt.close(4)

    def plot_datset_class_distribution(self):
        """Show balance of classes in dataset by plotting the frequency of each class"""
        # plt.figure(5)
        plt.clf()
        ax = sns.countplot(data=self.dataset.data, x=self.dataset.data.shape[1]-1)
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_title('Class Distribution - {}'.format(self.dataset.name))
        path = os.path.join(IMAGES_DIRECTORY, '{}-dataset_balance.png'.format(self.dataset.name))
        plt.savefig(path)
        # plt.close(5)

    def save_classification_report(self, gscv):
        estimator_name = self.get_estimator_name(gscv.estimator)
        logging.info('Save Classification Report for Dataset {} and Estimator {}'.format(self.dataset.name, estimator_name))

        path = os.path.join(LOGS_DIRECTORY, '{}-{}-gscv-classification_report.txt'.format(self.dataset.name, estimator_name))
        Y_pred = gscv.predict(self.dataset.X_test)
        report = classification_report(self.dataset.Y_test, Y_pred)
        with open(path, 'a') as stream:
            stream.write(report + os.linesep)

    def do_neural_network_random_hill_climb(self):
        logging.info('Start Neural Network Weights Optimization with Random Hill Climb Algorithm')
        estimator = mlrose.NeuralNetwork(hidden_nodes=[20],
                                # activation='relu',
                                algorithm='random_hill_climb',
                                # max_iters=100,
                                # bias=True,
                                # is_classifier=True,
                                # learning_rate=0.1,
                                # early_stopping=True,
                                # clip_max=5,
                                restarts=5,
                                # schedule=mlrose.GeomDecay,
                                # pop_size=200,
                                # mutation_prob=0.1,
                                # max_attempts=10,
                                random_state=self.random_state,
                                curve=True,
                                )

        param_grid = {
            'max_iters': [16000],
        }

        gscv = self.do_neural_network(estimator=estimator, param_grid=param_grid)
        self.save_neural_network_variables(gscv)
        logging.info('Finish Neural Network Weights Optimization with Random Hill Climb Algorithm')

    def do_neural_network_simulated_annealing(self):
        logging.info('Start Neural Network Weights Optimization with Simulated Annealing Algorithm')
        estimator = mlrose.NeuralNetwork(hidden_nodes=[20],
                                # activation='relu',
                                algorithm='simulated_annealing',
                                # max_iters=100,
                                # bias=True,
                                # is_classifier=True,
                                # learning_rate=0.1,
                                # early_stopping=True,
                                # clip_max=5,
                                # restarts=10,
                                # schedule=mlrose.GeomDecay,
                                # pop_size=200,
                                # mutation_prob=0.1,
                                max_attempts=5,
                                random_state=self.random_state,
                                curve=True,
                                )

        param_grid = {
            'max_iters': [16000],
        }

        gscv = self.do_neural_network(estimator=estimator, param_grid=param_grid)
        self.save_neural_network_variables(gscv)
        logging.info('Finish Neural Network Weights Optimization with Simulated Annealing Algorithm')

    def do_neural_network_genetic_algorithm(self):
        logging.info('Start Neural Network Weights Optimization with Genetic Algorithm')
        estimator = mlrose.NeuralNetwork(hidden_nodes=[20],
                                # activation='relu',
                                algorithm='genetic_alg',
                                max_iters=500,
                                # bias=True,
                                # is_classifier=True,
                                # learning_rate=0.1,
                                # early_stopping=True,
                                # clip_max=5,
                                # restarts=10,
                                # schedule=mlrose.GeomDecay,
                                # pop_size=200,
                                # mutation_prob=0.1,
                                max_attempts=5,
                                random_state=self.random_state,
                                curve=True,
                                )

        param_grid = {
            # 'max_attempts': [5, 10, 20],
            # 'max_iters': [500],
        }

        gscv = self.do_neural_network(estimator=estimator, param_grid=param_grid)
        self.save_neural_network_variables(gscv)
        logging.info('Finish Neural Network Weights Optimization with Genetic Algorithm')

    def do_neural_network_gradient_descent(self):
        logging.info('Start Neural Network Weights Optimization with Gradient Descent Algorithm')
        estimator = mlrose.NeuralNetwork(hidden_nodes=[100],
                                activation='relu',
                                algorithm='gradient_descent',
                                # max_iters=1000,
                                # bias=True,
                                # is_classifier=True,
                                # learning_rate=0.0001,
                                # early_stopping=True,
                                # clip_max=5,
                                # restarts=10,
                                # schedule=mlrose.GeomDecay,
                                # pop_size=200,
                                # mutation_prob=0.1,
                                max_attempts=100,
                                random_state=self.random_state,
                                curve=True,
                                )

        param_grid = {
            'hidden_nodes': ([2], [4], [8], [16], [32], [64], [128], [256], [512], [1024]),
            # 'hidden_nodes': ([2], [10], [50], [100], [200]),
            # 'learning_rate': [0.001, 0.0001],
            'max_iters': [10, 100, 1000, 10000],
            # 'activation': ['relu', 'sigmoid'],
            # 'max_attempts' [10, 100, 1000],
            # 'early_stopping': [True, False],
        }

        gscv = self.do_neural_network(estimator=estimator, param_grid=param_grid)
        self.save_neural_network_variables(gscv)
        logging.info('Finish Neural Network Weights Optimization with Gradient Descent Algorithm')

    def save_neural_network_variables(self, gscv):
        estimator = gscv.best_estimator_
        estimator_name = self.get_estimator_name(gscv.estimator)
        algorithm_name = estimator.algorithm
        logging.info('Save Neural Network Variables for Dataset {}, Estimator {}, Algorithm: {}'.format(self.dataset.name, estimator_name, algorithm_name))

        directory = os.path.join(LOGS_DIRECTORY, 'neural', algorithm_name)
        os.makedirs(directory)

        path = os.path.join(directory, '{}-{}-gscv-neural_network-{}-fitted_weights.csv'.format(self.dataset.name, estimator_name, algorithm_name))
        pd.DataFrame(estimator.fitted_weights, columns=['Fitted Weights']).to_csv(path)

        path = os.path.join(directory, '{}-{}-gscv-neural_network-{}-loss.csv'.format(self.dataset.name, estimator_name, algorithm_name))
        pd.DataFrame([{'loss': estimator.loss}]).to_csv(path)

        path = os.path.join(directory, '{}-{}-gscv-neural_network-{}-predicted_probs.csv'.format(self.dataset.name, estimator_name, algorithm_name))
        pd.DataFrame(estimator.predicted_probs, columns=['Predicted Probs']).to_csv(path)

        path = os.path.join(directory, '{}-{}-gscv-neural_network-{}-fitness_curve.csv'.format(self.dataset.name, estimator_name, algorithm_name))
        pd.DataFrame(estimator.fitness_curve, columns=['Fitness Curve']).to_csv(path)


class Dataset(object):
    def __init__(self, path, verbose=False, threads=-1):
        basename = os.path.basename(path)
        filename = os.path.splitext(basename)[0]
        self.name = filename
        self.path = path
        self.threads = threads
        self.verbose = verbose

        self.X = None
        self.Y = None
        self.X_test = None
        self.Y_test = None
        self.X_train = None
        self.Y_train = None

    def load(self, header=None):
        """Loads data from self.path. Modifies self.header, self.columns, self.X and self.Y"""
        self.header = header
        self.data = pd.read_csv(self.path, header=self.header)
        self.columns = self.data.columns
        self.X, self.Y = self.data[self.columns[:-1]], self.data[self.columns[-1]]      # X = features, Y = label
        self.binary = self.is_binary()
        self.balanced = self.is_balanced()
        self.sparse = self.is_sparse()

        description = {'path': self.path, 'shape': self.data.shape, 'sparse': self.sparse, 'binary': self.binary, 'balanced': self.balanced, 'missing': self.has_missing_values()}
        path = os.path.join(LOGS_DIRECTORY, 'DatasetDescriptions.csv'.format())
        pd.DataFrame([description]).to_csv(path, index=False, mode='a')
        logging.info('Loaded Dataset {}'.format(description))

        return self.X, self.Y

    def is_sparse(self):
        return isspmatrix(self.X)

    def is_binary(self):
        """Check if the dataset has only two classes, or more than 2 classes"""
        counts = np.unique(self.Y, return_counts=True)
        return len(counts[0]) <= 2      # Edge case of only one class? Declare it binary

    def is_balanced(self):
        """Reference: https://stats.stackexchange.com/a/239982/265283
        Use Shannon entropy as a measure of dataset's balance.  H == 0 (Imbalanced). H > 0 (Balanced)
        """
        counts = np.unique(self.Y, return_counts=True)
        H = entropy(counts[0])      # Shannon entropy
        return H > 0

    def has_missing_values(self):
        """Check if there are nan values in data"""
        return self.data.isnull().sum().sum() > 0

    def split(self, test_size, random_state, shuffle):
        """Split features and labels into training set and test set. Modifies self.random_state and self.test_size"""
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=self.Y)
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
        #     self.X, self.Y, test_size=test_size, random_state=random_state)
        logging.debug('Split Dataset {}'.format({'test_size': self.test_size, 'random_state': self.random_state}))
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def scale(self):
        """Standardize the training set of features or test set of features independently"""
        scaler = preprocessing.MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        return self.X_train, self.X_test


class ToyProblems(object):
    def __init__(self, random_state):
        self.random_state = random_state

    def do_run(self, problem, algorithm, params=None):
        params = params or {}
        fitness_name = problem.fitness_fn.__class__.__name__
        directory = os.path.join(LOGS_DIRECTORY, fitness_name)

        algorithm_name = algorithm.__name__
        details = os.path.join(directory, algorithm_name, 'fitness_curves')

        os.makedirs(details, exist_ok=True)

        logging.info('Start {} :: {} Run. Parameters: {}'.format(fitness_name, algorithm_name, params))
        start = timer()
        best_state, best_fitness, fitness_curve = algorithm(
            problem=problem,
            curve=True,
            random_state=self.random_state,
            **params,
        )
        duration = timer() - start
        path = os.path.join(details, 'FitnessCurves_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')))
        pd.DataFrame(fitness_curve, columns=['Fitness']).to_csv(path)
        logging.info('Finish {} :: {} Run. Results: {}'.format(fitness_name, algorithm_name, directory))
        return best_state, best_fitness, len(fitness_curve), duration

    def do_runs(self, num_runs, problem, algorithm, params):
        """Do several runs and compute the average"""
        np.random.seed(self.random_state)
        fitness_name = problem.fitness_fn.__class__.__name__

        _best_states, _best_fitnesses, _num_iterations, _durations = [], [], [], []
        for run in range(num_runs):
            logging.info('Fitness: {}. Length: {}. Run: {}/{}'.format(fitness_name, problem.length, run+1, num_runs))
            best_state, best_fitness, num_iteration, duration = self.do_run(problem, algorithm, params)
            # _best_states.append(best_state)
            _best_fitnesses.append(best_fitness)
            _num_iterations.append(num_iteration)
            _durations.append(duration)
        # best_state_mean, best_state_std = np.mean(_best_states), np.std(_best_states)
        best_fitness_mean, best_fitness_std = np.mean(_best_fitnesses), np.std(_best_fitnesses)
        num_iteration_mean, num_iteration_std = np.mean(_num_iterations), np.std(_num_iterations)
        duration_mean, duration_std = np.mean(_durations), np.std(_durations)

        results = {
            # 'best_state_mean': best_state_mean,
            # 'best_state_std': best_state_std,
            'best_fitness_mean': best_fitness_mean,
            'best_fitness_std': best_fitness_std,
            'num_iteration_mean': num_iteration_mean,
            'num_iteration_std': num_iteration_std,
            'duration_mean': duration_mean,
            'duration_std': duration_std,
        }

        return results

    def save_runs_results(self, fitness_fn, results):
        # Save Summary of (Algorithm, Length) pair results
        fitness_name = fitness_fn.__class__.__name__
        logs_directory = os.path.join(LOGS_DIRECTORY, fitness_name)
        os.makedirs(logs_directory, exist_ok=True)

        images_directory = os.path.join(IMAGES_DIRECTORY, fitness_name)
        os.makedirs(images_directory, exist_ok=True)

        path = os.path.join(logs_directory, 'Summary.csv'.format())
        rows = []
        for algorithm, values in results.items():
            for value in values:
                row = dict(value)
                row.update(algorithm=algorithm)
                rows.append(row)
        pd.DataFrame(rows).to_csv(path)

        algorithms = [key for key in results.keys()]
        lengths = [result['length'] for result in results[algorithms[0]]]

        # Plot Best Fitness vs Length
        plt.clf()
        plt.title('{}: Best Fitness vs Length'.format(fitness_name))
        plt.xlabel('Input Length')
        plt.ylabel('Best Fitness')
        plt.grid()
        for algorithm, values in results.items():
            best_fitness_means = np.array([value['best_fitness_mean'] for value in values])
            best_fitness_stds = np.array([value['best_fitness_std'] for value in values])
            plt.fill_between(lengths, best_fitness_means - best_fitness_stds, best_fitness_means + best_fitness_stds, alpha=0.1)
            plt.plot(lengths, best_fitness_means, label=algorithm)
        plt.legend(loc='best')
        path = os.path.join(images_directory, 'BestFitness_vs_Length.png')
        plt.savefig(path)

        # Plot Iterations vs Length
        plt.clf()
        plt.title('{}: Num Iterations vs Length'.format(fitness_name))
        plt.xlabel('Input Length')
        plt.ylabel('Number of Iterations')
        plt.grid()
        for algorithm, values in results.items():
            num_iteration_means = np.array([value['num_iteration_mean'] for value in values])
            num_iteration_stds = np.array([value['num_iteration_std'] for value in values])
            plt.fill_between(lengths, num_iteration_means - num_iteration_stds, num_iteration_means + num_iteration_stds, alpha=0.1)
            plt.plot(lengths, num_iteration_means, label=algorithm)
        plt.legend(loc='best')
        path = os.path.join(images_directory, 'Iterations_vs_Length.png')
        plt.savefig(path)

        # Plot Durations vs Length
        plt.clf()
        plt.title('{}: Durations vs Length'.format(fitness_name))
        plt.xlabel('Input Length')
        plt.ylabel('Duration (s)')
        plt.grid()
        for algorithm, values in results.items():
            duration_means = np.array([value['duration_mean'] for value in values])
            duration_stds = np.array([value['duration_std'] for value in values])
            plt.fill_between(lengths, duration_means - duration_stds, duration_means + duration_stds, alpha=0.1)
            plt.plot(lengths, duration_means, label=algorithm)
        plt.legend(loc='best')
        path = os.path.join(images_directory, 'Durations_vs_Length.png')
        plt.savefig(path)

    def do_one_max(self):
        fitness_fn = mlrose.OneMax()
        algorithms = {
            mlrose.random_hill_climb: {'max_attempts': 20, 'restarts': 20},
            mlrose.simulated_annealing: {'max_attempts': 20},
            mlrose.genetic_alg: {'max_attempts': 20},
            mlrose.mimic: {'fast_mimic': True, 'max_attempts': 20},
        }
        results = defaultdict(list)
        for length in LENGTHS:
            problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
            for algorithm, params in algorithms.items():
                _results = self.do_runs(NUM_RUNS, problem, algorithm, params)
                _results.update(length=length)
                results[algorithm.__name__].append(_results)

        self.save_runs_results(fitness_fn, results)

    def do_six_peaks(self):
        fitness_fn = mlrose.SixPeaks(t_pct=0.1)
        algorithms = {
            mlrose.random_hill_climb: {'max_attempts': 20, 'restarts': 20},
            mlrose.simulated_annealing: {'max_attempts': 20},
            mlrose.genetic_alg: {'max_attempts': 20},
            mlrose.mimic: {'fast_mimic': True, 'max_attempts': 20},
        }
        results = defaultdict(list)
        for length in LENGTHS:
            problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
            for algorithm, params in algorithms.items():
                _results = self.do_runs(NUM_RUNS, problem, algorithm, params)
                _results.update(length=length)
                results[algorithm.__name__].append(_results)

        self.save_runs_results(fitness_fn, results)

    def do_four_peaks(self):
        fitness_fn = mlrose.FourPeaks(t_pct=0.1)
        algorithms = {
            mlrose.random_hill_climb: {'max_attempts': 20, 'restarts': 20},
            mlrose.simulated_annealing: {'max_attempts': 20},
            mlrose.genetic_alg: {'max_attempts': 20},
            mlrose.mimic: {'fast_mimic': False, 'max_attempts': 20},
        }
        results = defaultdict(list)
        for length in LENGTHS:
            problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
            for algorithm, params in algorithms.items():
                _results = self.do_runs(NUM_RUNS, problem, algorithm, params)
                _results.update(length=length)
                results[algorithm.__name__].append(_results)

        self.save_runs_results(fitness_fn, results)

    def do_flip_flop(self):
        fitness_fn = mlrose.FlipFlop()
        algorithms = {
            mlrose.random_hill_climb: {'max_attempts': 20, 'restarts': 20},
            mlrose.simulated_annealing: {'max_attempts': 20},
            mlrose.genetic_alg: {'max_attempts': 20},
            mlrose.mimic: {'fast_mimic': True, 'max_attempts': 20},
        }
        results = defaultdict(list)
        for length in LENGTHS:
            problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
            for algorithm, params in algorithms.items():
                _results = self.do_runs(NUM_RUNS, problem, algorithm, params)
                _results.update(length=length)
                results[algorithm.__name__].append(_results)

        self.save_runs_results(fitness_fn, results)

    def plot_state_space(self, length=5, base=2):
        """Plot state space to visualize distribution of optima"""
        scorer = ScoreStates(length=length, base=base)
        states = scorer.get_states()

        fitness_fns = [mlrose.OneMax(), mlrose.FourPeaks(), mlrose.SixPeaks(), mlrose.FlipFlop()]
        for fitness_fn in fitness_fns:
            fitness_name = fitness_fn.__class__.__name__
            _max, _min, scores = scorer.score_states(fitness_fn, states)
            df = pd.DataFrame(scores.values(), columns=['Fitness'])

            directory = os.path.join(IMAGES_DIRECTORY, fitness_name)
            os.makedirs(directory, exist_ok=True)
            logging.info('Plotting State Space in {}'.format(directory))
            df.hist()
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('{} - State Space - Input Length {}'.format(fitness_name, length))
            plt.savefig(os.path.join(directory, 'StateSpaceHistogram.png'))

            df.plot.bar(xlabel='State', ylabel='Score')
            plt.grid()
            plt.title('{} - State Space - Input Length {}'.format(fitness_name, length))
            plt.savefig(os.path.join(directory, 'StateSpaceBarPlot.png'))


class ScoreStates(object):
    """Utility class to get score all possible states of a fitness function"""

    def __init__(self, length, base=2):
        """
        length: Number of values (bits) in each state
        base: Numeric base for each bit in each state. 2 == binary
        """
        self.length = length
        self.base = base

    def get_states(self):
        """Get all possible states, padding each state to they all have equal length"""
        states = []
        for value in range(self.base ** self.length):
            state = [int(v) for v in bin(value)[2:]]
            state = np.pad(state, (self.length - len(state), 0), 'constant', constant_values=(0))
            state = tuple(state)
            states.append(state)
        return states

    def score_states(self, fitness_fn, states):
        """Evaluate each state using a fitess function
        Return the maximum and minimum scores, and all scores
        """
        maximum = float('-inf')
        minimum = float('inf')
        scores = {}
        for state in states:
            score = fitness_fn.evaluate(state)
            if score > maximum:
                maximum = score
            if score < minimum:
                minimum = score
            scores[state] = score
        return maximum, minimum, scores

    def get_best_and_worst_states(self, maximum, minimum, scores):
        """Get the best and worse states based on the maximum and minimum scores"""
        best_states = []
        worst_states = []
        for state, score in scores.items():
            if score == maximum:
                best_states.append(state)
            if score == minimum:
                worst_states.append(state)
        return best, worst

    def score_fitness(self, fitness_fn):
        states = self.get_states()
        maximum, minimum, scores = self.score_states(fitness_fn, states)
        best, worst = self.get_best_and_worst_states(maximum, minimum, scores)
        return {maximum: best}, {minimum: worst}
        # fitness_name = fitness_fn.__class__.__name__
        # logging.info('{}:: Best Score: {}. Best States: {}'.format(fitness_name, maximum, best_states))
        # logging.info('{}:: Worst Score: {}. Worst States: {}'.format(fitness_name, minimum, worst_states))


def main004():
    """Machine Learning Weights Optimization Tutorial: https://mlrose.readthedocs.io/en/stable/source/tutorial3.html"""
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    
    data = load_iris()
    logging.info('Feature Values: {}'.format(data.data[0]))
    logging.info('Feature Names: {}'.format(data.feature_names))
    logging.info('First Observation Target Value: {}'.format(data.target[0]))
    logging.info('First Observation Target Name: {}'.format(data.target_names[data.target[0]]))
    logging.info('Minimum Feature Values: {}'.format(np.min(data.data, axis=0)))
    logging.info('Maximum Feature Values: {}'.format(np.max(data.data, axis=0)))
    logging.info('Unique Target Values: {}'.format(np.unique(data.target)))

    test_size = 0.2
    random_state = 33
    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    encoder = OneHotEncoder()
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1)).todense()
    Y_test = encoder.transform(Y_test.reshape(-1, 1)).todense()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm='random_hill_climb', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5,
                                    max_attempts=100, random_state=random_state)
    nn_model1.fit(X_train, Y_train)

    Y_train_pred = nn_model1.predict(X_train)
    Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)
    logging.info('Y_train Accuracy: {}'.format(Y_train_accuracy))

    Y_test_pred = nn_model1.predict(X_test)
    Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)
    logging.info('Y_test Accuracy: {}'.format(Y_test_accuracy))


    nn_model2 = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm='gradient_descent', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5,
                                    max_attempts=100, random_state=random_state)
    nn_model2.fit(X_train, Y_train)

    Y_train_pred = nn_model2.predict(X_train)
    Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)
    logging.info('Y_train Accuracy: {}'.format(Y_train_accuracy))

    Y_test_pred = nn_model2.predict(X_test)
    Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)
    logging.info('Y_test Accuracy: {}'.format(Y_test_accuracy))


def main015():
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    Args = namedtuple('Args', ['threads', 'random_state', 'verbose', 'dataset'])
    args = Args(threads=10, random_state=1, verbose=True, dataset=['/home/cyc/dev/gatech/cs7641/project2/workspace/data/CreditDefaults.csv'])
    # args = Args(threads=10, random_state=1, verbose=True, dataset=['/home/cyc/dev/gatech/cs7641/project2/workspace/data/PenDigits.csv'])
    logging.info('Experiment Config: {}'.format({'random_state': args.random_state, 'threads': args.threads, 'output': OUTPUT_DIRECTORY}))
    random_state = args.random_state
    for path in args.dataset:
        expr = Expr(path=path, threads=args.threads, random_state=args.random_state, verbose=args.verbose)
        expr.do_prepare_dataset()
        expr.plot_datset_class_distribution()

        params = {
            'hidden_nodes': [10], 'activation': 'relu', 'algorithm': 'gradient_descent', 'max_iters': 3000,
            'bias': True, 'is_classifier': True, 'learning_rate': 0.001, 'early_stopping': True, 'clip_max': 5,
            'restarts': 5, 'schedule': mlrose.GeomDecay(), 'pop_size': 200, 'mutation_prob': 0.1,
            'max_attempts': 10, 'random_state': random_state, 'curve': True
        }

        logging.info('NN Params: {}'.format(params))
        nn_model = mlrose.NeuralNetwork(**params)

        # nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[100], activation='relu', algorithm='simulated_annealing', max_iters=1000,
        #                             bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5,
        #                             max_attempts=100, random_state=random_state)

        X_train = expr.dataset.X_train
        Y_train = expr.dataset.Y_train
        X_test = expr.dataset.X_test
        Y_test = expr.dataset.Y_test

        nn_model.fit(X_train, Y_train)
        Y_train_pred = nn_model.predict(X_train)
        Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)
        logging.info('Y_train Accuracy: {}'.format(Y_train_accuracy))

        Y_test_pred = nn_model.predict(X_test)
        Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)
        logging.info('Y_test Accuracy: {}'.format(Y_test_accuracy))

        report = classification_report(Y_test, Y_test_pred)
        # logging.info('Classification Report:\n{}'.format(report))

        matrix = confusion_matrix(Y_test, Y_test_pred, normalize='true')
        logging.info('Confusion Matrix:\n{}'.format(matrix))

        logging.info('Loss: {}'.format(nn_model.loss))
        fitted_weights = nn_model.fitted_weights
        _mean = np.mean(fitted_weights)
        _max = np.max(fitted_weights)
        _min = np.min(fitted_weights)
        _std = np.std(fitted_weights)
        logging.info('Fitted Weights: {}'.format({'min': _min, 'max': _max, 'mean': _mean, 'std': _std}))

        directory = os.path.join(OUTPUT_DIRECTORY, 'neural')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'fitness_curve_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')))
        logging.info('NN Fitness Curve: {}'.format(path))
        pd.DataFrame(nn_model.fitness_curve).to_csv(path)


def main016():
    parser = argparse.ArgumentParser(description='CS 7641: Randomized Optimization')
    parser.add_argument('-d', '--dataset', action='append', help='Path to dataset to experiment on')
    parser.add_argument('-r', '--random_state', type=int, default=1, help='Random state for reproducible results')
    parser.add_argument('-t', '--threads', type=int, default=-1, help='Number of jobs for parallel processing')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbosity of experiment')
    parser.add_argument('--flip_flop', action='store_true', default=False, help='Run FlipFlop Toy Problem')
    parser.add_argument('--one_max', action='store_true', default=False, help='Run OneMax Toy Problem')
    parser.add_argument('--four_peaks', action='store_true', default=False, help='Run FourPeaks Toy Problem')
    parser.add_argument('--six_peaks', action='store_true', default=False, help='Run SixPeaks Toy Problem')
    parser.add_argument('--neural_network', choices=['random_hill_climb', 'simulated_annealing', 'genetic_algorithm', 'gradient_descent'], help='Find Neural Network Weights using Randomized Optimization')

    args = parser.parse_args([
        '--random_state', '1', 
        '--threads', '-1', 
        '--dataset', '../data/CreditDefaults.csv',
        '--dataset', '../data/PenDigits.csv',
    ])
    args = parser.parse_args()
    args.dataset = args.dataset or []   # Use empty list if no dataset is supplied

    config = args.__dict__
    config['output'] = OUTPUT_DIRECTORY
    logging.info('Experiment Config: {}'.format(config))

    # Validate paths
    for path in args.dataset:
        if not os.path.isfile(path):
            logging.error('Dataset {} is not a valid file'.format(path))
            return

    for path in args.dataset:
        expr = Expr(path=path, threads=args.threads, random_state=args.random_state, verbose=args.verbose)
        expr.do_prepare_dataset()
        expr.plot_datset_class_distribution()

        if args.neural_network == 'random_hill_climb':
            expr.do_neural_network_random_hill_climb()
        if args.neural_network == 'simulated_annealing':
            expr.do_neural_network_simulated_annealing()
        if args.neural_network == 'genetic_algorithm':
            expr.do_neural_network_genetic_algorithm()
        if args.neural_network == 'gradient_descent':
            expr.do_neural_network_gradient_descent()

    toy = ToyProblems(random_state=args.random_state)
    toy.plot_state_space()
    if args.one_max:
        toy.do_one_max()
    if args.four_peaks:
        toy.do_four_peaks()
    if args.six_peaks:
        toy.do_six_peaks()
    if args.flip_flop:
        toy.do_flip_flop()


def main017():
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    Args = namedtuple('Args', ['threads', 'random_state', 'verbose', 'dataset'])
    args = Args(threads=10, random_state=1, verbose=True, dataset=['/home/cyc/dev/gatech/cs7641/project2/workspace/data/CreditDefaults.csv'])
    logging.info('Experiment Config: {}'.format({'random_state': args.random_state, 'threads': args.threads, 'output': OUTPUT_DIRECTORY}))
    random_state = args.random_state
    for path in args.dataset:
        expr = Expr(path=path, threads=args.threads, random_state=args.random_state, verbose=args.verbose)
        expr.do_prepare_dataset()
        expr.plot_datset_class_distribution()

        param_grid = {
            'hidden_nodes': [[4], [8], [16]],
            'max_iters': [100, 200, 300],
            'clip_max': [5, 10],
            'learning_rate': [0.1, 0.9],
            'restarts': [0, 2],
            'max_attempts': [10, 20],
        }

        directory = os.path.join(LOGS_DIRECTORY, 'nn_random_hill_climb_search')
        fitness_name = expr.dataset.name
        details = os.path.join(directory, 'fitness_curves', fitness_name)
        os.makedirs(details)

        results = []

        param_grid = ParameterGrid(param_grid)
        logging.info('Start RandomHillClimb Parameter Search. {} Parameter Combinations'.format(len(param_grid)))
        for index, params in enumerate(param_grid):
            params['activation'] = 'relu'
            params['algorithm'] = 'random_hill_climb'
            params['bias'] = True
            params['is_classifier'] = True
            params['early_stopping'] = True
            params['curve'] = True

            logging.info('Running parameter combination: {}/{}: {}'.format((index+1), len(param_grid), params))
            start = timer()
            nn_model = mlrose.NeuralNetwork(**params)
            X_train = expr.dataset.X_train
            Y_train = expr.dataset.Y_train
            X_test = expr.dataset.X_test
            Y_test = expr.dataset.Y_test

            nn_model.fit(X_train, Y_train)
            Y_train_pred = nn_model.predict(X_train)
            Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)
            # logging.info('Y_train Accuracy: {}'.format(Y_train_accuracy))

            Y_test_pred = nn_model.predict(X_test)
            Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)
            # logging.info('Y_test Accuracy: {}'.format(Y_test_accuracy))

            # report = classification_report(Y_test, Y_test_pred)
            # logging.info('Classification Report:\n{}'.format(report))

            # matrix = confusion_matrix(Y_test, Y_test_pred, normalize='true')
            # logging.info('Confusion Matrix:\n{}'.format(matrix))

            # logging.info('Loss: {}'.format(nn_model.loss))
            fitted_weights = nn_model.fitted_weights
            _mean = np.mean(fitted_weights)
            _max = np.max(fitted_weights)
            _min = np.min(fitted_weights)
            _std = np.std(fitted_weights)
            # logging.info('Fitted Weights: {}'.format({'min': _min, 'max': _max, 'mean': _mean, 'std': _std}))
            duration = timer() - start

            # directory = os.path.join(OUTPUT_DIRECTORY, 'neural')
            # os.makedirs(directory, exist_ok=True)
            path = os.path.join(details, 'fitness_curve_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')))
            logging.info('NN Fitness Curve: {}'.format(path))
            pd.DataFrame(nn_model.fitness_curve).to_csv(path)
            params.update(duration=duration, Y_train_accuracy=Y_train_accuracy, Y_test_accuracy=Y_test_accuracy,
                loss=nn_model.loss, fitted_min=_min, fitted_max=_max, fitted_mean=_mean, fitted_std=_std)
            results.append(params)
            # path = os.path.join(details, '{}.csv'.format(index+1).zfill(3))
            # pd.DataFrame(fitness_curve, columns=['Fitness']).to_csv(path)
            # logging.info('Random Hill Climb: Params: {}. Best Fitness: {}. Duration: {} seconds'.format(params, best_fitness, duration))

        path = os.path.join(directory, 'RandomHillClimb_{}_SearchResults.csv'.format(fitness_name))
        pd.DataFrame(results).to_csv(path)
        logging.info('Finish RandomHillClimb Parameter Search. Results: {}'.format(directory))


def main018():
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    Args = namedtuple('Args', ['threads', 'random_state', 'verbose', 'dataset'])
    args = Args(threads=10, random_state=1, verbose=True, dataset=['/home/cyc/dev/gatech/cs7641/project2/workspace/data/CreditDefaults.csv'])
    logging.info('Experiment Config: {}'.format({'random_state': args.random_state, 'threads': args.threads, 'output': OUTPUT_DIRECTORY}))
    random_state = args.random_state
    for path in args.dataset:
        expr = Expr(path=path, threads=args.threads, random_state=args.random_state, verbose=args.verbose)
        expr.do_prepare_dataset()
        expr.plot_datset_class_distribution()

        param_grid = {
            'hidden_nodes': [[4], [8], [16]],
            'max_iters': [100, 200, 300],
            'learning_rate': [0.1, 0.9, 2.0],
            'restarts': [0, 2],
            'max_attempts': [10, 20],
        }

        directory = os.path.join(LOGS_DIRECTORY, 'nn_simulated_annealing_search')
        fitness_name = expr.dataset.name
        details = os.path.join(directory, 'fitness_curves', fitness_name)
        os.makedirs(details)

        results = []

        param_grid = ParameterGrid(param_grid)
        logging.info('Start Parameter Search. {} Parameter Combinations'.format(len(param_grid)))
        for index, params in enumerate(param_grid):
            params['activation'] = 'relu'
            params['algorithm'] = 'simulated_annealing'
            params['bias'] = True
            params['is_classifier'] = True
            params['early_stopping'] = True
            params['curve'] = True

            logging.info('Running parameter combination: {}/{}: {}'.format((index+1), len(param_grid), params))
            start = timer()
            nn_model = mlrose.NeuralNetwork(**params)
            X_train = expr.dataset.X_train
            Y_train = expr.dataset.Y_train
            X_test = expr.dataset.X_test
            Y_test = expr.dataset.Y_test

            nn_model.fit(X_train, Y_train)
            Y_train_pred = nn_model.predict(X_train)
            Y_train_accuracy = accuracy_score(Y_train, Y_train_pred)
            # logging.info('Y_train Accuracy: {}'.format(Y_train_accuracy))

            Y_test_pred = nn_model.predict(X_test)
            Y_test_accuracy = accuracy_score(Y_test, Y_test_pred)
            # logging.info('Y_test Accuracy: {}'.format(Y_test_accuracy))

            # report = classification_report(Y_test, Y_test_pred)
            # logging.info('Classification Report:\n{}'.format(report))

            # matrix = confusion_matrix(Y_test, Y_test_pred, normalize='true')
            # logging.info('Confusion Matrix:\n{}'.format(matrix))

            # logging.info('Loss: {}'.format(nn_model.loss))
            fitted_weights = nn_model.fitted_weights
            _mean = np.mean(fitted_weights)
            _max = np.max(fitted_weights)
            _min = np.min(fitted_weights)
            _std = np.std(fitted_weights)
            # logging.info('Fitted Weights: {}'.format({'min': _min, 'max': _max, 'mean': _mean, 'std': _std}))
            duration = timer() - start

            # directory = os.path.join(OUTPUT_DIRECTORY, 'neural')
            # os.makedirs(directory, exist_ok=True)
            path = os.path.join(details, 'fitness_curve_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')))
            logging.info('NN Fitness Curve: {}'.format(path))
            pd.DataFrame(nn_model.fitness_curve).to_csv(path)
            params.update(duration=duration, Y_train_accuracy=Y_train_accuracy, Y_test_accuracy=Y_test_accuracy,
                loss=nn_model.loss, fitted_min=_min, fitted_max=_max, fitted_mean=_mean, fitted_std=_std)
            results.append(params)
            # path = os.path.join(details, '{}.csv'.format(index+1).zfill(3))
            # pd.DataFrame(fitness_curve, columns=['Fitness']).to_csv(path)
            # logging.info('Random Hill Climb: Params: {}. Best Fitness: {}. Duration: {} seconds'.format(params, best_fitness, duration))

        path = os.path.join(directory, 'SearchResults.csv'.format(fitness_name))
        pd.DataFrame(results).to_csv(path)
        logging.info('Finish Parameter Search. Results: {}'.format(directory))



if __name__ == '__main__':
    # debug001()
    # debug002()
    # debug003()

    # main()
    # main002()
    # main003()
    # main004()
    # main005()
    # main006()
    # main007()
    # main008()
    # main009()
    # main010()

    # main011()

    # main012()
    # main013()
    # main014()

    # main015()
    main016()
    # main017()
    # main018()

