#!/usr/bin/env python
# coding: utf-8

# d.tarasov
# 8: Regularization: l2, Optim: ADAGRAD

import numpy as np


class LogisticRegression:
    def __init__(self, lambda_coef=1.0, regulatization='L2', alpha=1e-5, loss_eps=1e-5, eps=1e-6, batch_size=50,
                 max_iter=10):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regulatization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        :param batch_size: num sample per one model parameters update
        :param max_iter: maximum number of parameters updates
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lambda_coef = lambda_coef
        self.regulatization = regulatization
        self.loss_eps = loss_eps
        self.eps = eps

        return

    def get_grad(self, X_train, y_pred, y):
        grad = np.zeros_like(self.weights)

        n = X_train.shape[0]
        m = X_train.shape[1]

        for j in range(m):
            y_delta = y_pred - y
            grad_arr = [X_train[i, j] * np.sign(y_delta[i]) for i in range(n)]
            jgrad = np.sum(grad_arr)
            jgrad /= n

            if self.regulatization == 'L1':
                jgrad += self.alpha * abs(self.weights[j])
            elif self.regulatization == 'L2':
                jgrad += self.alpha * (self.weights[j]) ** 2
            else:
                raise Exception("Unknown regulatization")

            grad[j] = jgrad
        return grad

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """

        # name vars like sklearn
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        self.coef_ = list()
        self.intercept_ = np.random.uniform(size=(n_features, 1))  # aka bias
        num_classes = len(np.unique(y_train))

        self.weights = np.zeros((n_features + 1, num_classes - 1), dtype=np.float64)

        prev_loss = 0
        for i in range(self.max_iter):
            loss = 0
            adagrad_S = 0
            for j in range(0, 1, self.batch_size):
                X_batch = X_train[j:max(n_samples, j+self.batch_size)]
                y_batch = y_train[j:max(n_samples, j+self.batch_size)]

                y_pred = self.predict_proba(X_batch)
                loss += self.get_logloss(y_pred, y_batch)
                grad = self.get_grad(X_batch, y_pred, y_batch)

                # названия переменной отсюда взял
                # https://d2l.ai/chapter_optimization/adagrad.html
                adagrad_S += np.power(grad, 2)
                # adagrad optimisation
                self.weights -= self.lambda_coef / np.sqrt(adagrad_S + self.eps) * grad

            print('iter ', i)

            if abs(prev_loss - loss) < self.loss_eps:
                print("Loss eps reached")
                break

            prev_loss = loss

    def get_logloss(self, y_pred, y_true):
        loss_summ = 0
        loss_summ += np.sum(-y_true*np.log(y_pred))        # y_true == 1
        loss_summ += np.sum(-(1-y_true)*np.log(1-y_pred))  # y_true == 0
        return loss_summ / y_true.shape[0]

    def predict(self, X_test, treshhold=0.5):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        y_pred = self.predict_proba(X_test)
        return np.where(y_pred < treshhold, 0, 1)

    def predict_proba(self, X_test):
        """
        Predict probability using model.
        :param X_test: test data for predict in
        :return: y_test: predicted probabilities
        """

        def sigmoid(x):
            return 1. / (1. + np.exp(x))
        bias_col = np.ones((X_test.shape[0], 1))
        X_test_with_bias = np.hstack((bias_col, X_test))
        print(X_test_with_bias.shape, self.weights.shape)
        return sigmoid(X_test_with_bias @ self.weights)

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.weights
