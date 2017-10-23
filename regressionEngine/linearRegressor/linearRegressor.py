import numpy as np
from regressionEngine import abstractRegressor as ar


class LinearRegressor(ar.AbstractRegressor):
    def calculate_hypothesis(self, inputValues):
        return inputValues.dot(self.thetas.transpose())

    def compute_cost(self):
        # calculate the cost function
        hypothesis = self.calculate_hypothesis(self.normalizedFeatures)
        cost = np.square(hypothesis - self.inputYs)
        cost = cost.sum() / (2 * (np.shape(self.normalizedFeatures)[0]))
        return cost

    def predict(self, fileName):
        return self.make_prediction(fileName)

    def predict_for_plot(self, fileName):
        return self.make_prediction(fileName)

    def train(self, alpha, maxIterations, reportFrequency):
        self.perform_gradient_descent(alpha, maxIterations, reportFrequency)
