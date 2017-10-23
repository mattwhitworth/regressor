import numpy as np
from main import abstractRegressor as ar


class LinearRegressor(ar.AbstractRegressor):
    def calculate_hypothesis(self, inputValues):
        return inputValues.dot(self.thetas.transpose())

    def compute_cost(self):
        # calculate the cost function
        hypothesis = self.calculate_hypothesis(self.normalizedFeatures)
        cost = np.square(hypothesis - self.inputYs)
        cost = cost.sum() / (2 * (np.shape(self.normalizedFeatures)[0]))
        return cost

    def make_prediction_for_plot(self, fileName):
        return self.make_prediction(fileName)


def test():
    lr = LinearRegressor('../inputData/dataSet1')
    lr.perform_gradient_descent()
    lr.plot_prediction('../inputData/predictionData1')


def test_multiple():
    d = LinearRegressor('../inputData/dataSet2')
    d.perform_gradient_descent()
    print(d.make_prediction('../inputData/predictionData2'))

#test_multiple()
#test()
