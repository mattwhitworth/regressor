import numpy as np


class AbstractRegressor:
    def __init__(self, inputFile):
        self.inputValues = self.load_input_values(inputFile)
        self.inputFeatures = self.inputValues[:, :-1]
        self.inputYs = self.inputValues[:, -1:]
        self.thetas = np.ones((1, np.shape(self.inputFeatures)[1]))
        self.normalizedFeatures = self.normalize_features(self.inputFeatures)

    def calculate_hypothesis(self, inputValues):
        raise NotImplementedError("Please Implement this method")

    def compute_cost(self):
        raise NotImplementedError("Please Implement this method")

    def predict(self, fileName):
        raise NotImplementedError("Please Implement this method")

    def predict_for_plot(self, fileName):
        raise NotImplementedError("Please Implement this method")

    def train(self, alpha, maxIterations, reportFrequency):
        raise NotImplementedError("Please Implement this method")

    def load_input_values(self, filename):
        # load the input
        inputValues = np.loadtxt(filename, delimiter=",", ndmin=2)
        # insert a column of 1's to multiply with the Y intercept
        inputValues = np.insert(inputValues, 0, 1, axis=1)
        return inputValues

    def normalize_features(self, features):
        # subtract the mean, divide by standard deviation
        self.means = np.mean(features[:, 1:], axis=0)
        self.stds = np.std(features[:, 1:], axis=0)
        features = self.normalize_data(features)
        return features

    def normalize_data(self, data):
        # don't modify the input matrix
        copy = np.copy(data)
        # don't normalize the 1's column or the y's
        slicedValues = copy[:, 1:]
        # subtract the mean divide by standard deviation
        slicedValues = slicedValues - self.means
        # remove columns and thetas and std with a standard deviation of 0
        i = 0
        index = 0
        for std in self.stds:
            if np.isclose(std, 0):
                print("Removing feature" + str(index + 1))
                slicedValues = np.delete(arr=slicedValues, obj=i, axis=1)
                self.thetas = np.delete(arr=self.thetas, obj=i, axis=1)
                self.stds = np.delete(arr=self.stds, obj=i)
            else:
                i = i + 1
            index = index + 1
        # divide by standard deviation
        slicedValues = slicedValues / self.stds
        copy = np.insert(slicedValues, 0, 1, axis=1)
        return copy

    def update_thetas(self, alpha):
        # update thetas for gradient descent
        hypothesis = self.calculate_hypothesis(self.normalizedFeatures)
        differences = hypothesis - self.inputYs
        products = np.multiply(differences, self.normalizedFeatures)
        sums = products.sum(axis=0)
        newValues = sums / np.shape(self.normalizedFeatures)[0]
        self.thetas = self.thetas - (alpha * newValues)

    def perform_gradient_descent(self, alpha, maxIterations, reportFrequency):
        i = 0
        while i < maxIterations:
            self.update_thetas(alpha)
            if i % reportFrequency == 0:
                print("Current Cost", self.compute_cost())
            i = i + 1
        else:
            print("Final Cost", self.compute_cost())
            print("Final Thetas", self.thetas)

    def make_prediction(self, fileName):
        # make predictions based on the calculated thetas
        predictionData = self.load_input_values(fileName)
        normalizedPredictionData = self.normalize_data(predictionData)
        return self.calculate_hypothesis(normalizedPredictionData)

    def get_plot_x(self, fileName):
        xValues = [self.inputFeatures[:, 1:]]
        xValues.append(self.load_input_values(fileName)[:, 1:])
        return xValues

    def get_plot_y(self, fileName):
        yValues = [self.inputYs]
        yValues.append(self.predict_for_plot(fileName))
        return yValues
