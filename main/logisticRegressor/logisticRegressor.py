import numpy as np
from main import abstractRegressor as ar

np.set_printoptions(threshold=np.nan)


class LogisticRegressor(ar.AbstractRegressor):
    e = 2.71828

    def calculate_hypothesis(self, inputValues):
        exponent = -(inputValues.dot(self.thetas.transpose()))
        return (1 / (1 + self.e ** exponent))

    def compute_cost(self):
        # calculate the cost function
        hypothesis = self.calculate_hypothesis(self.normalizedFeatures)
        print(hypothesis.min())
        cost = (self.inputYs * np.log(hypothesis)) + ((1 - self.inputYs) * np.log(1 - hypothesis))
        cost = -(cost.sum() / (np.shape(self.inputValues)[0]))
        return cost

    def perform_multiclass_classification_descent(self):
        # perform gradient descent for each class type using one vs. all
        originalys = np.copy(self.inputYs)
        classRange = int(np.max(self.inputYs)) + 1 - int(np.min(self.inputYs))
        classThetas = np.empty((classRange, np.shape(self.thetas)[1]))
        for classValue in range(classRange):
            self.inputYs = np.where(originalys == classValue, 1, 0)
            print('Performing descent for type', classValue)
            self.perform_gradient_descent(reportFrequency=300, alpha=0.003, maxIterations=10000)
            classThetas[classValue:, :] = self.thetas
        self.thetas = classThetas
        self.inputYs = originalys
        print('Total Thetas \n', self.thetas)

    def make_multiclass_prediction(self, fileName):
        predictionMatrix = self.make_prediction(fileName)
        maxes = np.amax(predictionMatrix, axis=1)
        types = np.argmax(predictionMatrix, axis=1)
        returnMatrix = np.empty((np.shape(maxes)[0], 3))
        index = np.linspace(start=1, stop=np.shape(maxes)[0], num=np.shape(maxes)[0], retstep=False)

        returnMatrix[:, 0] = index.transpose()
        returnMatrix[:, 1] = types.transpose()
        returnMatrix[:, 2] = maxes.transpose()
        return returnMatrix

    def make_prediction_for_plot(self, fileName):
        return self.make_multiclass_prediction(fileName)[:, 1:-1]


def test():
    lr = LogisticRegressor('../inputData/logisticDataSet1')
    lr.perform_multiclass_classification_descent()

    print("Prediction:")
    print(lr.make_multiclass_prediction('../inputData/predictionData1'))

    lr.plot_prediction('../inputData/predictionData1')

#test()
