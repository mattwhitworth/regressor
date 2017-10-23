from regressionEngine.linearRegressor import linearRegressor as linR
from regressionEngine.logisticRegressor import logisticRegressor as logR


class RegressorFactory:
    @staticmethod
    def createRegressor(type, trainingFile):
        if type == "linear":
            return linR.LinearRegressor(trainingFile)
        elif type == "logistic":
            return logR.LogisticRegressor(trainingFile)
        else:
            raise ValueError("Regressor type: %(regressorType) is not valid" % {"regressorType": type})
