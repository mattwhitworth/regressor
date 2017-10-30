import argparse
from regressionEngine import regressorFactory as rf
from plotter import plotter as plot


# test logistic
# logistic inputData/logisticDataSet1 inputData/predictionData1 -p
# test linear
# linear inputData/linearDataSet1 inputData/predictionData1 -p
# test linear 3D
# linear inputData/linearDataSet2 inputData/predictionData2

def perform_regression(args):
    regressor = rf.RegressorFactory.createRegressor(type=args.regressorType, trainingFile=args.trainingFile)
    regressor.train(alpha=args.alpha, maxIterations=args.iterations, reportFrequency=args.frequency)

    print("Prediction:")
    print(regressor.predict(args.predictionFile))

    if args.plot:
        plotter = plot.Plotter()
        plotter.plot_scatter_data(regressor.get_plot_x(args.predictionFile), regressor.get_plot_y(args.predictionFile))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("regressorType", help="The type of regression to perform (linear or logistic)")
    parser.add_argument("trainingFile", help="File path to the CSV containing the training data (X and Y values")
    parser.add_argument("predictionFile", help="File path to the file containing the data to predict with (X values)")

    parser.add_argument("-a", "--alpha", type=float, default=0.03,
                        help="How fast to train, larger is more likely to diverge but faster(default=.03)")
    parser.add_argument("-f", "--frequency", type=int, default=100,
                        help="How often to print the current cost while training")
    parser.add_argument("-i", "--iterations", type=int, default=1000,
                        help="How long to train for")
    parser.add_argument("-p", "--plot", action="store_true", default=False,
                        help="Plots the prediction, only works on 2D data")
    return parser.parse_args()


def main():
    args = parse_arguments()
    perform_regression(args)


main()
