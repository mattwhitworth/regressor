import sys
import random
import pyqtgraph as pg
from PyQt5 import QtGui


class Plotter:
    def plot_scatter_data(self, xValues, yValues, symbolList=['o', '+', 't', 'd', 's']):
        app = QtGui.QApplication(sys.argv)
        plotWidget = pg.plot(title="Hypothesis")
        for xVals, yVals, symbol in zip(xValues, yValues, symbolList):
            randomColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            plotWidget.scatterPlot(xVals[:, 0], yVals[:, 0], symbol=symbol, brush=randomColor, pen=randomColor)
        sys.exit(app.exec_())
