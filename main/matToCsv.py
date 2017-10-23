import scipy.io
import numpy as np
import csv
import itertools as IT

def convertFile(filename, saveLocation):

    data = scipy.io.loadmat(filename)

    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((saveLocation+i+".csv"),data[i],delimiter=',')


def combineCSV(fileOne, fileTwo, outputFile):
    filenames = [fileOne, fileTwo]
    handles = [open(filename, 'r') for filename in filenames]
    readers = [csv.reader(f, delimiter=',') for f in handles]

    with  open(outputFile, 'w') as h:
        writer = csv.writer(h, delimiter=',', lineterminator='\n', )
        for x, y in zip(*readers):
            writer.writerow(x + y)


    for f in handles:
        f.close()

def main():
    convertFile("inputData/ex3data1.mat" , "inputData/")
    #combineCSV("inputData/X.csv" , "inputData/y.csv", "inputData/numberTrainingData.csv")

main()