import cantera as ct
import numpy as np


def isFloat(value):
    '''
    function isfloat
    ==========================================================================
    hacky python way to detect floats
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False


def loadData(fileName):
    '''
    function loadData
    ==========================================================================
    This function loads the raw data contained in the csv file and initiates
    a list of dictionaries containg the data
        fileName: file name of csv data
        Return: list of dictionaries for each example
    '''
    import csv
    rawData = []
    with open(fileName) as csvFile:
        reader = csv.DictReader(csvFile)
        for dictRow in reader:
            cleanDictRow = {key: (float(dictRow[key]) if isFloat(dictRow[key]) else dictRow[key]) for key in dictRow}
            rawData.append(cleanDictRow)
    return rawData


def getPressureData(fileName):
    '''
    function getPressureData
    ==========================================================================
    This function returns the formatted pressure vs time data
        Inputs:
            fileName: file name of csv data
        Outputs:
             t = time [s]
             p = pressure [Pa]
    '''
    rawData = loadData(fileName)
    t = np.array([example["Time (s)"] for example in rawData])
    p = np.array([example["Pressure (atm)"] for example in rawData])
    p *= ct.one_atm
    return (t, p)