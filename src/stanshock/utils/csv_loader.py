from __future__ import annotations

from pathlib import Path

import cantera as ct
import numpy as np


def is_float(value):
    """
    Hacky python way to detect floats.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_data(fileName):
    """
    This function loads the raw data contained in the csv file and initiates
    a list of dictionaries containing the data.
        fileName: file name of csv data
        Return: list of dictionaries for each example
    """
    import csv

    rawData = []
    with Path.open(fileName) as csvFile:
        reader = csv.DictReader(csvFile)
        for dictRow in reader:
            cleanDictRow = {
                key: (float(dictRow[key]) if is_float(dictRow[key]) else dictRow[key])
                for key in dictRow
            }
            rawData.append(cleanDictRow)
    return rawData


def get_pressure_data(fileName):
    """
    This function returns the formatted pressure vs time data.
        Inputs:
            fileName: file name of csv data
        Outputs:
             t = time [s]
             p = pressure [Pa]
    """
    rawData = load_data(fileName)
    t = np.array([example["Time (s)"] for example in rawData])
    p = np.array([example["Pressure (atm)"] for example in rawData])
    p *= ct.one_atm
    return (t, p)
