import pandas as pd
import os
import arff
import matplotlib.pyplot as plt
import numpy as np
from fnmatch import fnmatch
import re

def loadDataset(path, header=None, drop=None, sep=",", target="target", na=None, delim_whitespace=False, skiprows=0, binarize_target=None, verbose=0, evaluate_verbose=False):
    """
    Loads a dataset with various format and returns a cleaned standardized version to be inputted later into learning algorithms
    Any file with a csv structure can be inputted (.csv, .tsv, .data) considering relevant pd.read_csv parameters are set (separator, skiprows, header, ...)

    :path - string:              path of the input file
    :header - iterable:          if not None, header must correspond to the name of the columns (len(header) == len(columns))
    :drop - iterable:            if not None, the list of columns to be dropped and ignored in the process
    :sep - string:               csv-like separator character. For more information see pandas.read_csv documentation
    :target - string:            target column, to be used later as Y
    :na - string:                if not None, the string value to be considered as a missing value. ONLY NUMERICAL COLUMNS ARE HANDLED, Missing values in string columns are already considered as their own
    :delim_whitespace - boolean: if true then sep='\s+'. For more information see pandas.read_csv documentation
    :skiprows - integer:         number of rows in the original file to be ignored. For more information see pandas.read_csv documentation
    :binarize_target - function: if not None, the function must correspond to a mapping [any iterable] -> [iterable with binary values]. This function can correspond to any decision/threshhold function
    :verbose - integer:          if >0, the number of heading rows displayed at the end of the cleaning process. Debugging purposes
    :unique_verbose - boolean:   whether unique values of each columns should be displayed. Debugging purposes

    :return dataset - Dataframe: clean standardized dataframe
    """
    print("Loading {}".format(path))
    params = {"sep":sep, "delim_whitespace":delim_whitespace, "skiprows":skiprows}
    # header handling
    if(header):
        d = pd.read_csv(path, header=None, **params)
        d.columns = header
    else:
        d = pd.read_csv(path, **params)

    if(verbose>0):
        display(d.head(verbose))

    # drop unused columns
    if(drop):
        d = d.drop(drop, axis=1)

    # setting target column
    d = d.rename(columns={target: "target"})

    # uniform missing values (only considering missing in numeric columns)
    if(na):
        d = d.replace(na, np.nan)
        for c in d.columns[d.isna().any()]:
            try:
                d[c] = pd.to_numeric(d[c])

                # filling na values with avg column value
                d[c].fillna(d[c].mean(), inplace=True)
            except ValueError:
                d[c].fillna("missing", inplace=True)
                pass


    # header cleaning
    cleanString = lambda x : re.sub('[^A-Za-z0-9]+', '', x) if type(x)==str else x
    d.columns = [cleanString(c) for c in d.columns]

    # binary target {-1, +1}
    if(binarize_target):
        d["target"] = binarize_target(d["target"]).astype(int)
    index = d["target"].value_counts().sort_values(ascending=True).index
    t = lambda x : {index[0] : +1, index[1] : -1}[x]
    d["target"] = d["target"].apply(t)


    # values cleaning
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].apply(lambda x : str(x)).apply(lambda x : cleanString(x)).apply(lambda x : x if len(x)>0 else "missing")

    # display
    if(verbose>0):
        display(d.head(verbose))
    if(evaluate_verbose):
        print(evaluateDataset(d))
    return d


def countMissingValues(dataset):
    """
    Counts the number of missing value per columns

    :dataset - Dataframe:     dataset to be analyzed

    :return count - iterable: amount of missing value per column
    """
    return dataset.isna().sum()


def minorityBalance(dataset, target="target"):
    """
    Compute the balance (amount/total amount) of the minority (less represented) class. Should be used with binary classes only

    :dataset - Dataframe:  dataset to be analyzed
    :target - string:      name of the target/class column

    :return count - float: balance of the minority class
    """
    count = dataset[target].value_counts(dropna=False)
    if(len(count)!=2):
        print("Warning ! Class not binary ! ({})".format(len(count)))
    return count.min()/len(dataset)


def evaluateDataset(dataset):
    """
    Compute some metrics on the dataset to be inputted in a dataframe

    :dataset - Dataframe:   dataset to be analyzed

    :return metrics - dict: some metrics
    """
    missing_values = countMissingValues(dataset)
    return {
        "length" : len(dataset),
        "columns" : len(dataset.columns)-1,
        "cells" : len(dataset)*(len(dataset.columns)-1),
        "num_columns" : len(dataset.select_dtypes(include=np.number).columns)-1,
        "cat_columns" : len(dataset.select_dtypes(include="object").columns),
        "mean_cat_outcomes" : dataset.select_dtypes(include="object").apply(lambda x : len(x.unique()), axis=0).mean(),
        "columns_w_missing_values" : sum([c>0 for c in missing_values]),
        "missing_values_percent" : sum(missing_values)/(len(dataset)*len(dataset.columns)) * 100.0,
        "minority_balance_percent" : minorityBalance(dataset) * 100.0
    }
