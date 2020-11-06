import pandas as pd
import numpy as np
import os, sys
from fnmatch import fnmatch
import random, math
import sklearn.model_selection as skm
from sklearn import metrics
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from timeit import default_timer as timer
import hashlib, datetime
from multiprocessing.pool import ThreadPool
import itertools
import xgboost as xgb
import copy

from weka.core.dataset import Instances, Instance, Attribute
from weka.classifiers import Classifier, Evaluation

# needs java jre and jdk version = 1.8
import weka.core.jvm as jvm
sys.path.append("./pykhiops/")
from pykhiops import PyKhiopsClassifier



def evaluateProba(m, x):
    """
    To use with sklearn wrapper only
    Computes probabilities from features for model evaluation

    :m - sk model:             model to be evaluated
    :x - dataframe:            features used to predict probas

    :return probas - iterable: predicted probabilities
    """
    return [i[1] for i in m.predict_proba(x)]

def evaluateProbaKhiops(m, x):
    """
    To use with khiops wrapper for sklearn only
    Same as evaluateProba() but necessary because of inconsistent interface

    May be unnecessary on future versions of Khiops

    :m - sk model:             model to be evaluated
    :x - dataframe:            features used to predict probas

    :return probas - iterable: predicted probabilities
    """
    return m.predict_proba(x).iloc[:, 2]

def evaluateScore(m, x):
    """
    To use with sklearn wrapper only
    Computes scores from features for model evaluation

    :m - sk model:             model to be evaluated
    :x - dataframe:            features used to predict scores

    :return scores - iterable: predicted scores
    """
    return m.decision_function(x)


def sklearnFit(model, evaluate, params = {}, n_trees=None):
    """
    Create a proper fit function with chosen attributes to be used into runModels.
    To be used with Sklearn models only, see others functions for Weka or XGBoost

    :model - object:        Scikit learn object reference to be used to create model
    :evaluate - function:   wrapper to make use of scores/probabilities uniformly
                            shall be either evaluateProba, evaluateProbaKhiops or evaluateScore
    :params - dict:         parameters to be passed to the model
    :n_trees - int:         used for Khiops only :
                                if n_trees > 0 this will trigger a random forest like algorithm
                                to try to improve khiops precision
                            see Khiops doc for more details

    :return fit - function: built training and evaluation function to be used inside runModels
    """
    def _fit(xtrain, ytrain, xtest, ytest):
        m = model(**params)
        start = timer()
        if n_trees:
            m.fit(xtrain, ytrain, n_trees=n_trees)
        else:
            m.fit(xtrain, ytrain)
        end = timer()
        return m.predict(xtrain), m.predict(xtest), evaluate(m, xtrain), evaluate(m, xtest), end-start
    return _fit

def xgbFit(objective, num_rounds=10, params = {}, prepy = lambda ys : ys.copy()):
    """
    Create a proper fit function with chosen attributes to be used into runModels.
    To be used with Sklearn models only, see others functions for Scikit learn or Weka

    :objective - string or callable: objective function to be used during training, see XGB doc
                                     can be passed as either string reference or custom callable function
    :num_rounds - integer:           parameter of the model, see XGB docs
    :params - dict:                  parameters to be passed to the model
    :prepy - callable:               preprocessing applied on the targets

    :return fit - function: built training and evaluation function to be used inside runModels
    """
    predict = lambda m, x : [1 if i>0 else -1 for i in m.predict(x)]
    predict_score = lambda m, x : m.predict(x)

    def _fit(xtrain, ytrain, xtest, ytest):
        dtrain = xgb.DMatrix(xtrain, label=prepy(ytrain))
        dtest = xgb.DMatrix(xtest)

        start = timer()
        if callable(objective):
            m = xgb.train(params, dtrain, num_boost_round=num_rounds, obj=objective, verbose_eval=True)
        else:
            params["objective"] = objective
            m = xgb.train(params, dtrain, num_boost_round=num_rounds, verbose_eval=True)
        end = timer()

        return predict(m, dtrain), predict(m, dtest), predict_score(m, dtrain), predict_score(m, dtest), end-start
    return _fit

def buildUnhinged(ep = 1e-32):
    def unhinged(preds, dtrain):
        ytrue = dtrain.get_label() * 2.0 - 1.0
        scpreds = preds * 2.0 - 1.0
        grad, hess = [], []
        for p, y in zip(scpreds, ytrue):
            grad.append(-1.0 * y);
            hess.append(ep)
        #print([i for i in zip(preds, ytrue, grad, hess)])
        #print(sum(hess))
        return grad, hess
    return unhinged

def buildModifiedRamp(ep = 1e-32, r = 0.5):
    def ramp(preds, dtrain):
        ytrue = dtrain.get_label() * 2.0 - 1.0
        scpreds = preds * 2.0 - 1.0
        grad, hess = [], []
        for p, y in zip(scpreds, ytrue):
            z = 1.0 * p * y
            g = 0.0
            h = ep
            if -(1/(r*2)) <= z <= (1/(r*2)):
                g = -r * y
            grad.append(g)
            hess.append(h)
            #print(p, y, z, g)
        #print([i for i in zip(preds, ytrue, grad, hess)])
        return grad, hess
    return ramp



def parralelize(f, runs, n_jobs):
    """
    Wrapper for starmap, runs a function on array of args with pooled or regular version of starmap

    :f - string:             function to run
    :runs - iterable:        iterable of arguments to run the function on
    :n_jobs - integer:       if n_jobs=1 starmap is run with itermaps, else with ThreadPool

    :return out - iterable:  unaggregated output of each function results
    """
    results = ThreadPool(n_jobs).starmap(f, runs) if(n_jobs!=1) else [r for r in itertools.starmap(f, runs)]
    return results

def splitXY(df, target="target"):
    """
    Split features X and target Y columns

    :df - pd.Dataframe:       input dataframe to split
    :target - string:         name of the Y column

    :return X - pd.Dataframe: features
    :return Y - pd.Series:    target values
    """
    return df.drop(target, axis=1), df[target]


def storeDF(df, out_path, output=True):
    """
    Store a dataframe as a compressed csv

    :df - pd.Dataframe: input dataframe
    :out_path - string: output csv.gz path
    :output - boolean:  verbose / not verbose

    :return - None:
    """
    df.to_csv(out_path, chunksize=100000, compression='gzip', mode="w", index=False)
    if(output):
        print("Stored {}".format(out_path))


def noised(y, rho, balance, info=True):
    """
    Noise a binary target Series (classes : +1/-1) with a rho * balance probability

    :y - pd.Series or any iterable: input target column
    :rho - float:                   flat noise insertion probability
    :balance - float:               balance/imbalance percentage of the minority class, used to scale the noise
    :info - boolean:                display the true information kept despite noise (noise verification)
                                    example : must be almost equal to 0.7 for rho = 0.3

    :return noisedY - pd.Series:    noised target values
    """
    bal_ratio = balance/100 # scaling percentage as ratio [0, 1]
    ny = y.apply(lambda x : -x if random.random()<(rho*bal_ratio) else x)
    if(info):
        # Accuracy used to verify the actual noise inserted
        print("Rho : {:.2f} Minority% : {:.3f} Scaled noise : {:.4f} True information kept : {:.4f}".format(rho, bal_ratio, rho*bal_ratio, metrics.accuracy_score(y, ny)))
    return ny

def prc_auc_score(ytrue, yscore):
    """
    Evaluation metric
    Compute the area under precision-recall curve

    :ytrue - iterable:   true labels
    :yscore - iterable:  scores computed by the model from features

    :return prc - float: area under precision recall curve value
    """
    precision, recall, _ = metrics.precision_recall_curve(ytrue, yscore)
    return metrics.auc(recall, precision)

def dfToWekaInstance( d):
    """
    Convert a pandas dataframe to the native weka data model (Instances)
    Dataframe -> Instances
    Columns   -> Attributes
    Row       -> Instance

    :d - DataFrame:           dataframe to be converted

    :return inst - Instances: native Weka data object
    """

    # work on a copy
    df = d.copy()

    # converting target
    df["target"] = df["target"].apply(lambda x : str(x))
    df["target"] = df["target"].astype("object")

    # functions to convert columns types into either numeric or nominal (mapping of pandas type to weka types)
    numeric = lambda n, _ : Attribute.create_numeric(n)
    nominal = lambda n, u : Attribute.create_nominal(n, u)
    mapping = {
        "int64" : numeric,
        "float64" : numeric
    }

    # applying conversions
    atts = [mapping.get(str(t), nominal)(c, df[c].sort_values().unique()) for c, t in zip(df.columns, df.dtypes)]
    inst = Instances.create_instances("tmp", atts, len(df))
    for c in df.loc[:, df.dtypes == object].columns:
        df[c] = df[c].factorize(sort=True)[0]
    for i, row in df.iterrows():
        inst.add_instance(Instance.create_instance([r for r in row]))

    # setting target column for learning into weka
    inst.class_is_last()

    return inst

def wekaFit(model = "weka.classifiers.trees.RandomForest", params = {}):
    """
    Create a proper fit function with chosen attributes to be used into runModels.
    To be used with Weka models only, see others functions for Sklearn or XGBoost

    :model - string:        Weka class to be used to create model
    :params - dict:         parameters to be passed to the model

    :return fit - function: built training and evaluation function to be used inside runModels
    """

    def _fit(xtrain, ytrain, xtest, ytest):
        train_inst = dfToWekaInstance(pd.concat([xtrain, ytrain], axis=1))
        opts = [
            "-I", str(params.get("n_estimators", 100)),
            "-num-slots", str(params.get("n_jobs", 1)),
            "-M", str(params.get("min_samples_leaf", 1)),
            "-depth", str(params.get("max_depth", 0))
        ]
        m = Classifier(classname=model, options=opts)
        start = timer()
        m.build_classifier(train_inst)
        end = timer()

        test_inst = dfToWekaInstance(pd.concat([xtest, ytest], axis=1))

        try:
            _predict = lambda  x : [{1: 1, 0: -1}[int(m.classify_instance(i))] for i in x]
            _proba = lambda  x : [m.distribution_for_instance(i)[1] for i in x]
        except JavaException as e:
            raise e

        return (_predict(train_inst),
                _predict(test_inst),
                _proba(train_inst),
                _proba(test_inst),
                end-start )
    return _fit



class Protocol:
    def __init__(self, K, R, RHO, PREPROCESSING, DATA_FOLDER = "Cleaned_Datasets", NOISED_DATA_FOLDER = "NoisedKFolds", PATH_SPLIT = "-", EXTENSION = "csv.gz", SEED = 16, EPSILON = 1e-32):
        """
        :K - integer:          see RepeatedStratifiedKFold doc
        :R - integer:          see RepeatedStratifiedKFold doc
        :RHO - integer:        noise applied to the splits (scaled with respect to the class balance)
        :EXTENSION - string:   extension of the dump files.
        """
        self.K = K
        self.R = R
        self.RHO = RHO
        self.DATA_FOLDER = DATA_FOLDER
        self.NOISED_DATA_FOLDER = NOISED_DATA_FOLDER
        self.PATH_SPLIT = PATH_SPLIT
        self.EXTENSION = EXTENSION
        self.SEED = SEED
        self.EPSILON = EPSILON
        self.PREPROCESSING = PREPROCESSING
        self.DATASETS = pd.read_json(os.path.join(DATA_FOLDER, "caracs.json"))

        self.RUN_FOLDER = os.path.join(NOISED_DATA_FOLDER, "RUN_{}x{}x{}".format(R, K, len(RHO)))
        try:
            os.mkdir(self.RUN_FOLDER)
        except OSError as e:
            print("Error: {}".format(e))

        #for weka
        jvm.start()

    def dataPath(self, name, xy, rk, split, rho=None, prep=None):
        """
        Datasets split and processed characteristics are stored within uniform file names

        :name - string:         dataset name
        :rk - string:           number of the repeated kfold split
        :xy - string:           "X" or "Y" whether features or classes will be stored
        :split - string:        "train" or "test" whether training or testing individuals will be stored
        :rho - float or string: noise probability applied on the dataset. ON "Y" ONLY
        :prep - string:         name of the preprocessing applied as defined in PREPROCESSING. ON "X" ONLY


        :return path - string:  generated output path
        """
        return os.path.join(self.RUN_FOLDER, self.PATH_SPLIT.join([str(i) for i in [name, rk, xy, rho, split, prep] if i is not None])+".{}".format(self.EXTENSION))



    def evaluateTrainTest(self, xtrain, ytrain, xtest, ytest, name, rho, m_name, modelFit, i):
        """
        Learns and evaluate a model according to the train/test split
        Returns results under a dict format

        :xtrain - dataframe:    features of the train split
        :xtest  - dataframe:    features of the test split
        :ytrain - dataframe:    targets of the train split
        :ytest  - dataframe:    targets of the test split
        :name - string:         name of the dataset
        :rho - float:           level of noise (rho*balance is applied on the dataset, not raw rho)
        :m_name - string:       name of the model tested
        :modelFit - callable:   function to fit the model (sklearnFit, wekaFit or xgbFit)
        :i - int:               identifier of the fold (0 < i < R*K)

        :return results - dict: float and string desciptors of computed metrics
        """

        # only minimum information is output if exception is raised in learning or evaluation
        out = {
            "name" : name,
            "model" : m_name,
            "rho" : rho,
            "i" : i
        }
        try:
            # fits the model
            ypredtrain, ypredtest, yscoretrain, yscoretest, time = modelFit(xtrain, ytrain, xtest, ytest)
            print("{} - {} ({} / {}) evaluated in {:.2f} ".format(name, m_name, i, rho, time))
            # computes the learning time (can be unprecise due to multiprocessing implementation)
            out["time"] = time

            # computes set of metrics on both train and test sets
            eval_metrics = {
                "rocauc" : lambda y, ys, yp : metrics.roc_auc_score(y, ys),
                "prauc" : lambda y, ys, yp : prc_auc_score(y, ys),
                "bacc" : lambda y, ys, yp : metrics.balanced_accuracy_score(y, yp),
                "kappa" : lambda y, ys, yp : metrics.cohen_kappa_score(y, yp)
            }
            splits = {
                "train" : (ytrain, yscoretrain, ypredtrain),
                "test" : (ytest, yscoretest, ypredtest)
            }
            for m, eval_metric in eval_metrics.items():
                for s, split in splits.items():
                    out["{}_{}".format(s, m)] = eval_metric(*split)

        except Exception as e:
            # exception is printed
            print(m_name, e)

        return out

    def splitDataset(self, name, balance, n_jobs=1):
        """
        Splits datasets into train/test and X/Y files, according to repeated stratified k fold.
        X are stored with every rho values and every preprocessing defined in PREPROCESSING

        :name - string:          name of the dataset to be processed
        :balance - float:        balance of the minority class to compute the scaled noise level
        :n_jobs - integer:       parallelization through parralelize()
        """
        file = os.path.join(self.DATA_FOLDER, name+".csv")
        print("--- Loaded {}".format(file))

        d = pd.read_csv(file)
        X, Y = splitXY(d)

        folds = skm.RepeatedStratifiedKFold(n_splits=self.K, n_repeats=self.R, random_state=self.SEED)
        for i, indexes in enumerate(folds.split(X, Y)):
            runs = []
            # test
            for p, prep in self.PREPROCESSING.items():
                runs.append((prep(X).loc[indexes[1], :], self.dataPath(name, "X", i, "test", prep=p)))
            runs.append((Y.loc[indexes[1]], self.dataPath(name, "Y", i, "test", rho=0.0)))

            # train
            for p, prep in self.PREPROCESSING.items():
                runs.append((prep(X).loc[indexes[0], :], self.dataPath(name, "X", i, "train", prep=p)))
            for rho in self.RHO:
                runs.append((noised(Y.loc[indexes[0]], rho, balance), self.dataPath(name, "Y", i, "train", rho=rho)))

            parralelize(storeDF, runs, n_jobs)

    def runModels(self, dataset_name, models, rk, n_jobs=1):
        """
        Runs a list of models through multiple preprocessing on a dataset
        The dataset must have been split with splitDataset() previously
        Split input datasets are retrieved with dataPath() automatically from protocol parameters
        Results are output to a file through dumpResults()

        :dataset_name - string: name of the dataset to be evaluated
        :models - dict:         dict of models to be tested. The dict must correspond to this example format :
                                    {
                                     "preprocessing_1": {
                                        "mysklmodel" : sklearnFit(SklearnModel, evaluateProba)
                                    }
                                This would run one model "mysklmodel" on datasets preprocessed with "preprocessing_1".
                                Fit function can correspond to any function corresponding to sklearnFit, wekaFit or xgbFit. See this functions for more details
        :rk - integer:          identifier of the split to use
        :n_jobs - integer:      parallelization through parralelize()
        """

        ytest = pd.read_csv(self.dataPath(dataset_name, "Y", rk, "test", rho=0.0))["target"]
        for prep, ms in models.items():
            runs = []
            if ms:
                xtest = pd.read_csv(self.dataPath(dataset_name, "X", rk, "test", prep=prep))
                xtrain = pd.read_csv(self.dataPath(dataset_name, "X", rk, "train", prep=prep))
                for rho in self.RHO:
                    ytrain = pd.read_csv(self.dataPath(dataset_name, "Y", rk, "train", rho=rho))["target"]
                    for model_name, model in ms.items():
                        runs.append((xtrain, ytrain, xtest, ytest, dataset_name, rho, model_name, model, rk))
                results = parralelize(self.evaluateTrainTest, runs, n_jobs)
                self.dumpResults(dataset_name, results)


    def dumpResults(self, name, results):
        """
        Dump results obtained with evaluateTrainTest into a file named in function of the dataset, kfold and noises parameters
        Create a new file with an explicit path describing protocol parameters if it does not exist
        Else, only appends new rows at the end

        :name - string:  name of the evaluated dataset
        :results - dict: results to be output. Obtained from evaluateTrainTest()
        """
        out_path = os.path.join("Results", "results_{}_{}x{}x{}.csv".format(name, self.R, self.K, len(self.RHO)))
        pd.DataFrame(results).to_csv(out_path, mode='a', header=not os.path.isfile(out_path), index=False)
