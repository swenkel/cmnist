###############################################################################
#                                                                             #
#                                                                             #
# Script for generating the CMNIST dataset                                    #
#                                                                             #
#                                                                             #
# (c) Simon Wenkel (https://www.simonwenkel.com)                              #
# released under the 3-Clause BSD license                                     #
#                                                                             #
# Technical documentation:                                                    #
#   https://www.simonwenkel.com/publications/articles/pdf/20190924_CMNIST.pdf #
#                                                                             #
# Folder structure                                                            #
# ./CMNIST/ <- contains csv files with generated datasets and label mappings  #
#          orig/ <- downloaded files                                          #
#                  e.g. EMNIST-MNIST/ <- dataset name is preserved            #
#          dumped/                                                            #
#               e.g. EMNIST-MNIST/ <- dataset name is preserved               #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################


import gc
import gzip
import hashlib
import itertools
import os
import random
import shutil
import sys
import time
import zipfile

from mnist import MNIST # pip: python-mnist
import numpy as np
import pandas as pd
from PIL import Image
import wget



def seed_everything(seed):
    """
    Getting rid of all the randomness in the world :(
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)



def downloadDataset(datasetName):
    """
    1. creates the folder structure required if not existing
    2. downloads dataset if not existing or checksum error detected
    """
    def checkCHECKSUM(file):
        """
        Check if correct file was downloaded (md5sum is enough for that)
        """
        f = open(file, "rb")
        try:
            fileHash =  hashlib.md5(f.read()).hexdigest()
        finally:
            f.close()
        return fileHash

    dumpedPath = CONSTANTS["PATH"]+"dumped/"
    dumpedPathDS = dumpedPath+datasetName+"/"
    if not os.path.exists(CONSTANTS["PATH"]):
        os.makedirs(CONSTANTS["PATH"])
    if not os.path.exists(dumpedPath):
        os.makedirs(dumpedPath)
    if not os.path.exists(dumpedPathDS):
        os.makedirs(dumpedPathDS)
    if "EMNIST" in datasetName: # work around for EMNIST file
        datasetName = "EMNIST"
    origPath = CONSTANTS["PATH"]+"orig/"
    origPathDS = origPath+datasetName+"/"
    if not os.path.exists(origPath):
        os.makedirs(origPath)
    if not os.path.exists(origPathDS):
        os.makedirs(origPathDS)

    for partition in CONSTANTS["URLS"][datasetName]:
        downloadStatus = False
        url = CONSTANTS["URLS"][datasetName][partition]
        fileName = url.split('/')[-1]
        checksumReference = CONSTANTS["MD5-CHECKSUMS"][datasetName][partition]
        if os.path.isfile(origPathDS+fileName):
            print(origPathDS+fileName,"downloaded already.")
            if checkCHECKSUM(origPathDS+fileName) != checksumReference:
                os.remove(origPathDS+fileName)
                print("Incorrect checksum. File broken. Re-download initiated.")
            else:
                downloadStatus = True
        if not downloadStatus:
            print("Downloading",url)
            print()
            wget.download(url, out=origPathDS+fileName)
            print()
            if checkCHECKSUM(origPathDS+fileName) != checksumReference:
                os.remove(origPathDS+fileName)
                raise ValueError("Incorrect checksum of "+origPathDS+fileName+ \
                            ". File deleted. Dataset generation aborted.")
            if datasetName == "EMNIST":
                unzip = zipfile.ZipFile(origPathDS+fileName,"r")
                unzip.extractall(origPathDS)
                unzip.close()
            elif url.split(".")[-1] == "gz":
                with gzip.open(origPathDS+fileName, "rb") as inputFile:
                        with open(origPathDS+fileName.split(".")[-2], "wb") as outputFile:
                            shutil.copyfileobj(inputFile, outputFile)
                inputFile.close()
                outputFile.close()
            downloadStatus = True
    return dumpedPath,dumpedPathDS,origPath,origPathDS



def generateBaseDatasets(datasetName,dumpedPath,dumpedPathDS,origPath,origPathDS):
    """
    Converting datasets in MNIST-format and NumPy arrays to an ImageNet-like
    folder structure
    """
    def dumpToImages(X,y,setType,datasetName,dumpedPathDS):
        """
        Dump images from numpy arrays to images of size 28x28(x1) - grayscale
        """
        def saveImg(X,y,fullPath,datasetName,SetType):
            imgDict = {}
            for i in range(len(X)):
                imgTmp = np.asarray(X[i])
                imgTmp = imgTmp.astype(np.uint8)
                img = Image.fromarray(imgTmp)
                filePathAndName = fullPath+str(y[i])+"/"+str(i)+".png"
                img.save(filePathAndName)
                imgDict[i] = {"file": filePathAndName,
                              "label": y[i]}
            print(datasetName,SetType,"set dumped.")
            return imgDict

        y_unique = np.unique(y)
        dumpedPathDS = dumpedPathDS+setType+"/"
        if setType == "train":
            if not os.path.exists(dumpedPathDS):
                os.makedirs(dumpedPathDS)
            for i in y_unique:
                if not os.path.exists(dumpedPathDS+"/"+str(i)+"/"):
                    os.makedirs(dumpedPathDS+"/"+str(i)+"/")
            result = saveImg(X,y,dumpedPathDS,datasetName,setType)
        elif setType == "test":
            if not os.path.exists(dumpedPathDS):
                os.makedirs(dumpedPathDS)
            for i in y_unique:
                if not os.path.exists(dumpedPathDS+"/"+str(i)+"/"):
                    os.makedirs(dumpedPathDS+"/"+str(i)+"/")
            result = saveImg(X,y,dumpedPathDS,datasetName,setType)
        return pd.DataFrame.from_dict(result, orient="index")


    def loadDataset(datasetName,origPathDS):
        """
        Load dataset from either MNIST-style files or dumped numpy arrays
        """
        def loadNPZ(f):
            return np.load(f)['arr_0']

        mnistDataType = True
        if  "EMNIST" not in datasetName and CONSTANTS["URLS"][datasetName]["X_train"].split(".")[-1] == "npz":
            mnistDataType = False
        if mnistDataType:
            if "EMNIST" in datasetName:
                origPathDS+="gzip/"
                mndata = MNIST(origPathDS)
                mndata.select_emnist(datasetName.split("-")[-1].lower())
            else:
                mndata = MNIST(origPathDS)
            X_train, y_train = mndata.load_training()
            X_test, y_test = mndata.load_testing()
            X_train = np.asarray(X_train).reshape(-1,28,28)
            X_test = np.asarray(X_test).reshape(-1,28,28)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
        else:
            X_train = loadNPZ(origPathDS+CONSTANTS["URLS"][datasetName]["X_train"].split("/")[-1])
            X_test = loadNPZ(origPathDS+CONSTANTS["URLS"][datasetName]["X_test"].split("/")[-1])
            y_train = loadNPZ(origPathDS+CONSTANTS["URLS"][datasetName]["y_train"].split("/")[-1])
            y_test = loadNPZ(origPathDS+CONSTANTS["URLS"][datasetName]["y_test"].split("/")[-1])
        return X_train,X_test,y_train,y_test


    if not os.path.isfile(dumpedPath+datasetName+"_train.csv") \
     and not os.path.isfile(dumpedPath+datasetName+"_test.csv"):
        X_train, X_test, y_train, y_test = loadDataset(datasetName,origPathDS)
        dfTrain = dumpToImages(X_train, y_train, "train", datasetName,dumpedPathDS)
        dfTrain["decoded_label"] = dfTrain["label"].apply(lambda x:CONSTANTS["LabelMapping"][datasetName][x])
        dfTrain.to_csv(dumpedPath+datasetName+"_train.csv",index=False)
        del X_train
        del y_train
        gc.collect()
        dfTest = dumpToImages(X_test, y_test, "test", datasetName,dumpedPathDS)
        dfTest["decoded_label"] = dfTest["label"].apply(lambda x:CONSTANTS["LabelMapping"][datasetName][x])
        dfTest.to_csv(dumpedPath+datasetName+"_test.csv",index=False)
        del X_test
        del y_test
        gc.collect()
        print(datasetName," dumped.")
    else:
        print(datasetName,"already dumped.")



def createCMNISTSubset(CMNISTNAME,datasets,dumpedPath,classFilter=False):
    """
    Creating CMNIST subsets from existing MNIST-like datasets
    Subsets are dumped as csv files with new labels and decoded labels
    Images themselves are not copied
    """
    if not os.path.isfile(dumpedPath+CMNISTNAME+"_train.csv") \
     and not os.path.isfile(dumpedPath+CMNISTNAME+"_test.csv"):
        dfTrain = pd.DataFrame(columns=["file","label","decoded_label","source"])
        dfTest = pd.DataFrame(columns=["file","label","decoded_label","source"])
        currLabelID = 0
        for datasetName in datasets:
            print("Adding",datasetName)
            dfTrainTmp = pd.read_csv(dumpedPath+datasetName+"_train.csv")
            dfTestTmp = pd.read_csv(dumpedPath+datasetName+"_test.csv")
            if classFilter:
                labels = datasets[datasetName]
            else:
                labels = dfTrainTmp["label"].unique()
            for label in labels:
                dfTmp = dfTrainTmp[dfTrainTmp["label"] == label].copy(deep=True)
                dfTmp["decoded_label"] = dfTmp["label"].apply(lambda x:CONSTANTS["LabelMapping"][datasetName][x])
                dfTmp["label"] = dfTmp["label"].apply(lambda x:currLabelID)
                dfTmp["source"] = [datasetName for i in range(len(dfTmp))]
                dfTrain = dfTrain.append(dfTmp, ignore_index=True)
                dfTmp = dfTestTmp[dfTestTmp["label"] == label].copy(deep=True)
                dfTmp["decoded_label"] = dfTmp["label"].apply(lambda x:CONSTANTS["LabelMapping"][datasetName][x])
                dfTmp["label"] = dfTmp["label"].apply(lambda x:currLabelID)
                dfTmp["source"] = [datasetName for i in range(len(dfTmp))]
                dfTest = dfTest.append(dfTmp, ignore_index=True)
                currLabelID+=1
        dfTrain.to_csv(dumpedPath+CMNISTNAME+"_train.csv",index=False)
        dfTest.to_csv(dumpedPath+CMNISTNAME+"_test.csv",index=False)
    else:
        print(CMNISTNAME,"was generated already.")



def createSubsetCombinations():
    def checkValidity(combination,exclusionDict):
        """
        Check if combination of datasets is valid using the exclusion dict
        """
        for dataset in combination:
            for exclusionSet in  exclusionDict[dataset]:
                if exclusionSet in combination:
                    validity = False
                    return validity
        validity = True
        return validity

    def checkOverlappingLabels(dataset1,dataset2):
        """
        Check if datasets contain overlapping labels.
        If so, they are added to the exclusion dict
        """
        dataset1Labels = list(CONSTANTS["LabelMapping"][dataset1].values())
        dataset2Labels = list(CONSTANTS["LabelMapping"][dataset2].values())
        for i in dataset1Labels:
            if i in dataset2Labels:
                overlapping = True
                return overlapping
        overlapping = False
        return overlapping

    exclusionDict = {}
    for dataset1 in list(CONSTANTS["LabelMapping"].keys()):
        exclusionDict[dataset1] = []
        for dataset2 in list(CONSTANTS["LabelMapping"].keys()):
            if dataset1 != dataset2:
                if checkOverlappingLabels(dataset1,dataset2):
                    exclusionDict[dataset1].append(dataset2)

    combinations = []
    for i in range(2,len(CONSTANTS["Datasets"])+1):
        for combination in list(itertools.combinations(list(CONSTANTS["Datasets"].keys()),i)):
            if checkValidity(combination, exclusionDict):
                combinations.append(combination)
    return combinations


def getDatasetCombinationName(datasetCombination):
    """
    CONSTANTS["Reduced-training-set-sizes"]
    """
    abbreviation = "CMNIST-"+str(len(datasetCombination))+"-"
    for dataset in datasetCombination:
        abbreviation+=CONSTANTS["Datasets"][dataset]
    return abbreviation


def generateReducedCMNIST(inputFile,dumpedPath):
    """
    Generate reduced CMNIST training sets as defined in
    CONSTANTS["Reduced-training-set-sizes"].
    Using the first i samples of each class.
    """
    trainDF = pd.read_csv(dumpedPath+inputFile)
    setSizes = CONSTANTS["Reduced-training-set-sizes"]
    reducedSets = {}
    for i in setSizes:
        reducedSets[i] = {}
    for label in sorted(trainDF["label"].unique()):
        dfTmp = trainDF[trainDF["label"] == label]
        for i in setSizes:
            reducedSets[i][label] = dfTmp.iloc[:i]
    for i in setSizes:
        dfTmp = trainDF.iloc[:0]
        for j in reducedSets[i]:
            dfTmp = dfTmp.append(reducedSets[i][j], ignore_index=True)
        outputfile = inputFile.split("_")[0]+"-r-"+str(i)+"_train.csv"
        dfTmp.to_csv(dumpedPath+outputfile,index=False)



def generateCMNIST():
    """
    Generate CMNIST with all subsets
    1. Download original MNIST-like datasets, if required
    2. Dump all images into an ImageNet-like structure and generate dataset CSVs
    3. Create all valid CMNIST subsets and CMNIST-X subsets
    4. Create reduced CMNIST training sets
    """
    for datasetName in CONSTANTS["Datasets"]:
        print("*" * 40)
        print("Dumping", datasetName)
        dumpedPath,dumpedPathDS,origPath,origPathDS = downloadDataset(datasetName)
        generateBaseDatasets(datasetName,
                             dumpedPath,
                             dumpedPathDS,
                             origPath,
                             origPathDS)
        print("*" * 40)
    print("*" * 40)
    print("Generating CMNIST-X-12")
    createCMNISTSubset("CMNIST-X-12",
                       CONSTANTS["CMNIST-X"]["CMNIST-X-12"],
                       dumpedPath,
                       classFilter=True)
    print("*" * 40)
    print("Generating CMNIST-X-24")
    createCMNISTSubset("CMNIST-X-24",
                       CONSTANTS["CMNIST-X"]["CMNIST-X-24"],
                       dumpedPath,
                       classFilter=True)
    print("*" * 40)
    datasetCombinations = createSubsetCombinations()
    for datasetCombination in datasetCombinations:
        datasetCombinationName = getDatasetCombinationName(datasetCombination)
        print("*" * 40)
        print("Generating",datasetCombinationName)
        createCMNISTSubset(datasetCombinationName,
                           datasetCombination,
                           dumpedPath,
                           classFilter=False)
        print("*" * 40)
    datasets = sorted([x for x in os.listdir(dumpedPath) \
                       if "CMNIST" in x and "train" in x])
    reducedSetsExsists = [True for x in datasets if "-r-" in x]
    if True in reducedSetsExsists:
        "Reduced datasets generated already."
    else:
        for dataset in datasets:
            print("*" * 40)
            print("Generating reduced",dataset.split("_")[0])
            generateReducedCMNIST(dataset,dumpedPath)
            print("*" * 40)




CONSTANTS = {}
CONSTANTS["SEED"] = 1
CONSTANTS["PATH"] = "./CMNIST/"
CONSTANTS["Datasets"] = dict(zip(["EMNIST-MNIST", "EMNIST-Digits",
                                  "EMNIST-Letters", "EMNIST-Balanced",
                                  "EMNIST-Bymerge", "EMNIST-Byclass",
                                  "Fashion-MNIST", "Kannada","KMNIST", "K49",
                                  "notMNIST"],
                                  ["E", "Ed", "El", "Eb", "Em", "Ec", "F", "Kd",
                                   "Km", "K49", "N"]))
CONSTANTS["URLS"] = {}
CONSTANTS["URLS"]["EMNIST"] = {}
CONSTANTS["URLS"]["EMNIST"]["fullset"] = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip" # mnist filetype
#CONSTANTS["URLS"]["EMNIST"]["fullset"] = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip" # matlab files
CONSTANTS["URLS"]["Fashion-MNIST"] = {}
CONSTANTS["URLS"]["Fashion-MNIST"]["X_train"] = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
CONSTANTS["URLS"]["Fashion-MNIST"]["X_test"] = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
CONSTANTS["URLS"]["Fashion-MNIST"]["y_train"] = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
CONSTANTS["URLS"]["Fashion-MNIST"]["y_test"] = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
CONSTANTS["URLS"]["Kannada"] = {}
CONSTANTS["URLS"]["Kannada"]["X_train"] = "https://github.com/vinayprabhu/Kannada_MNIST/raw/master/data/output_tensors/MNIST_format/X_kannada_MNIST_train.npz"
CONSTANTS["URLS"]["Kannada"]["X_test"] = "https://github.com/vinayprabhu/Kannada_MNIST/raw/master/data/output_tensors/MNIST_format/X_kannada_MNIST_test.npz"
CONSTANTS["URLS"]["Kannada"]["y_train"] = "https://github.com/vinayprabhu/Kannada_MNIST/raw/master/data/output_tensors/MNIST_format/y_kannada_MNIST_train.npz"
CONSTANTS["URLS"]["Kannada"]["y_test"] = "https://github.com/vinayprabhu/Kannada_MNIST/raw/master/data/output_tensors/MNIST_format/y_kannada_MNIST_test.npz"
CONSTANTS["URLS"]["KMNIST"] = {}
CONSTANTS["URLS"]["KMNIST"]["X_train"] = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz"
CONSTANTS["URLS"]["KMNIST"]["X_test"] = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz"
CONSTANTS["URLS"]["KMNIST"]["y_train"] = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz"
CONSTANTS["URLS"]["KMNIST"]["y_test"] = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz"
CONSTANTS["URLS"]["K49"] = {}
CONSTANTS["URLS"]["K49"]["X_train"] = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz"
CONSTANTS["URLS"]["K49"]["X_test"] = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz"
CONSTANTS["URLS"]["K49"]["y_train"] = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz"
CONSTANTS["URLS"]["K49"]["y_test"] = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz"
CONSTANTS["URLS"]["notMNIST"] = {}
CONSTANTS["URLS"]["notMNIST"]["X_train"] = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-images-idx3-ubyte.gz"
CONSTANTS["URLS"]["notMNIST"]["X_test"] = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-images-idx3-ubyte.gz"
CONSTANTS["URLS"]["notMNIST"]["y_train"] = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-labels-idx1-ubyte.gz"
CONSTANTS["URLS"]["notMNIST"]["y_test"] = "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-labels-idx1-ubyte.gz"
CONSTANTS["MD5-CHECKSUMS"] = {}
CONSTANTS["MD5-CHECKSUMS"]["EMNIST"] = {}
CONSTANTS["MD5-CHECKSUMS"]["EMNIST"]["fullset"] = "58c8d27c78d21e728a6bc7b3cc06412e" # mnist filetype
#CONSTANTS["MD5-CHECKSUMS"]["EMNIST"]["fullset"]= "1bbb49fdf3462bb70c240eac93fff0e4" # matlab files
CONSTANTS["MD5-CHECKSUMS"]["Fashion-MNIST"] = {}
CONSTANTS["MD5-CHECKSUMS"]["Fashion-MNIST"]["X_train"] = "8d4fb7e6c68d591d4c3dfef9ec88bf0d"
CONSTANTS["MD5-CHECKSUMS"]["Fashion-MNIST"]["X_test"] = "bef4ecab320f06d8554ea6380940ec79"
CONSTANTS["MD5-CHECKSUMS"]["Fashion-MNIST"]["y_train"] = "25c81989df183df01b3e8a0aad5dffbe"
CONSTANTS["MD5-CHECKSUMS"]["Fashion-MNIST"]["y_test"] = "bb300cfdad3c16e7a12a480ee83cd310"
CONSTANTS["MD5-CHECKSUMS"]["Kannada"] = {}
CONSTANTS["MD5-CHECKSUMS"]["Kannada"]["X_train"] = "8b2a3078f805b92ab2511d75932f8372"
CONSTANTS["MD5-CHECKSUMS"]["Kannada"]["X_test"] = "a21bc9f8cbca65fc9c359e8c8665a077"
CONSTANTS["MD5-CHECKSUMS"]["Kannada"]["y_train"] = "2d4859bfa11c546bd593aafd86c1aa31"
CONSTANTS["MD5-CHECKSUMS"]["Kannada"]["y_test"] = "834d18e99f16665f6a28b7f8dd92eece"
CONSTANTS["MD5-CHECKSUMS"]["KMNIST"] = {}
CONSTANTS["MD5-CHECKSUMS"]["KMNIST"]["X_train"] = "a183fbc01ed9a345513b145673d3d0bb"
CONSTANTS["MD5-CHECKSUMS"]["KMNIST"]["X_test"] = "dace6de62ca8b90843cc45366337d02e"
CONSTANTS["MD5-CHECKSUMS"]["KMNIST"]["y_train"] = "b0e1d240e8afec72b50437fc3bc5908a"
CONSTANTS["MD5-CHECKSUMS"]["KMNIST"]["y_test"] = "3ae489a79983d0fe386152684e3b5412"
CONSTANTS["MD5-CHECKSUMS"]["K49"] = {}
CONSTANTS["MD5-CHECKSUMS"]["K49"]["X_train"] = "7ac088b20481cf51dcd01ceaab89d821"
CONSTANTS["MD5-CHECKSUMS"]["K49"]["X_test"] = "d352e201d846ce6b94f42c990966f374"
CONSTANTS["MD5-CHECKSUMS"]["K49"]["y_train"] = "44a8e1b893f81e63ff38d73cad420f7a"
CONSTANTS["MD5-CHECKSUMS"]["K49"]["y_test"] = "4da6f7a62e67a832d5eb1bd85c5ee448"
CONSTANTS["MD5-CHECKSUMS"]["notMNIST"] = {}
CONSTANTS["MD5-CHECKSUMS"]["notMNIST"]["X_train"] = "d916d7283fce4d08db9867c640ec0042"
CONSTANTS["MD5-CHECKSUMS"]["notMNIST"]["X_test"] = "2c87c839a4ef9b238846600eec8c35b7"
CONSTANTS["MD5-CHECKSUMS"]["notMNIST"]["y_train"] = "eab59f88903339e01dac19deed3824c0"
CONSTANTS["MD5-CHECKSUMS"]["notMNIST"]["y_test"] = "7ea9118cbafd0f6e3ee2ad771d782a01"
CONSTANTS["LabelMapping"] = {}
CONSTANTS["LabelMapping"]["EMNIST-MNIST"] = dict(zip([i for i in range(10)],
                                                     [str(i) for i in range(10)]))
CONSTANTS["LabelMapping"]["EMNIST-Digits"] = dict(zip([i for i in range(10)],
                                                     [str(i) for i in range(10)]))
CONSTANTS["LabelMapping"]["EMNIST-Letters"] = dict(zip([i for i in range(1,27)],
                                                        ["A","B","C","D","E",
                                                        "F","G","H","I","J",
                                                        "K","L","M","N","O",
                                                        "P","Q","R","S","T",
                                                        "U","V","W","X","Y",
                                                        "Z"]))
CONSTANTS["LabelMapping"]["EMNIST-Balanced"] = dict(zip([i for i in range(0,47)],
                                                        ["0","1","2","3","4",
                                                        "5","6","7","8","9",
                                                        "A","B","C","D","E",
                                                        "F","G","H","I","J",
                                                        "K","L","M","N","O",
                                                        "P","Q","R","S","T",
                                                        "U","V","W","X","Y",
                                                        "Z","a","b","d","e",
                                                        "f","g","h","n","q",
                                                        "r","t"]))
CONSTANTS["LabelMapping"]["EMNIST-Bymerge"] = dict(zip([i for i in range(0,47)],
                                                        ["0","1","2","3","4",
                                                        "5","6","7","8","9",
                                                        "A","B","C","D","E",
                                                        "F","G","H","I","J",
                                                        "K","L","M","N","O",
                                                        "P","Q","R","S","T",
                                                        "U","V","W","X","Y",
                                                        "Z","a","b","d","e",
                                                        "f","g","h","n","q",
                                                        "r","t"]))
CONSTANTS["LabelMapping"]["EMNIST-Byclass"] = dict(zip([i for i in range(0,62)],
                                                       ["0","1","2","3","4",
                                                       "5","6","7","8","9",
                                                       "A","B","C","D","E",
                                                       "F","G","H","I","J",
                                                       "K","L","M","N","O",
                                                       "P","Q","R","S","T",
                                                       "U","V","W","X","Y",
                                                       "Z","a","b","c","d",
                                                       "e","f","g","h","i",
                                                       "j","k","l","m","n",
                                                       "o","p","q","r","s",
                                                       "t","u","v","w","x",
                                                       "y","z"]))
CONSTANTS["LabelMapping"]["Fashion-MNIST"] = dict(zip([i for i in range(10)],
                                                      ["T-shirt/top","Trouser",
                                                      "Pullover", "Dress",
                                                      "Coat","Sandal",
                                                      "Shirt","Sneaker",
                                                      "Bag","Ankle boot"]))
CONSTANTS["LabelMapping"]["Kannada"] = dict(zip([i for i in range(10)],
                                                ["೦","೧","೨","೩","೪","೫",
                                                 "೬","೭","೮","೯"]))
CONSTANTS["LabelMapping"]["KMNIST"] = dict(zip([i for i in range(10)],
                                               ["お","き","す","つ","な",
                                               "は","ま","や","れ","を"]))
CONSTANTS["LabelMapping"]["K49"] = dict(zip([i for i in range(49)],
                                            ["あ","い","う","え","お","か","き",
                                            "く","け","こ","さ","し","す","せ",
                                            "そ","た","ち","つ","て","と","な",
                                            "に","ぬ","ね","の","は","ひ","ふ",
                                            "へ","ほ","ま","み","む","め","も",
                                            "や","ゆ","よ","ら","り","る","れ",
                                            "ろ","わ","ゐ","ゑ","を","ん","ゝ"]))
CONSTANTS["LabelMapping"]["notMNIST"] = dict(zip([i for i in range(10)],
                                                        ["A","B","C","D","E",
                                                        "F","G","H","I","J"]))
CONSTANTS["CMNIST-X"] = {}
CONSTANTS["CMNIST-X"]["CMNIST-X-12"] = {}
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["EMNIST-MNIST"] = [0,4]
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["EMNIST-Letters"] = [15,21]
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["Fashion-MNIST"] = [0,8]
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["Kannada"] = [1,8]
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["KMNIST"] = [5,9]
CONSTANTS["CMNIST-X"]["CMNIST-X-12"]["notMNIST"] = [1,9]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"] = {}
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["EMNIST-MNIST"] = [0,4,5,9]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["EMNIST-Letters"] = [15,16,19,21]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["Fashion-MNIST"] = [0,4,7,8]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["Kannada"] = [1,2,4,8]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["KMNIST"] = [5,7,8,9]
CONSTANTS["CMNIST-X"]["CMNIST-X-24"]["notMNIST"] = [1,3,4,9]
CONSTANTS["Reduced-training-set-sizes"] = [1,5,10,50,100]


def main():
    startTime = time.time()
    seed_everything(CONSTANTS["SEED"])
    print("=" * 80)
    print("Generating CMNIST")
    print()
    generateCMNIST()
    print()
    print("CMNIST generated in {:.2f} minutes.".format((time.time()-startTime)/60))
    print("=" * 80)


if __name__ == "__main__":
    main()
