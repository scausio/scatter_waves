import munch
import yaml
import os

def getConfigurationByID(path,confId):
    globalConf = yaml.load(open(os.path.join(path,"config.yaml")))
    return munch.Munch.fromDict(globalConf[confId])

def checkOutdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
