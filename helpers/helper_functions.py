# -*- coding: utf-8 -*-



import time
import re
import os
import string
import json
import codecs
import pickle
import collections
# from itertools import permutations
from collections import Counter
import random
import gc
import math
import numbers
import heapq
from ast import literal_eval



import networkx as nx
from networkx.utils import pairwise
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
pd.set_option('display.max_columns', None)
import fiona
os.environ['USE_PYGEOS'] = '0'  ##-- to allows geopandas to use Shapely2.0 instead
import geopandas as gp

from shapely import wkt
from shapely.geometry import box
from shapely.geometry import shape, mapping
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import LinearRing
from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.affinity import translate
from shapely.strtree import STRtree
from shapely.ops import nearest_points
from shapely.ops import unary_union
from shapely.ops import cascaded_union
from shapely.ops import transform

#from shapely.validation import explain_validity
#import gdal
from pyproj import Proj, Transformer

# import geopy
# import geopy.distance
#from geopy import Point
#import momepy

import rasterio
#import rasterio.env
#from rasterio.features import shapes

from scipy.spatial import distance_matrix
# from scipy.spatial import distance
# from scipy.spatial import cKDTree
# from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit   ## for the elevation profile approximation
from statistics import mode
from statistics import stdev

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score as randIndex
from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI
from sklearn.metrics.cluster import contingency_matrix
import munkres

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestClassifier

# import statsmodels.api as sm

#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
#from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import colorsys

# import contextily as ctx  ## needed for plotting on basemaps

mapboxKey = 'pk.eyJ1Ijoic2h1dG9hcmFraSIsImEiOiJja2F4bGpwZGgwMXdoMnNwaTZwNzZ1N2ozIn0.4MK9evmXh1eQPTUauJQbMg'


# import sys
# print(sys.version)
# print(nx.__version__)

tokyoCoreBounds = [139.6774997727269, 35.644167974043, 139.7859865903687, 35.699309477838504]


###=====================================================================
standardCRS = 'epsg:4326'
mappingCRS = 'epsg:3857'
areaCalcCRS = '+proj=cea +lat_0=35.6812 +lon_0=139.7671 +units=m'
distCalcCRS = '+proj=eqc +lat_0=35.6812 +lon_0=139.7671 +units=m'
# test edit

###=== For use with direct pyproj transforming
#starndardProj = Proj('epsg:4326')
#mappingProj = Proj('epsg:3857')
#areaCalcProj = Proj('+proj=cea +lat_0=35.6812 +lon_0=139.7671 +units=m')
#distCalcProj = Proj('+proj=eqc +lat_0=35.6812 +lon_0=139.7671 +units=m')

###=== Projection creator and presets for common conversions
def createProjection(fromCRS, toCRS):
    return Transformer.from_proj(Proj(fromCRS), Proj(toCRS), always_xy=True)

standardToDistProj = createProjection(standardCRS, distCalcCRS)
standardToMapProj = createProjection(standardCRS, mappingCRS)
standardToAreaProj = createProjection(standardCRS, areaCalcCRS)

distToStandardProj = createProjection(distCalcCRS, standardCRS)
distToMapProj = createProjection(distCalcCRS, mappingCRS)
distToAreaProj = createProjection(distCalcCRS, areaCalcCRS)

areaToStandardProj = createProjection(areaCalcCRS, standardCRS)
areaToMapProj = createProjection(areaCalcCRS, mappingCRS)
areaToDistProj = createProjection(areaCalcCRS, distCalcCRS)

mapToStandardProj = createProjection(mappingCRS, standardCRS)
mapToDistProj = createProjection(mappingCRS, distCalcCRS)
mapToAreaProj = createProjection(mappingCRS, areaCalcCRS)

def createDistToStandardProj():
    return createProjection(distCalcCRS, standardCRS)
def createDistToMappingProj():
    return createProjection(distCalcCRS, mappingCRS)
def createAreaToStandardProj():
    return createProjection(areaCalcCRS, standardCRS)
def createstandardToDistProj():
    return createProjection(standardCRS, distCalcCRS)
def createStandardToAreaProj():
    return createProjection(standardCRS, areaCalcCRS)
def createStandardToMapProj():
    return createProjection(standardCRS, mappingCRS)

###=== Convert a geometry from one CRS to another using a projection created once, supports vectorized application
def convertGeomCRS(thisGeom, thisProjection=None, toCRS=None, fromCRS=standardCRS):
    bufferedTypes = ['Polygon', 'MultiPolygon']
    if thisProjection != None:  ##-- if the projection is specified
        if isinstance(thisGeom,list):
            return [transform(thisProjection.transform, geom).buffer(0) if geom.geom_type in bufferedTypes else transform(thisProjection.transform, geom) for geom in thisGeom]
        else:
            return transform(thisProjection.transform, thisGeom).buffer(0) if thisGeom.geom_type in bufferedTypes else transform(thisProjection.transform, thisGeom)
    elif toCRS != None:   ##-- if the fromCRS is isn't specified, assume it's from standardCRS
        thisProjection = Transformer.from_proj(Proj(fromCRS), Proj(toCRS), always_xy=True)
        if isinstance(thisGeom,list):
            return [transform(thisProjection.transform, geom).buffer(0) if geom.geom_type in bufferedTypes else transform(thisProjection.transform, geom) for geom in thisGeom]
        else:
            return transform(thisProjection.transform, thisGeom).buffer(0) if thisGeom.geom_type in bufferedTypes else transform(thisProjection.transform, thisGeom)
    else:
        print("  -- You need to specify either a projection or the CRSs")
        return None


####======================================================================
####====================== LOADING AND SAVING FILES ======================
####======================================================================

def inferDataType(fileName):
    dataType = None
    dataType = 'hexData' if 'hexData' in fileName else dataType
    dataType = 'chomeData' if 'chomeData' in fileName else dataType
    dataType = 'networkData' if 'networkData' in fileName else dataType
    return dataType

def getMergeKey(dataType):
    mergeKey = None
    mergeKey = 'hexNum' if 'hexData' == dataType else mergeKey
    mergeKey = 'addressCode' if 'chomeData' == dataType else mergeKey
    mergeKey = 'nodeNum' if 'networkData' == dataType else mergeKey
    return mergeKey

# def getDtypes(dataType=None, dTypeFile = '../Data/DataMasters/variableDTypeDict.pkl'):
#     variableDTypeDict = readPickleFile(dTypeFile)
#     validDataTypes = list(set([k for k,v in variableDTypeDict.items()]))
# #    print("validDataTypes:",validDataTypes)
#     if dataType in validDataTypes:
# #        print(variableDTypeDict[dataType])
#         return variableDTypeDict[dataType]
#     else:
#         return None


#    if dataType == 'hexData':
#        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str', 'totalPopulation': 'float', 'numHouseholds': 'float', 'pop_Total_A': 'float', 'pop_0-4yr_A': 'float', 'pop_5-9yr_A': 'float', 'pop_10-14yr_A': 'float', 'pop_15-19yr_A': 'float', 'pop_20-24yr_A': 'float', 'pop_25-29yr_A': 'float', 'pop_30-34yr_A': 'float', 'pop_35-39yr_A': 'float', 'pop_40-44yr_A': 'float', 'pop_45-49yr_A': 'float', 'pop_50-54yr_A': 'float', 'pop_55-59yr_A': 'float', 'pop_60-64yr_A': 'float', 'pop_65-69yr_A': 'float', 'pop_70-74yr_A': 'float', 'pop_75-79yr_A': 'float', 'pop_80-84yr_A': 'float', 'pop_85-89yr_A': 'float', 'pop_90-94yr_A': 'float', 'pop_95-99yr_A': 'float', 'pop_100yr+_A': 'float', 'pop_AgeUnknown_A': 'float', 'pop_AverageAge_A': 'float', 'pop_15yrOrLess_A': 'float', 'pop_15-64yr_A': 'float', 'pop_65yr+_A': 'float', 'pop_75yr+_A': 'float', 'pop_85yr+_A': 'float', 'pop_Foreigner_A': 'float', 'pop_0-19yr_A': 'float', 'pop_20-69yr_A': 'float', 'pop_70yr+_A': 'float', 'pop_20-29yr_A': 'float', 'pop_30-44yr_A': 'float', 'pop_percentForeigners': 'float', 'pop_percentChildren': 'float', 'pop_percentMale': 'float', 'pop_percentFemale': 'float', 'pop_percent30-44yr': 'float'}
#    elif dataType == 'chomeData':
#        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str', 'totalPopulation': 'int', 'numHouseholds': 'int', 'pop_Total_A': 'int', 'pop_0-4yr_A': 'int', 'pop_5-9yr_A': 'int', 'pop_10-14yr_A': 'int', 'pop_15-19yr_A': 'int', 'pop_20-24yr_A': 'int', 'pop_25-29yr_A': 'int', 'pop_30-34yr_A': 'int', 'pop_35-39yr_A': 'int', 'pop_40-44yr_A': 'int', 'pop_45-49yr_A': 'int', 'pop_50-54yr_A': 'int', 'pop_55-59yr_A': 'int', 'pop_60-64yr_A': 'int', 'pop_65-69yr_A': 'int', 'pop_70-74yr_A': 'int', 'pop_75-79yr_A': 'int', 'pop_80-84yr_A': 'int', 'pop_85-89yr_A': 'int', 'pop_90-94yr_A': 'int', 'pop_95-99yr_A': 'int', 'pop_100yr+_A': 'int', 'pop_AgeUnknown_A': 'int', 'pop_AverageAge_A': 'float', 'pop_15yrOrLess_A': 'int', 'pop_15-64yr_A': 'int', 'pop_65yr+_A': 'int', 'pop_75yr+_A': 'int', 'pop_85yr+_A': 'int', 'pop_Foreigner_A': 'int', 'pop_0-19yr_A': 'int', 'pop_20-69yr_A': 'int', 'pop_70yr+_A': 'int', 'pop_20-29yr_A': 'int', 'pop_30-44yr_A': 'int', 'pop_percentForeigners': 'float', 'pop_percentChildren': 'float', 'pop_percentMale': 'float', 'pop_percentFemale': 'float', 'pop_percent30-44yr': 'float'}
#    elif dataType == 'networkData':
#        return {'modality': 'str'}
#    else:
#        return None

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def loadOtherGeoms(thisData, geomCols=[]):

    for thisCol in geomCols:
        if thisCol in list(thisData):    ###=== support for csv data with special shapely objects
            try:
                thisData[thisCol] = thisData[thisCol].apply(wkt.loads)
            except:
                thisData[thisCol] = thisData[thisCol].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'geomDist' in list(thisData):    ###=== support for csv data with geomDist shapely objects
        try:
            thisData['geomDist'] = thisData['geomDist'].apply(wkt.loads)
        except:
            thisData['geomDist'] = thisData['geomDist'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'geomMap' in list(thisData):    ###=== support for csv data with geomMap shapely objects
        try:
            thisData['geomMap'] = thisData['geomMap'].apply(wkt.loads)
        except:
            thisData['geomMap'] = thisData['geomMap'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'geomAngle' in list(thisData):    ###=== support for csv data with geomMap shapely objects
        print("== Data contains geomAngle column, converted to geomMap")
        try:
            thisData['geomMap'] = thisData['geomAngle'].apply(wkt.loads)
            thisData.drop('geomAngle', axis=1, inplace=True)
        except:
            thisData['geomMap'] = thisData['geomAngle'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)
            thisData.drop('geomAngle', axis=1, inplace=True)
        # print("  -- new header:", list(thisData))

    if 'point' in list(thisData):    ###=== support for csv data with geomDist shapely objects
        try:
            thisData['point'] = thisData['point'].apply(wkt.loads)
        except:
            thisData['point'] = thisData['point'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'pointMap' in list(thisData):    ###=== support for csv data with geomDist shapely objects
        try:
            thisData['pointMap'] = thisData['pointMap'].apply(wkt.loads)
        except:
            thisData['pointMap'] = thisData['pointMap'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'centroid' in list(thisData):    ###=== support for csv data with geomDist shapely objects
        try:
            thisData['centroid'] = thisData['centroid'].apply(wkt.loads)
        except:
            thisData['centroid'] = thisData['centroid'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    if 'lineGeom' in list(thisData):    ###=== support for csv data with geomDist shapely objects
        try:
            thisData['lineGeom'] = thisData['lineGeom'].apply(wkt.loads)
        except:
            thisData['lineGeom'] = thisData['lineGeom'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else val)

    return thisData



def convertToGeoPandas(thisData, fromCRS=standardCRS, toCRS=None):
    thisData = gp.GeoDataFrame(thisData)
    if 'geometry' not in list(thisData):   ###== if there is no geometry column, try to create it
        if (('lat' in list(thisData)) & ('lon' in list(thisData))):
            thisData['geometry'] = thisData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
            thisData.set_geometry("geometry", inplace=True)
        elif (('latitude' in list()) & ('longitude' in list(thisData))):
            thisData['geometry'] = thisData.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
            thisData.set_geometry("geometry", inplace=True)
        else:
            print("  -- Data has no geometry")
            return thisData
    else:
        try:
            thisData['geometry'] = thisData['geometry'].apply(wkt.loads)
            thisData.set_geometry("geometry", inplace=True)
        except:
            # thisData['geometry'] = thisData.apply(lambda row: wkt.loads(row.geometry) if isinstance(row.geometry, str) else False, axis=1)
            thisData['geometry'] = thisData['geometry'].apply(lambda val: wkt.loads(val) if isinstance(val, str) else False)
            thisData.set_geometry("geometry", inplace=True)
    thisData.crs = fromCRS                     ##== 4326 corresponds to "naive geometries", normal lat/lon values
    if toCRS != None:
        thisData = thisData.to_crs(toCRS)

    thisData = loadOtherGeoms(thisData)

    gc.collect()
    return thisData

def readJSON(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        jsonData = f.read()
    return eval(jsonData)

def readJSONGraph(filename):
    with open(filename, encoding='utf-8-sig') as f:
        js_graph = json.load(f)
    return nx.json_graph.node_link_graph(js_graph, directed=False, multigraph=False)

def readJSONDiGraph(filename):
    with open(filename, encoding='utf-8-sig') as f:
        js_graph = json.load(f)
    return nx.json_graph.node_link_graph(js_graph, directed=True, multigraph=False)

def writeJSONGraph(graphData, filePathName):
    with codecs.open(filePathName, 'w', encoding="utf-8-sig") as jsonFile:
        jsonFile.write(json.dumps(nx.json_graph.node_link_data(graphData), cls = MyEncoder))

def writeJSONFile(dictData, filePathName):
    with codecs.open(filePathName, 'w', encoding="utf-8-sig") as jsonFile:
        jsonFile.write(json.dumps(dictData, cls = MyEncoder))

###=== Keep geometry data in the NetworkX graph, and save as geoJSON...not yet tested or used
def writeGeoJSON(data, fileName):
    data.to_file(fileName, driver="GeoJSON")


def writePickleFile(theData,filePathName):
    with open(filePathName, 'wb') as fp:
        pickle.dump(theData, fp)

def readPickleFile(filePathName):
    with open (filePathName, 'rb') as fp:
        return pd.read_pickle(fp)
        # return pickle.load(fp)


def readGeoPickleAndFix(filePathName, toCRS=None):
    '''
        Since we are clever and save .pkl and .csv, we can attempt to catch the error and then read the csv in place instead.
        The error happens to be TypeError: __cinit()__ takes at least 2 positional arguments (0 given)
    '''
    try:
        with open (filePathName, 'rb') as fp:
            thisData = pickle.load(fp)
            print(f'Loaded pickled data {fp}')
            thisData.crs = standardCRS  ##== 4326 corresponds to "naive geometries", normal lat/lon values
            if toCRS != None:
                thisData = thisData.to_crs(toCRS)
            return thisData
    except:
        print(f'Pickle load failed, trying to load csv {filePathName[:-4]+".csv"}')
        with open (filePathName[:-4]+".csv", 'rb') as fp:
            thisData = pd.read_csv(fp)
            print(f'Loaded csv data {fp}')
            thisData.crs = standardCRS  ##== 4326 corresponds to "naive geometries", normal lat/lon values
            if toCRS != None:
                thisData = thisData.to_crs(toCRS)
            return thisData

def readGeoPickle(filePathName, toCRS=None):
    with open (filePathName, 'rb') as fp:
        # thisData = pickle.load(fp)
        thisData = pd.read_pickle(fp)
        thisData.crs = standardCRS  ##== 4326 corresponds to "naive geometries", normal lat/lon values
        if toCRS != None:
            thisData = thisData.to_crs(toCRS)
        return thisData

def readCSV(fileName, useCols=None, fillNaN='', theEncoding='utf-8', dtypes=None, theColNames=None):
    useCols = [useCols] if isinstance(useCols, str) else useCols  ##-- Support entering a single text field as useCols
    theHeader = None if theColNames != None else 'infer'
    dataType = inferDataType(fileName)
    # dtypes = dtypes if dtypes != None else getDtypes(dataType)  ##-- Set the dTypes for whatever data, get the Dtypes for master data automatically (above)
    try:
        return pd.read_csv(fileName, encoding=theEncoding, usecols=useCols, dtype=dtypes, names=theColNames, header=theHeader).fillna(fillNaN)
    except:
        return pd.read_csv(fileName, encoding='shift-jis', usecols=useCols, dtype=dtypes, names=theColNames, header=theHeader).fillna(fillNaN)

def writeCSV(data, fileName):
    data.to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)

def writeGeoCSV(data, fileName):
    outputData = pd.DataFrame(data, copy=True)
    geoms = data.geometry.apply(wkt.dumps)
    outputData['geometry'] = geoms
    outputData.to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)
    # pd.DataFrame(data.assign(geometry=data.geometry.apply(wkt.dumps))).to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)


def readGeoPandasCSV(fileName, fromCRS=standardCRS, toCRS=None, useCols=None, fillNaN='', theEncoding='utf-8', dtypes=None):
    return convertToGeoPandas(readCSV(fileName, useCols=None, fillNaN=fillNaN, theEncoding=theEncoding, dtypes=dtypes), fromCRS=fromCRS, toCRS=toCRS)


def writeFeatherFile(theData,filePathName):
    theData.to_feather(filePathName);

def readFeatherFile(filePathName):
    return pd.read_feather(filePathName, columns=None, use_threads=True)


###=== Networks are stored as node and edge dataframes in the database, so rebuild the entwork from them.
def networkFromDataframes(nodesDF, edgesDF, source='source', target='target', nodeID='id', graphType=nx.DiGraph):
    edgesDF[source] = edgesDF[source].map(str)  ##-- the fucking database converted these back, again!
    edgesDF[target] = edgesDF[target].map(str)
    nodesDF[nodeID] = nodesDF[nodeID].map(str)
    # edgesDF = edgesDF.drop_duplicates(subset=[source,target])  ##-- not necessary because duplicates are dropped from the DB
    # nodesDF = nodesDF.drop_duplicates(subset=[nodeID])
    theNetwork = nx.from_pandas_edgelist(edgesDF, source=source, target=target, edge_attr=True, create_using=graphType)
    node_attr = nodesDF.set_index(nodeID).to_dict('index')
    # print(node_attr)
    nx.set_node_attributes(theNetwork, node_attr)
    return theNetwork
###-----------------------------------



####==== Write Pandas CSV File to S3
#import s3fs
#def writePandasToCSV(theData,theFilename,theBucket = 'geodata-processing'):
#    s3 = s3fs.S3FileSystem(anon=False)
#    with s3.open(theBucket+'/'+theFilename+'.csv','w') as f:
#        theData.to_csv(f)


# ####======================================================================
# ####====================== DATA HELPER FUNCTIONS =========================
# ####======================================================================
# ###=== Load and preprocess the grid file to ensure it has the needed info.
# ###=== Minimally it needs a geometry column and uniqueID to use in the tile name

# def loadGridData(filename, gridIDVarName):
#     if '.csv' in filename:
#         gridData = readGeoPandasCSV(filename)
#     elif '.pkl' in filename:
#         gridData = readGeoPickle(filename)
#     else:
#         print("  -- Unknown grid file type")
#         return None

#     if 'geomDist' not in gridData.columns.values.tolist():
#         gridData['geomDist'] = gridData['geometry'].apply(lambda thisGeom: convertGeomCRS(thisGeom, standardToDistProj))

#     if 'xMin' not in gridData.columns.values.tolist():
#         gridData['xMin'],gridData['yMin'],gridData['xMax'],gridData['yMax'] = gridData['geometry'].bounds

#     # gridData = gridData[[gridIDVarName, 'geometry', 'geomDist', 'xMin', 'yMin', 'xMax', 'yMax']]  ##--keep only needed columns
#     return (gridData, gridIDVarName)


# ####==== In the old network, the hexes were nodes, and this gets the hexData from the network's hex nodes as a geodataframe.
# def loadHexDataFromNetwork(thisNetwork, toCRS=None):
#     ###=== Isolate the hex nodes
#     thisNetwork = thisNetwork.subgraph([node for node,attr in thisNetwork.nodes(data=True) if attr['modality']=="hex"])
#     ###=== convert node properties to pandas dataframe
#     hexData = pd.DataFrame.from_dict(dict(thisNetwork.nodes(data=True)), orient='index')
#     ###=== Convert pandas geometry data into actual geodata
#     hexData = convertToGeoPandas(hexData, toCRS=toCRS)
#     #xMin, yMin, xMax, yMax = hexData['geometry'].total_bounds
# #    print()
#     gc.collect()
#     return hexData

# ####==== Read the JSON of the transportation network and extract the hexes with data as geoPandas
# def loadHexDataFromNetworkFile(filename, toCRS=None):
#     loadHexDataFromNetwork(readJSONDiGraph(filename), toCRS=toCRS)

# ####========= Master Data Helper Functions ========
# def getTopicFile(thisTopic, dataType='hexData'):
#     return '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv'

# ###--------------
# def loadCoreData(dataType='hexData', fillNaN=None, toCRS=None):
#     try:
#         coreData = readGeoPickle('../Data/DataMasters/'+dataType+'-Core.pkl', toCRS)
#     except:
#         if fillNaN != None:
#             coreData = convertToGeoPandas(pd.read_csv(getTopicFile('Core', dataType), encoding='utf-8', dtype=getDtypes(dataType)).fillna(fillNaN), toCRS=toCRS)
#         else:
#             coreData = convertToGeoPandas(pd.read_csv(getTopicFile('Core', dataType), encoding='utf-8', dtype=getDtypes(dataType)), toCRS=toCRS)
#     return coreData


# ###=== Read in a master data file, core or otherwise.
# def readMasterCSV(fileName, useCols=None, toCRS=None, fillNaN=None):
#     dataType = inferDataType(fileName)
#     if "Core" in fileName:
#         return loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)
#     else:
#         if fillNaN != None:
#             return pd.read_csv(fileName, encoding='utf-8', usecols=useCols, dtype=getDtypes(dataType)).fillna(fillNaN)
#         else:
#             return pd.read_csv(fileName, encoding='utf-8', usecols=useCols, dtype=getDtypes(dataType))

# ####==== returns the topic of a variable
# def getVariableDict(dataType='hexData'):
#     return readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')[dataType]

# def getVariableTopic(thisVariable, dataType='hexData'):
#     variableLocatorDict = getVariableDict(dataType)
#     thisTopic = variableLocatorDict[thisVariable] if thisVariable in list(variableLocatorDict.keys()) else None
#     return thisTopic

# def getVariableFilename(thisVariable, dataType='hexData'):
#     thisTopic = getVariableTopic(thisVariable, dataType)
#     if thisTopic == None:
#         print("Variable '"+thisVariable+"' not found in master data")
#         return None
#     else:
#         return getTopicFile(thisTopic, dataType)

# def getVariableList(dataType='hexData'):
#     variableLocatorDict = getVariableDict(dataType)
#     return list(variableLocatorDict.keys())

# def getVariablesByTopic(dataType='hexData'):
#     variableLocatorDict = getVariableDict(dataType)
#     variablesByTopic = {thisTopic:[k for k,v in variableLocatorDict.items() if v == thisTopic] for thisTopic in list(set(variableLocatorDict.values()))}
#     return variablesByTopic

# def getTopicList(dataType='hexData'):
#     return list(set(getVariableDict(dataType).values()))

# def getVariablesForTopic(thisTopic, dataType='hexData'):
#     variableLocatorDict = getVariableDict(dataType)
#     return [k for k,v in variableLocatorDict.items() if v == thisTopic]


# ####==== For a provided variable or list of variables, return a geopandas dataframe of the core data plus selected columns
# def getDataForVariables(thisVarList, dataType="hexData", fillNaN=None, toCRS=None):
#     ###=== Support entering of single variable name instead of a list
#     thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
#     ###=== Remove variables that are not in this master data
#     thisVarList = [thisVar for thisVar in thisVarList if getVariableTopic(thisVar, dataType) is not None]
#     ###=== First, get a list of topics (beyond the core data) needed for the variables listed
#     theTopicList = list(set([getVariableTopic(thisVar, dataType) for thisVar in thisVarList]))
# #    print("theTopicList:", theTopicList)
#     if "Core" in theTopicList:
#         theTopicList.remove('Core')
#     ###=== Next Load and merge the files acording to the appropriate indexer
#     combinedData = loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)
# #    print(combinedData.head(5))

#     if len(theTopicList) > 0:
#         mergeKey = getMergeKey(dataType)
#         for thisTopic in theTopicList:
# #            try:
# #               thisData = readPickleFile('../Data/DataMasters/'+dataType+'-'+thisTopic+'.pkl')
# #            except:
#             theseVars = [mergeKey] + [var for var in thisVarList if var in getVariablesForTopic(thisTopic, dataType)]
#             # print("thisTopic:",thisTopic)
#             # print("theseVars:",theseVars)
#             thisData = readMasterCSV(getTopicFile(thisTopic, dataType), useCols=theseVars, toCRS=toCRS, fillNaN=fillNaN)
# #            thisData = readCSV('../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv', fillNaN=fillNaN, useCols=theseVars)
# #            reportRunTime(startTime)
#             combinedData = pd.merge(combinedData, thisData, how='left', on=mergeKey)
# #    combinedData = combinedData[variablesToKeep]
#     return combinedData

# ####==== After adding a new dataset to the core data, extract the new data and add it to the appropriate topic.
# def addDataToTopic(thisData, thisTopic, theseVars=None, theseDTypes=None, dataType='hexData'):
#     print("==== Adding Data to Topic File ====")
#     mergeKey = getMergeKey(dataType)
#     if theseVars != None:  ##-- reduce the data to only those columns being used + the mergeKey
#         theseVars = [theseVars] if isinstance(theseVars, str) else theseVars  ## Support a single variable not ina list
#         ###--- if only a single string of dTypes is entered, assign that type to all
#         theseDTypes = [theseDTypes] * len(theseVars) if isinstance(theseDTypes, str) else theseDTypes
#         thisData = thisData[[mergeKey] + theseVars]
#         ###=== Test that the DTypes list matches the variable list
#         if isinstance(theseDTypes, list):
#             if len(theseDTypes) != len(theseVars):
# #                print("  -- Enter the correct DTypes list")
#                 raise ValueError("Enter a Dtype for each variable, or a single string of the Dtype to use for all.")

#     else:  ##-- if no variables are specified, then do all of them except the core variables...should not really be used.
#         coreVariables = getVariablesForTopic("Core", dataType)  ## get a list of core variables to remove
#         theseVars = [thisVar for thisVar in list(thisData) if thisVar not in coreVariables]  ## keep vars not in the core variables
#         thisData = thisData[[mergeKey] + theseVars]

#     ###=== If the topic already exists, add the data to that dataset, otherwise create a new topic file
#     if thisTopic in getTopicList(dataType):
#         thisTopicData = readMasterCSV(getTopicFile(thisTopic, dataType))  ## get the existing topic data
#         ###--- Check if the data column already exists...if so, overwrite it instead of creating _x and _y columns through merge
#         for thisVar in theseVars:
#             if thisVar in list(thisTopicData):
#                 thisTopicData.drop(thisVar, axis=1, inplace=True)  ##-- Overwriting is achieved by deleting the old data column
#         thisTopicData = pd.merge(thisTopicData, thisData, how='left', on=mergeKey)
#     else:
#         thisTopicData = thisData
#     writeCSV(thisTopicData, '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv')
#     ###=== Either way, one has to add the new variables and topic to the data catalog dictionary
#     variableLocatorDict = readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')
#     variableDTypeDict = readPickleFile('../Data/DataMasters/variableDTypeDict.pkl')
#     for index,thisVar in enumerate(theseVars):
#         variableLocatorDict[dataType][thisVar] = thisTopic
#         variableDTypeDict[dataType][thisVar] = theseDTypes[index] ##== Also add the new variables' dType to that lookup dict
#     writePickleFile(variableLocatorDict, '../Data/DataMasters/variableLocatorDict.pkl')
#     writePickleFile(variableDTypeDict, '../Data/DataMasters/variableDTypeDict.pkl')


# ####==== Remove the chosen variables from the topic files containing them for the chosen dataType
# def removeVariables(thisVarList, dataType='hexData', thisTopic=None):
#     print("==== Removing Variable(s) from Topic File ====")
#     variableLocatorDict = readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')
#     variableDTypeDict = readPickleFile('../Data/DataMasters/variableDTypeDict.pkl')

#     ###=== Support entering of single variable name instead of a list
#     thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
#     ###=== This is slow if there are multiple variables per topic, but easier to do this one variable at a time
#     for thisVar in thisVarList:
#         if ((thisVar != 'hexNum') & (thisVar != 'addressCode')):
#             ###=== Delete from a specific topic file or from whereever the locatorDict says that variables is located
#             thisTopic = thisTopic if thisTopic != None else getVariableTopic(thisVar, dataType)
#             thisData = readMasterCSV(getTopicFile(thisTopic, dataType))
#             thisData.drop(columns=[thisVar], inplace=True)
#             writeCSV(thisData, '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv')

#             ###=== Also remove this variable from the lookup dictionaries
#             del variableLocatorDict[dataType][thisVar]
#             del variableDTypeDict[dataType][thisVar]
#             writePickleFile(variableLocatorDict, '../Data/DataMasters/variableLocatorDict.pkl')
#             writePickleFile(variableDTypeDict, '../Data/DataMasters/variableDTypeDict.pkl')





# ####==== For a chosen topic, get the core data and that data together.
# def getDataForTopic(thisTopic, dataType='hexData', fillNaN=None, toCRS=None):
#     mergeKey = getMergeKey(dataType)
#     coreData = loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)
#     if thisTopic == 'Core':
#         return coreData
#     else:
#         thisData = readMasterCSV(getTopicFile(thisTopic, dataType), toCRS=toCRS, fillNaN=fillNaN)
#         return pd.merge(coreData, thisData, how='left', on=mergeKey)


# ###=== Get the hexNum for the hex that includes the input lat/lon
# # def getHexNumForPoint(pointGeom, theHexData):
# #     vals = theHexData[theHexData['geometry'].intersects(pointGeom)]['hexNum'].values
# #     if len(vals) > 0:
# #         return vals[0]
# #     else:  # Shapely intersecet can work inproperly on floating number so round lon/lat
# #         x_round, y_round = np.round(pointGeom.x, 5), np.round(pointGeom.y, 5)
# #         vals = theHexData[theHexData['geometry'].intersects(Point(x_round, y_round))]['hexNum'].values
# #         if len(vals) > 0:
# #             return vals[0]
# #         else:
# #             x_round, y_round = np.round(pointGeom.x, 4), np.round(pointGeom.y, 4)
# #             vals = theHexData[theHexData['geometry'].intersects(Point(x_round, y_round))]['hexNum'].values
# #             if len(vals) > 0:
# #                 return vals[0]
# #             else:
# #                 return None

# ###=== Get the gridID (e.g. hexNum) for the grid/hex that includes the input lat/lon
# def getGridIDForPoint(pointGeom, theGridData, gridID="hexNum"):
#     vals = theGridData[theGridData['geometry'].intersects(pointGeom)][gridID].values
#     if len(vals) > 0:
#         return vals[0]  ##-- if at least one is found, return the first one (should be only one)
#     else:  # Shapely intersects can work inproperly on floating number so try again with rounded lon/lat values
#         x_round, y_round = np.round(pointGeom.x, 5), np.round(pointGeom.y, 5)
#         vals = theGridData[theGridData['geometry'].intersects(Point(x_round, y_round))][gridID].values
#         if len(vals) > 0:
#             return vals[0]
#         else:  ##-- If rounding to 5 places doesn't work, try 4 places.
#             x_round, y_round = np.round(pointGeom.x, 4), np.round(pointGeom.y, 4)
#             vals = theGridData[theGridData['geometry'].intersects(Point(x_round, y_round))][gridID].values
#             if len(vals) > 0:
#                 return vals[0]
#             else:
#                 return None  ##-- It will quietly return None if no intersecting grid is found.


# ###=== Get the hexNums for all the hexes that intersect the input geometry (e.g. a chome polygon)
# def getHexNumsForGeom(someGeom, theHexData):
#     return list(theHexData[theHexData['geometry'].intersects(someGeom)]['hexNum'])

# ###=== Get the gridIDs for all the grids that intersect the input geometry (e.g. a chome polygon)
# ###=== Can also be used to get all the elevation Files that intersect a network tile with gridID="filename"
# def getGridIDsForGeom(someGeom, theGridData, gridID="hexNum"):
#     return list(theGridData[theGridData['geometry'].intersects(someGeom)][gridID])

# ####==== check the variable locator file for multiple occurences of the same variable and report the offenders
# ####=== *** This is unnecessary because it's a dictionary, and dicts can't have duplicate keys!
# #def checkVariableRedundancy(dataType='hexData'):
# #    variableList = getVariableList(dataType)
# #    duplicateVars = set(thisVar for thisVar in variableList if thisVar in seenVars or seenVars.add(thisVar))
# #    report duplicateVars



####=============================================================================
####==== It converts a dataframe with multipolygons into one with only polygons, copying rows
def explode(indata):
#    indf = gp.GeoDataFrame.from_file(indata)
    indf = indata
    outdf = gp.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gp.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf

####=========================== HELPER FUNCTIONS =========================

###=== shortCut for evaluating list and dict variables stored in pandas
def evalStr(thisStr):
    return literal_eval(thisStr)

###=== Convert a 2D list into a 1D list with duplicates removed
def flattenList(thisList):
    return list(set([item for sublist in thisList for item in sublist]))

###=== Create a(lookup) dictionary from two columns of a dataframe
def dictFromColumns(thisData, keyColumn, valueColumn):
    # return dict(zip(thisData[keyColumn], thisData[valueColumn]))
    return thisData.set_index(keyColumn).to_dict()[valueColumn]

####==== Take a non-nested dictionary, and sort the entries by the value, default is larger numbers first
def sortDictByValue(theDict, largerFirst=True):
    return {k: v for k, v in sorted(theDict.items(), key=lambda item: item[1], reverse=largerFirst)}

def sortDictByKey(theDict, largerFirst=True):
    return {k: v for k, v in sorted(theDict.items(), key=lambda item: item[0], reverse=largerFirst)}

###--- get a dictionary of item counts from a list of items
def getItemCounts(theList, outputDict=False, printOutput=True, largerFirst=True):
    itemCounts = sortDictByValue(Counter(theList), largerFirst)  ## this is a dict of item:count
    if printOutput:
        for thisItem,thisCount in itemCounts.items():
            print("  --",thisItem,":",thisCount)
    if outputDict:
        return itemCounts

###--- reduce a list to the unique items in the list, also, converts to list if it's some other collection
def uniqueItems(theList):
    if isinstance(theList, list):
        return list(set(theList))
    else:
        return list(set(list(theList)))

###=== remove items in one list from another list.  Using sets is faster than list comprehension, and/but also removes duplicates.
def removeItemsFromList(theList, listOfItemsToRemove):
    # [k for k in theList if k not in listOfItemsToRemove ]  < equivalent to this, but much faster
    return list(set(theList) - set(listOfItemsToRemove))


####--- Smarter handling of None and zero values when calculating percentages (esp of hex/chome values)
def percentValue(numerator, denominator):
    if ((numerator != None) & ((denominator != None))):
        if denominator != 0:
            return numerator / denominator
        else:
            return 0
    else:
        return None


###=== calculate the average percent change in values across a list
def meanPercentChange(valueList, timeList=None):
    valueList = [val for val in valueList if pd.isnull(val) == False]
    if len(valueList) > 1:
        if timeList == None:
            timeList = list(range(len(valueList)))
        changeList = []
        for index in range(len(valueList) - 1):
            changeList.append( (valueList[index + 1] - valueList[index]) / valueList[index] )
        return np.mean(changeList)
    elif len(valueList) == 1:
        return valueList[0]
    else:
        return np.nan

###=== check whether a variable is some kind of number (not NaN, str, bool, etc.)
def isNumber(var):
    if var is None:
        return False
    elif isinstance(var, numbers.Number):
        if np.isnan(var):
            return False
        else:
            return True
    else:
        return False


def isNan(var):
    if isinstance(var, numbers.Number):
        if np.isnan(var):
            return True
        else:
            return False
    else:
        return False


###=== convert a strong input to a number of the appropriate type
def strToNum(var):
    if isinstance(var, numbers.Number):
        if np.isnan(var):
            return np.nan
        else:
            return var
    elif isinstance(var, str):
        try:
            if "." in var:
                return float(var)
            else:
                return int(var)
        except:
            return np.nan
    else:
        return np.nan


def NanToNone(val):
    if val is None:
        return None  ##-- if already None, then return None
    elif isinstance(val, str):
        val = strToNum(val)  ##-- if it's a string, try to convert to a number (returns NaN if not a number)

    if isNan(val):
        return None  ##-- return None whenever it's not a numerical value
    else:
        return val  ##-- otherwise return the value

###=== Extract numeric data from a str and convert to appropriate number type (or leave as number)
def extractNumeric(val):
    if isNumber(val):
        return val
    elif isinstance(val,str):
        val = ''.join(re.findall(r"[-+]?\d*\.\d+|\d+", val))  ##-- extract the numerical elements as string
        return strToNum(val)
    else:
        print("!! Input is neither a number nor contains a numeric string.")
        return None

####--- Round a number to zero decimals and then convert to int instead of just cutting the decimal part.
def makeInt(someNumber):
    someNumber = float(someNumber) if isinstance(someNumber, str) else someNumber
    if isinstance(someNumber, numbers.Number):
        return int(np.round(someNumber, decimals=0))
    else:
        print("!! Input is neither a number nor a numeric string.")

####--- Shorthand for calling numpy's rounding function
def rnd(someNumber, decimals=3):
    if someNumber > 0:
        return np.round(someNumber, decimals=decimals) if someNumber > (1 / (10 ** (1 + decimals))) else 0
    else:
        return 0 - np.round(abs(someNumber), decimals=decimals) if abs(someNumber) > (1 / (10 ** (1 + decimals))) else 0

####--- apply rounding to coordinates (helps removefloating point problems in operations)
def roundCoords(geom, decimals=3):
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(np.array(geojson['coordinates']), decimals)
    return  shape(geojson)


###=== Check whether a string contains any of a list of characters or substrings
def containsAnyChar(val, matchList):
    if isinstance(val, str):
        if any(substr in val for substr in matchList):
            return True
        else:
            return False
    else:
        return False

###=== Remove either a single item or a list of items from a list.
###--- check whether the item is in the list to prevent errors
def removeFromList(theList, toRemove):
    toRemove = [toRemove] if not isinstance(toRemove, list) else toRemove
    for item in toRemove:
        if item in theList:
            theList.remove(item)
        # else:
        #     print("  -- Element", item, "not in list.")
    return theList


####---Distance in meters between two points
# def distanceBetweenLonLats(x1,y1,x2,y2):
#     return np.round(geopy.distance.distance(geopy.Point(y1,x1), geopy.Point(y2,x2)).m, decimals=0)

def euclideanDistance(px1, py1, px2, py2):
    return math.sqrt((px2-px1)**2 + (py2-py1)**2)

def euclideanPointDistance(pointGeom1, pointGeom2):
    px1,py1 = pointGeom1.coords[0]
    px2,py2 = pointGeom2.coords[0]
    return math.sqrt((px2-px1)**2 + (py2-py1)**2)

def distWithHeight(px1, py1, px2, py2, elevation1, elevation2):
    deltaElevation = abs(elevation1 - elevation2)
    euclideanDist = math.sqrt((px2-px1)**2 + (py2-py1)**2)
    return math.sqrt((deltaElevation)**2 + (euclideanDist)**2)

####==== Calculate the great circle distance in meters for two lat/lon points
def haversineDist(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    theAngle = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6367000 * 2 * math.asin(math.sqrt(theAngle))   ## distance in meters

def getXY(pt):
    return [pt.x, pt.y]


def reportRunTime(thisStartTime, preface=None):
    if preface==None:
        preface = "    -- runtime:"
    else:
        preface = str(preface) if not isinstance(preface,str) else preface
        preface = "    -- "+preface+":"
    newStartTime = time.time()
    if newStartTime - thisStartTime < 60:
        print(preface, np.round((newStartTime - thisStartTime), decimals=2),"seconds")
    elif newStartTime - thisStartTime < 3600:
        print(preface, np.round((newStartTime - thisStartTime)/60, decimals=1),"minutes")
    else:
        print(preface, np.floor((newStartTime - thisStartTime)/3600),"hour(s) and ", np.round(((newStartTime - thisStartTime) % 3600) / 60, decimals=0),"minutes")

# runStartTime = reportProgressTime(runStartTime)
def reportProgressTime(thisStartTime, preface=None):
    if preface==None:
        preface = "  -- runtime:"
    else:
        preface = str(preface) if not isinstance(preface,str) else preface
        preface = "  -- "+preface+":"
    newStartTime = time.time()
    if newStartTime - thisStartTime < 60:
        print(preface, np.round((newStartTime - thisStartTime), decimals=2),"seconds")
    elif newStartTime - thisStartTime < 3600:
        print(preface, np.round((newStartTime - thisStartTime)/60, decimals=1),"minutes")
    else:
        print(preface, np.floor((newStartTime - thisStartTime)/3600),"hour(s) and ", np.round((newStartTime - thisStartTime)/60 % 60, decimals=0),"minutes")
    return time.time()


#### Usage is runStartTime = printProgress(runStartTime,index,len(fromGeoms))
def printProgress(thisStartTime,index,totalNum):
    oneBlock = makeInt(totalNum / 100)  ## approximately how many are in 1%
    if oneBlock > 0:
        if (index % oneBlock == 0) and (index > 0):
            newStartTime = time.time()
            if newStartTime - thisStartTime < 60:
                print("  == Analyzed",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): ==>",rnd((newStartTime - thisStartTime),1),"seconds")
            else:
                print("  == Analyzed",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): ==>",rnd((newStartTime - thisStartTime)/60,2),"minutes")
            return newStartTime
        else:
            return thisStartTime
    ##== If there are fewer than 100 items to do, just report the time for each one.
    else:
        newStartTime = time.time()
        if newStartTime - thisStartTime < 60:
            print("  == Analyzed",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): ==>",rnd((newStartTime - thisStartTime),1),"seconds")
        else:
            print("  == Analyzed",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): ==>",rnd((newStartTime - thisStartTime)/60,2),"minutes")
        return newStartTime


###=== print just the time elapsed for some process
def getRunTime(thisStartTime):
    newStartTime = time.time()
    if newStartTime - thisStartTime < 60:
        return str(rnd((newStartTime - thisStartTime), decimals=2))+" seconds"
    elif newStartTime - thisStartTime < 3600:
        return str(rnd((newStartTime - thisStartTime)/60, decimals=2))+" minutes"
    else:
        return str(np.floor((newStartTime - thisStartTime)/3600))+" hour(s) and "+str(makeInt(((newStartTime - thisStartTime) % 3600) / 60))+" minutes"



def makeNumberString(number, length=1):
    return str(int(float(number))).zfill(length)


###=== When loading a geoCSV, only the 'geometry' column is converted into Shapely geoms, so this converts other columns
def wktLoadOtherGeoms(theData,geomVarList):
    if not isinstance(geomVarList,list):
        geomVarList = [geomVarList]
    for thisGeomVar in geomVarList:
        try:
            theData[thisGeomVar] = theData[thisGeomVar].apply(wkt.loads)
        except:
            theData[thisGeomVar] = theData[thisGeomVar].apply(lambda row: wkt.loads(row) if isinstance(row, str) else False)
    return theData


###=== append a line to an existing dataframe, not that DF.append(newLine) has been depricated (even though it's useful and needed)
def appendDictToDF(origDF, dictToAppend):
    return pd.concat([origDF, pd.DataFrame.from_records([dictToAppend])], ignore_index=True)



####=============================================================================
####========================== INPTEROLATION ALGORITHMS =========================
####=============================================================================


####=============================================================================
####=== Isolate the geometry information and create the geometry objects with indices
def createGeomData(thisData, theGeomCol='geometry'):
    ###=== Determine if the dataset is chome, and reduce it to build geometries only from the lowest level
    if 'lowestLevel' in list(thisData):
        thisData = thisData[thisData['lowestLevel'] == True][[theGeomCol]]
    else:
        thisData = thisData[[theGeomCol]]

    ###=== Add index values as geometry attributes, then Build an R-tree of the hexes
    Geoms = list(thisData[theGeomCol])
    GeomsIdx = list(thisData.index)
    for index, geom in enumerate(Geoms):
        geom.idx = GeomsIdx[index]   ##-- store the original index value in the geom to reference the data cell later

    return Geoms


####===============================================================================
def guessCalculation(thisVar):
    if 'min' in thisVar.lower():
        return 'min'
    elif 'max' in thisVar.lower():
        return 'max'
    elif 'mean' in thisVar.lower():
        return 'mean'
    elif 'median' in thisVar.lower():
        return 'mean'
    elif 'percent' in thisVar.lower():
        return 'percent'
    else:
        return 'sum'


####===============================================================================
def aggregationCalculations(theseRows, theAdminLevel, theData, thisVarList, thisVarDict):
    for thisRow in theseRows:
        ###=== Filter to the data rows relevant to aggregating at this admin level
        if theAdminLevel == 2:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['cityCode'] == theData.at[thisRow,'cityCode']) & (theData['oazaCode'] == theData.at[thisRow,'oazaCode']) & (theData['adminLevel'] == 3)]
        elif theAdminLevel == 1:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['cityCode'] == theData.at[thisRow,'cityCode']) & (theData['adminLevel'] == 2)]
        else:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['adminLevel'] == 1)]

        for thisVar in thisVarList:
            ###=== Only process rows where the variable of interest exists.
            thisRelevantData = relevantData[((relevantData[thisVar] != '') & (relevantData[thisVar].notnull()))]
            if len(thisRelevantData) > 0:
                if thisVarDict[thisVar] == 'sum':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].sum()
                elif thisVarDict[thisVar] == 'binary':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].any()
                elif thisVarDict[thisVar] == 'min':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].min()
                elif thisVarDict[thisVar] == 'max':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].max()
                elif ((thisVarDict[thisVar] == 'mean') | (thisVarDict[thisVar] == 'median') | (thisVarDict[thisVar] == 'percent')):
#                    print("value", list(thisRelevantData[thisVar]), "    area", list(thisRelevantData['landArea']))
                    totalAreaInvolved = sum([value * area for (value,area) in zip(list(thisRelevantData[thisVar]),list(thisRelevantData['landArea']))])
                    theData.at[thisRow,thisVar] = percentValue(totalAreaInvolved, theData.at[thisRow,'landArea'])
                else: ##sum by default
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].sum()
#            else:   ###=== if there is no relevant data, then leave it as None
    return theData


####=============================================================================
####==== Take binary variables at the lowest level (chome) and aggregate them up to all higher levels
####==== Currently, we make the variable true if any of the lower-level contained areas are true
def aggregateUp(theData, thisVarList=None, thisVarDict=None):

    ###=== Provide support for specifiying a single variable by converting it into a list here
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    if thisVarList == None:
        thisVarList = [k for k,v in thisVarDict.items()]

    ###=== If the addressCode parts are not in this data, then create them from the full AddressCOde (removed at the end)
    createAddressCodes = False
    if 'prefCode' not in list(theData):
        createAddressCodes = True
        theData['prefCode'] = theData.apply(lambda row: row['addressCode'][0:2], axis=1)
        theData['cityCode'] = theData.apply(lambda row: row['addressCode'][2:5], axis=1)
        theData['oazaCode'] = theData.apply(lambda row: row['addressCode'][5:9], axis=1)
#    print(theData.at[0,'addressCode'],"=",theData.at[0,'prefCode'],"+",theData.at[0,'cityCode'],"+",theData.at[0,'oazaCode'],"+ chomeCode")

    ###=== if the landArea column is not included, get it from the geometry whenever it is needed for a calculation (probably slow)
    createLandArea = False
    listOfCalculationTypes = [v for k,v in thisVarDict.items()]
    if any(x in listOfCalculationTypes for x in ['mean','median','percent']):
        if 'landArea' not in list(theData):
            createLandArea = True
            theData['landArea'] = theData.apply(lambda row: row['geometry'].area, axis=1)

    ####---- Start with Oaza that are not the lowest level
    nonLowestOazaRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 2)].index.values)
    theData = aggregationCalculations(nonLowestOazaRows, 2, theData, thisVarList, thisVarDict)
#    print(theData.head())

    cityRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 1)].index.values)
    theData = aggregationCalculations(cityRows, 1, theData, thisVarList, thisVarDict)
#    print(theData.head())

    prefRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 0)].index.values)
    theData = aggregationCalculations(prefRows, 0, theData, thisVarList, thisVarDict)
#    print(theData.head())

    ###=== Remove temporary variables
    if createAddressCodes == True:
        theData.drop(columns=['prefCode', 'cityCode', 'oazaCode'], inplace=True)
    if createLandArea == True:
        theData.drop(columns=['landArea'], inplace=True)

    return theData


####=============================================================================
####=== Take data using one set of polygons and interpolate it to a different set of polygons using overlap percents.
####=== For example, this can be used to convert between hex <=> chome, but also other "free polygon" data
####=== The varDict is a {varName:aggregationOperation} pair that should be used...it overwrites any thisVarList.
####=== If the varList is specifed instead of the varDict, this method will try to guess the operation based on varName, default is sum.
def interpolateGeoData(fromGeoData, toGeoData, thisVarDict=None, fromVarList=None, thisVarList=None, defaultZero=False, overwrite=True):

    startTime = time.time()
    dataType = 'hexData' if 'hexNum' in list(fromGeoData) else 'chomeData'

    if thisVarDict == None:
        if thisVarList == None:
            ## Convert all non-Core variables in fromGeoData,
            coreVariables = getVariablesForTopic("Core", dataType)  ## get a list of core variables to remove except the mergeKey
            thisVarList = [thisVar for thisVar in list(fromGeoData) if thisVar not in coreVariables]  ## keep vars not in the core variables

        ###=== Infer the interpolation calculcation from the variable names
        thisVarDict = {thisVar:guessCalculation(thisVar) for thisVar in thisVarList}
    else:
        if thisVarList == None:    ###=== if a conversion dictionary was specified, then get the varList from it
            thisVarList = [k for k,v in thisVarDict.items()]
        ###=== Else we are converting a subset of the dictionary, so keep the var list as is, but check if there is something missing from the dict
        else:
            missingDictVars = [thisVar for thisVar in thisVarList if thisVar not in [k for k,v in thisVarDict.items()] ]
            thisVarDict = {thisVar:guessCalculation(thisVar) for thisVar in missingDictVars}

    ###=== Provide support for specifiying a single variable by converting it into a list here
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    ###=== Seed the toGeoData with Nones for each variable being processed
    for thisVar in thisVarList:
        toGeoData[thisVar] = [None] * len(toGeoData)

    ###=== Support specifiying a single source variable by converting it into a list here
    fromVarList = [fromVarList] if isinstance(fromVarList, str) else fromVarList
    ###=== Ensure the length of the source list is the same as the thisVarList
    if fromVarList == None:
        fromVarList = thisVarList
    elif len(fromVarList) != len(thisVarList):
        if len(fromVarList) == 1:
            fromVarList = fromVarList * len(thisVarList)
        else:
            raise ValueError("You need to properly specify the source variables.")

    ###=== Set the initial CRSs in case they weren't set before
    fromGeoData.crs = standardCRS
    toGeoData.crs = standardCRS

    ###=== Many calculations require the area of the poygons and their overlaps, so we set them both to an equal-area projection
    # fromGeoData = fromGeoData.to_crs(areaCalcCRS)
    # toGeoData = toGeoData.to_crs(areaCalcCRS)

    ## Fix malformed (self-intersecting) polygons created by the change in CRS
    # fromGeoData.loc[:,'geometry'] = fromGeoData['geometry'].apply(lambda row: row.buffer(0))
    # toGeoData.loc[:,'geometry'] = toGeoData['geometry'].apply(lambda row: row.buffer(0))

    ###=== Create and use a geomArea column instead of converting the CRS of the dataframe
    standardToAreaProj = createStandardToAreaProj()
    fromGeoData['geomArea'] = fromGeoData['geometry'].apply(lambda row: convertGeomCRS(row, standardToAreaProj).buffer(0))
    toGeoData['geomArea'] = toGeoData['geometry'].apply(lambda row: convertGeomCRS(row, standardToAreaProj).buffer(0))

    ###=== Create indexed geometry object lists for both datasets for STRtree usage
    toGeoms = createGeomData(toGeoData, 'geomArea')
    fromGeoms = createGeomData(fromGeoData, 'geomArea')
    ## The tree is built from the fromData so the tree query returns Indices of the fromData that intersect each toData polygon
    geomTree = STRtree(fromGeoms)

    runStartTime = time.time()
    ###=== For each element being converted into...
    for index, thisToGeom in enumerate(toGeoms):
        runStartTime = printProgress(runStartTime,index,len(toGeoms))
        overlappingFromGeoms = geomTree.query(thisToGeom)
        for thisFromGeom in overlappingFromGeoms:
            if thisFromGeom.intersects(thisToGeom):
                if type(thisFromGeom) == Point:
                    thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
                    for varNum,thisVar in enumerate(thisVarList):
                        currentToValue = toGeoData.at[thisToGeom.idx, thisVar]
                        fromValue = thisFromGeoData[fromVarList[varNum]].values[0]
                        if fromValue != None:
                            if thisVarDict[thisVar] == 'sum':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else currentToValue + fromValue

                else:
                    ## get the proportion of overlap of the from data to calculate the value to add to this toGeom
                    sourceArea = thisFromGeom.area
                    targetArea = thisToGeom.area
                    overlapArea = thisFromGeom.intersection(thisToGeom).area
                    sourceOverlapProportion = (overlapArea / sourceArea)
                    targetOverlapProportion = (overlapArea / targetArea)
                    thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
                    for varNum,thisVar in enumerate(thisVarList):
                        currentToValue = toGeoData.at[thisToGeom.idx, thisVar]
                        fromValue = thisFromGeoData[fromVarList[varNum]].values[0]
                        if fromValue != None:
                            if thisVarDict[thisVar] == 'binary':
                                ##-- if the current value is either None or False, i.e. stay True if True, else set to the var
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue != True else currentToValue

                            if thisVarDict[thisVar] == 'sum':
                                toGeoData.at[thisToGeom.idx, thisVar] = (fromValue * sourceOverlapProportion) if currentToValue == None else currentToValue + (fromValue * sourceOverlapProportion)

                            if thisVarDict[thisVar] == 'max':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else max(fromValue, currentToValue)

                            if thisVarDict[thisVar] == 'min':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else min(fromValue, currentToValue)

                            if thisVarDict[thisVar] == 'mean':
                                toGeoData.at[thisToGeom.idx, thisVar] = (fromValue * targetOverlapProportion) if currentToValue == None else currentToValue + (fromValue * targetOverlapProportion)

                            if thisVarDict[thisVar] == 'percent':  ##== For simple overlap percent of boolean data...like land use.
                                toGeoData.at[thisToGeom.idx, thisVar] = targetOverlapProportion if currentToValue == None else currentToValue + targetOverlapProportion

                            if thisVarDict[thisVar] == 'proportion':  ##== For proportional values...like percent populations...use mean
                                toGeoData.at[thisToGeom.idx, thisVar] = (targetOverlapProportion * fromValue) if currentToValue == None else (currentToValue + (targetOverlapProportion * fromValue))

                            if thisVarDict[thisVar] == 'area':  ## there are no values, just sum the overlapping area (e.g. GreenData)
                                toGeoData.at[thisToGeom.idx, thisVar] = overlapArea if currentToValue == None else currentToValue + overlapArea

    ###=== In some cases (like Green area) we want the the places with no overlap to be zero instead of None, so correct before aggregating
    if defaultZero:
        for thisVar in thisVarList:
            toGeoData[thisVar] = toGeoData.apply(lambda row: 0 if row[thisVar] == None else row[thisVar], axis=1)

    ###=== If the toData is chome data, then only the lowest level has been filled in, so now aggregate up
    if 'addressCode' in list(toGeoData):
        ###--- Aggregate up from chome to their Oaza (using only Oaza that have chome..i.e., not lowest level oaza)
        toGeoData = aggregateUp(toGeoData, thisVarList, thisVarDict)

    toGeoData.drop(columns=['geomArea'], inplace=True)
    # toGeoData = toGeoData.to_crs(standardCRS)  ## Convert back to naive geometries before returning the data
    print("==== Completed Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
    return toGeoData


###=== The values are wonky where the population is small, so correct for small populations with a city-level leaning rate.
###=== Create Adjusted percentage values to unbias the low-population areas
def smallSampleAdjustedPercent(thisValue, totalValue, priorPercent, sampleThreshold = 50):
    if totalValue >= sampleThreshold:
        return thisValue / totalValue
    elif totalValue > 0:
        scalingFactor = totalValue / sampleThreshold  ## the proportion of the needed population attained by this population
        return ((thisValue / totalValue) * scalingFactor) + (priorPercent * (1 - scalingFactor))
    else:
        return priorPercent



####=============================================================================
####=== Take data using one set of polygons and interpolate it to a different set of polygons using overlap percents.
####=== For example, this can be used to convert between hex <=> chome, but also other "free polygon" data
#def interpolateAreaData(fromGeoData,toGeoData,thisVarList):
#
#    ###=== The land areas were computed using this CRS so they matched the values provided by the government.
#    ###=== Because we are calculating area and percent areas here, the idea is that we need to use the same projection
#    fromGeoData = fromGeoData.to_crs(areaCalcCRS)
#    toGeoData = toGeoData.to_crs(areaCalcCRS)
#
#    ###==== Check the noise polygons are all valid, ...some may be made invalid by changing the CRS, so fix them.
#    for thisIndex in fromGeoData.index.values:
#        if fromGeoData.at[thisIndex, 'geometry'].is_valid == False:
#            fromGeoData.at[thisIndex, 'geometry'] = fromGeoData.at[thisIndex, 'geometry'].buffer(0)
##            print("row:", index, "    geometry is valid:", fromGeoData.at[index, 'geometry'].is_valid )
#
##    print(toGeoData.at[0,'geometry'])
##    print("Area of 54,127m2 hexagon using cea crs is:",toGeoData.at[0,'geometry'].area)
#
#    ###=== Seed the toGeometry with Nones for each variable used from the fromGeometry
#    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
#    for thisVar in thisVarList:
#        toGeoData[thisVar] = [None] * len(toGeoData)
#    #    toGeometry[thisVar] = np.zeros(len(toGeometry)).tolist()
#
#    toGeoms = createGeomData(toGeoData)
#    fromGeoms = createGeomData(fromGeoData)
#    ## The tree is built from the fromData so the tree query returns Indices of the fromData that intersect each toData polygon
#    geomTree = STRtree(fromGeoms)
#
#    startTime = time.time()
#    runStartTime = time.time()
#    ###=== For each element being converted into...
#    for index, thisToGeom in enumerate(toGeoms):
#        runStartTime = printProgress(runStartTime,index,len(toGeoms))
#        overlappingFromGeoms = geomTree.query(thisToGeom)
#        for thisFromGeom in overlappingFromGeoms:
#            if thisFromGeom.intersects(thisToGeom):
#                ## get the overlap area and add it to this toGeom
#    #            overlapArea = thisFromGeom.intersection(thisToGeom).area
#                overlapProportion = (thisFromGeom.intersection(thisToGeom).area / thisToGeom.area)  ## the percent of the toGeom that is covered by this free geom
#                thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
#    #            print(overlapArea)
#                for thisVar in thisVarList:
#                    if toGeoData.at[thisToGeom.idx, thisVar] == None:
#                        toGeoData.at[thisToGeom.idx, thisVar] = thisFromGeoData[thisVar].values[0] * overlapProportion
#                    else:
#                        toGeoData.at[thisToGeom.idx, thisVar] += overlapProportion
#
#    #                toGeoData.at[thisToGeom.idx, thisVar] += overlapArea
#    #                if toGeoData.at[thisToGeom.idx,thisVar] > 54127:
#    #                    print("hexIndex:",thisToGeom.idx,"   var:",thisVar,"   value:", toGeoData.at[thisToGeom.idx,thisVar])
#
#    ###=== If the toData is chome data, then only the lowest level has been filled in, so now aggregate up
##    if 'lowestLevel' in list(toGeoData):
##        print("  -- Aggregating Variables Upwards")
##        ###--- Aggregate up from chome to their Oaza (using only Oaza that have chome..i.e., not lowest level oaza)
##        toGeoData = aggregateUpValues(toGeoData, thisVarList)
##        toGeoData = aggregateUpPercents(toGeoData, thisVarList)
##        ###--- Add a percent area variable for each area variable for the chome data
##        for thisVar in thisVarList:
##            toGeoData['percent'+thisVar] = toGeoData.apply(lambda row: percentValue(row[thisVar], row['landArea']), axis=1)
#
#    toGeoData = toGeoData.to_crs(standardCRS)  ## Convert back to naive geometries before returning the data
#    print("==== Completed Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
#    return toGeoData


####========================================================================
####================= GEOGRAPHIC DATA FUNCTIONS ============================
####========================================================================

###=== The rows of the source data in geomTree to keep based on overlapping the input polyGeoms.
def getRowsToKeep(polyGeoms, geomTree):
#    runStartTime = time.time()
    rowsToKeep = []
    for index,thisPolyGeom in enumerate(polyGeoms):
#        runStartTime = printProgress(runStartTime,index,len(polyGeoms))
        overlappingGeoms = geomTree.query(thisPolyGeom)   ### Returns the polygon indices that intersect this hex/chome
        if overlappingGeoms != []:
            ###--- this just means the bounding boxes overlap, so now check that they actually intersect
            reallyOverlap = False
            for thisGeom in overlappingGeoms:
                if thisPolyGeom.intersects(thisGeom):
                    reallyOverlap = True
            if reallyOverlap:
                rowsToKeep.append(thisGeom.idx)
    #print(rowsToKeep)
    return rowsToKeep

#####=============================================================================

###=== Get the rows of a dataframe that intersect a given polygon (not necessarily from the same dataframe)
def getDataForPolygon(thisPolygon, thisData):
    polyGeoms = createGeomData(thisData)
    geomTree = STRtree(polyGeoms)
    rowsToKeep = getRowsToKeep([thisPolygon], geomTree)
    return thisData[thisData.index.isin(rowsToKeep)]


####=== Return a list of indices from one dataset for all polygons that intersect some polygon in another dataset
####=== For example, this is used to isolate the green areas within TokyoMain to reduce the filesize to something manageable.
####=== Can also be used to get all the hexagons that overlap with some chome polygon.
def getRowsWithinArea(boundingGeometries, dataToBeBound):
    boundingGeoms = boundingGeometries if isinstance(boundingGeometries,list) else list(dataToBeBound['geometry'])
#    boundingValues = list(boundingData.index)
#    for index, geom in enumerate(boundingGeoms):
#        boundingGeoms[index].idx = boundingValues[index]  ##== set the idx data to be the row index of the original data
    polyGeoms = createGeomData(dataToBeBound)
    geomTree = STRtree(polyGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    return getRowsToKeep(boundingGeoms, geomTree)


####=== Return a list of indices from one dataset for all polygons that contain some lat/lon point
####=== Accepts a buffer (in meters) for the point so get "within radius" polygons of a point.
def getRowsContainingLonLat(thisLon, thisLat, thisData, thisBuffer=0):
    thisBuffer = thisBuffer * 0.000011   ##--convert buffer in meters to degrees using ave spherical at 37deg latitude
    thisPoint = Point(thisLon, thisLat).buffer(thisBuffer)
    polyGeoms = createGeomData(thisData)
    geomTree = STRtree(polyGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    return getRowsToKeep([thisPoint], geomTree)


####=== Return a list of indices from one dataset for all polygons that contain some lat/lon point
####=== Accepts a buffer (in meters) for the point so get "within radius" polygons of a point.
def getRowsContainingPoints(theseLons, theseLats, thisData, thisBuffer=0):
    if len(theseLons) != len(theseLats):
        raise ValueError("The lists of X and Y coords must be the same length.")
    listOfListOfRowsToKeep = []
    polyGeoms = createGeomData(thisData)
    geomTree = STRtree(polyGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    thisBuffer = thisBuffer * 0.000011   ##--convert buffer in meters to degrees using ave spherical at 37deg latitude
    thesePoints = [Point(pt[0], pt[1]).buffer(thisBuffer) for pt in zip(theseLons, theseLats)]   #Point(thisLon, thisLat).buffer(thisBuffer)
    for thisPoint in thesePoints:
        listOfListOfRowsToKeep.append(getRowsToKeep(thesePoints, geomTree))
    return listOfListOfRowsToKeep

###=== Close polygon holes by limitation to the exterior ring.
###=== Example: df.geometry.apply(lambda p: close_holes(p))
def close_holes(poly: Polygon) -> Polygon:
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly

###--- input a dataframe and specify which column to clean, and it removes rows with bad geometries
def removeBadGeometries(thisData, geomColumn='geometry'):
    thisData = thisData[~thisData[geomColumn].is_empty]
    thisData = thisData[~thisData[geomColumn].isna()]
    thisData = thisData[thisData[geomColumn].is_valid]
    return thisData

###=== buffer a geometry in meters
def bufferGeometry(thisGeom, bufferSize):
    thisGeomDist = convertGeomCRS(thisGeom, standardToDistProj)
    thisGeomDist = thisGeomDist.buffer(bufferSize)
    return convertGeomCRS(thisGeomDist, distToStandardProj)






####========================================================================
#####=========================== FOR PLOTTING ==============================
####========================================================================
def ceilInt(number):
    if isinstance(number, collections.Sequence):
        return [int(np.ceil(x)) for x in number]
    else:
        return int(np.ceil(number))

def round1000s(someNumber):
    return int(np.round(someNumber, decimals=-3))

def round100s(someNumber):
    return int(np.round(someNumber, decimals=-2))

def round10s(someNumber):
    return int(np.round(someNumber, decimals=-1))

def normalizeDataPoint(thisValue, dataMin, dataMax):
    return (thisValue - dataMin)/(dataMax - dataMin)

def normalizeVariable(thisDataList, lowerThresh=None, upperThresh=None, clipEnds=False):
    dataMin = np.min(thisDataList) if lowerThresh == None else lowerThresh
    dataMax = np.max(thisDataList) if upperThresh == None else upperThresh
    rescaled = [(x - dataMin)/(dataMax - dataMin) if pd.notnull(x) else np.nan for x in thisDataList]
    if clipEnds:
        rescaled = [np.clip(x, 0, 1) for x in rescaled if pd.notnull(x)]
    return rescaled

def standardizeDataPoint(thisValue, theMean, theStd):
    return (thisValue - theMean) / theStd

def standardizeVariable(thisDataList):
    if type(thisDataList) != list:
        try:
            list(thisDataList)
        except:
            print("  != Can only standardize lists.")
            return None
    thisMean = np.mean(thisDataList)
    theStd = np.std(thisDataList)
    return [(x - thisMean)/theStd if theStd != 0 else 0 for x in thisDataList]

def scaleDataPoint(thisProportion, dataMin, dataMax):
    return dataMin + (thisProportion * (dataMax - dataMin))

def scaleVariable(thisData, thisVariable, thisLevel):
    return thisData[thisVariable].min() + thisLevel * (thisData[thisVariable].max() - thisData[thisVariable].min() )

def normRGB(Red, Green, Blue, A=1.0):
    A = A / 255.0 if A > 1 else A
    return (Red / 255.0, Green / 255.0, Blue / 255.0, A)

def setOpacity(RGBcolor, opacity):
    if not isinstance(RGBcolor,list):
        RGBcolor = list(RGBcolor)
    if len(RGBcolor) == 3:
        RGBcolor.append(opacity)
    elif len(RGBcolor) == 4:
        RGBcolor[3] = opacity
    else:
        print("!! Can only set opacity of RGB and RGBA colors")
    return RGBcolor

###=== Convert colors to hexidecimal format for the visualization layer config
def rgb2hex(r, g, b, a=1):
    baseColor = '#%02x%02x%02x' % (r, g, b)
    opacity = str(makeInt(a * 100)) if a < 1 else ""
    return baseColor + opacity

def hex2rgb(hexColor):
    return mc.hex2color(hexColor)
    # hexColor = hexColor[:-1] if hexColor[0] == '#' else hexColor
    # return tuple(int(hexColor[i:i+2], 16) for i in (0, 2, 4))

def makeColorMap(listOfValues,listOfColors, normed=True, numVals=512):
    if len(listOfValues) != len(listOfColors):
        listOfValues = np.linspace(np.min(listOfValues), np.max(listOfValues), num=len(listOfColors), endpoint=True)
    if normed:
        norm=plt.Normalize(min(listOfValues),max(listOfValues))
        tuples = list(zip(map(norm,listOfValues), listOfColors))
    else:
        tuples = list(zip(listOfValues, listOfColors))
    return LinearSegmentedColormap.from_list("", tuples, N=numVals)

def adjust_lightness(color, amount=0.5):
    # import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

##== input a normalized value (from 0 to 1) and this returns the color for it.
def getColorFromColormap(thisColormap, value):
    return thisColormap(value)

###-----------------------------------------
###=== 8-step single color gradient schemes from ColorBrewer converted to RGBA constants below for running speed
#blueColors = [normRGB(247,251,255), normRGB(222,235,247), normRGB(198,219,239), normRGB(158,202,225), normRGB(107,174,214), normRGB(66,146,198), normRGB(33,113,181), normRGB(8,69,148)]
#greenColors = [normRGB(247,252,245), normRGB(229,245,224), normRGB(199,233,192), normRGB(161,217,155), normRGB(116,196,118), normRGB(65,171,93), normRGB(35,139,69), normRGB(0,90,50)]
#grayColors = [normRGB(255,255,255), normRGB(240,240,240), normRGB(217,217,217), normRGB(189,189,189), normRGB(150,150,150), normRGB(115,115,115), normRGB(82,82,82), normRGB(37,37,37)]
#solarColors = [normRGB(255,245,235), normRGB(254,230,206), normRGB(253,208,162), normRGB(253,174,107), normRGB(253,141,60), normRGB(241,105,19), normRGB(217,72,1), normRGB(140,45,4)]
#purpleColors = [normRGB(252,251,253), normRGB(239,237,245), normRGB(218,218,235), normRGB(188,189,220), normRGB(158,154,200), normRGB(128,125,186), normRGB(106,81,163), normRGB(74,20,134)]
#redColors = [normRGB(255,245,240), normRGB(254,224,210), normRGB(252,187,161), normRGB(252,146,114), normRGB(251,106,74), normRGB(239,59,44), normRGB(203,24,29), normRGB(153,0,13)]

myGreenMed = normRGB(65,174,118)
myBlueLight = normRGB(139,204,247)
myBlueMed = normRGB(66,146,198)
myBlueDark = normRGB(20,78,117)
myRedMed = normRGB(239,59,44)
myGoldMed = normRGB(254,196,79)
myGrayMed = normRGB(150,150,150)
myAlmostBlack = normRGB(10,10,10)
myPureWhite = normRGB(255,255,255)

temperatureColors = [normRGB(234,99,99),normRGB(245,245,245),normRGB(99,131,234)]  ##=== from red at -1 to blue at 1
temperatureColors2 = [normRGB(178,24,43),normRGB(214,96,77),normRGB(244,165,130),normRGB(253,219,199), normRGB(209,229,240),normRGB(146,197,222),normRGB(67,147,195),normRGB(33,102,172)]  ##-- 8 levels
temperatureColors3 = [normRGB(178,24,43),normRGB(178,24,43),normRGB(214,96,77),normRGB(244,165,130),normRGB(200,200,200), normRGB(146,197,222),normRGB(67,147,195),normRGB(33,102,172),normRGB(33,102,172)]  ##-- 9 levels

red2blue = [normRGB(215,25,28),normRGB(253,174,97),normRGB(255,255,191),normRGB(171,217,233),normRGB(44,123,182)] ## 5 step through yellow
blue2red = [normRGB(44,123,182),normRGB(171,217,233),normRGB(255,255,191),normRGB(253,174,97),normRGB(215,25,28)]

blueColors = [(0.9686274509803922, 0.984313725490196, 1.0, 1.0), (0.8705882352941177, 0.9215686274509803, 0.9686274509803922, 1.0), (0.7764705882352941, 0.8588235294117647, 0.9372549019607843, 1.0), (0.6196078431372549, 0.792156862745098, 0.8823529411764706, 1.0), (0.4196078431372549, 0.6823529411764706, 0.8392156862745098, 1.0), (0.25882352941176473, 0.5725490196078431, 0.7764705882352941, 1.0), (0.12941176470588237, 0.44313725490196076, 0.7098039215686275, 1.0), (0.03137254901960784, 0.27058823529411763, 0.5803921568627451, 1.0)]

greenColors = [(0.9686274509803922, 0.9882352941176471, 0.9607843137254902, 1.0), (0.8980392156862745, 0.9607843137254902, 0.8784313725490196, 1.0), (0.7803921568627451, 0.9137254901960784, 0.7529411764705882, 1.0), (0.6313725490196078, 0.8509803921568627, 0.6078431372549019, 1.0), (0.4549019607843137, 0.7686274509803922, 0.4627450980392157, 1.0), (0.2549019607843137, 0.6705882352941176, 0.36470588235294116, 1.0), (0.13725490196078433, 0.5450980392156862, 0.27058823529411763, 1.0), (0.0, 0.35294117647058826, 0.19607843137254902, 1.0)]

grayColors = [(1.0, 1.0, 1.0, 1.0), (0.9411764705882353, 0.9411764705882353, 0.9411764705882353, 1.0), (0.8509803921568627, 0.8509803921568627, 0.8509803921568627, 1.0), (0.7411764705882353, 0.7411764705882353, 0.7411764705882353, 1.0), (0.5882352941176471, 0.5882352941176471, 0.5882352941176471, 1.0), (0.45098039215686275, 0.45098039215686275, 0.45098039215686275, 1.0), (0.3215686274509804, 0.3215686274509804, 0.3215686274509804, 1.0), (0.1450980392156863, 0.1450980392156863, 0.1450980392156863, 1.0)]

purpleColors = [(0.9882352941176471, 0.984313725490196, 0.9921568627450981, 1.0), (0.9372549019607843, 0.9294117647058824, 0.9607843137254902, 1.0), (0.8549019607843137, 0.8549019607843137, 0.9215686274509803, 1.0), (0.7372549019607844, 0.7411764705882353, 0.8627450980392157, 1.0), (0.6196078431372549, 0.6039215686274509, 0.7843137254901961, 1.0), (0.5019607843137255, 0.49019607843137253, 0.7294117647058823, 1.0), (0.41568627450980394, 0.3176470588235294, 0.6392156862745098, 1.0), (0.2901960784313726, 0.0784313725490196, 0.5254901960784314, 1.0)]

redColors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0), (0.996078431372549, 0.8784313725490196, 0.8235294117647058, 1.0), (0.9882352941176471, 0.7333333333333333, 0.6313725490196078, 1.0), (0.9882352941176471, 0.5725490196078431, 0.4470588235294118, 1.0), (0.984313725490196, 0.41568627450980394, 0.2901960784313726, 1.0), (0.9372549019607843, 0.23137254901960785, 0.17254901960784313, 1.0), (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0), (0.6, 0.0, 0.050980392156862744, 1.0)]

solarColors = [(1.0, 0.9607843137254902, 0.9215686274509803, 1.0), (0.996078431372549, 0.9019607843137255, 0.807843137254902, 1.0), (0.9921568627450981, 0.8156862745098039, 0.6352941176470588, 1.0), (0.9921568627450981, 0.6823529411764706, 0.4196078431372549, 1.0), (0.9921568627450981, 0.5529411764705883, 0.23529411764705882, 1.0), (0.9450980392156862, 0.4117647058823529, 0.07450980392156863, 1.0), (0.8509803921568627, 0.2823529411764706, 0.00392156862745098, 1.0), (0.5490196078431373, 0.17647058823529413, 0.01568627450980392, 1.0)]

solarColors2 = [normRGB(144,12,63), normRGB(199,0,57), normRGB(227,97,28), normRGB(241,146,14), normRGB(255,195,0), normRGB(255,226,120)]

darkBlueTab = [0.12156863, 0.46666667, 0.70588235, 1.]
lightBlueTab = [0.68235294, 0.78039216, 0.90980392, 1.]
darkOrangeTab = [1., 0.49803922, 0.05490196, 1.]
lightOrangeTab = [1., 0.73333333, 0.47058824, 1.]
darkGreenTab = [0.17254902, 0.62745098, 0.17254902, 1.]
lightGreenTab = [0.59607843, 0.8745098,  0.54117647, 1.]
darkRedTab = [0.83921569, 0.15294118, 0.15686275, 1.]
lightRedTab = [1., 0.59607843, 0.58823529, 1.]
darkPurpleTab = [0.58039216, 0.40392157, 0.74117647, 1.]
lightPurpleTab = [0.77254902, 0.69019608, 0.83529412, 1.]
darkBrownTab = [0.54901961, 0.3372549, 0.29411765, 1.]
lightBrownTab = normRGB(217, 180, 102)  #[0.76862745, 0.61176471, 0.58039216, 1.]
darkPinkTab =  [0.89019608, 0.46666667, 0.76078431, 1.]
lightPinkTab = [0.96862745, 0.71372549, 0.82352941, 1.]
darkGrayTab = [0.49803922, 0.49803922, 0.49803922, 1.]
lighGrayTab = [0.78039216, 0.78039216, 0.78039216, 1.]
darkOliveTab = [0.7372549,  0.74117647, 0.13333333, 1.]
lightOliveTab = [0.85882353, 0.85882353, 0.55294118, 1.]
darkCyanTab = [0.09019608, 0.74509804, 0.81176471, 1.]
lightCyanTab = [0.61960784, 0.85490196, 0.89803922, 1.]

darkTabColors = [darkBlueTab, darkOrangeTab, darkGreenTab, darkRedTab, darkPurpleTab, darkBrownTab, darkPinkTab, darkGrayTab, darkOliveTab, darkCyanTab]
lightTabColors = [lightBlueTab, lightOrangeTab, lightGreenTab, lightRedTab, lightPurpleTab, lightBrownTab, lightPinkTab, lighGrayTab, lightOliveTab, lightCyanTab]

darkBlueJet = normRGB(0, 0, 255)
lightBlueJet = normRGB(0, 113, 255)
darkOrangeJet = normRGB(255, 121, 0)
lightOrangeJet = normRGB(255, 194, 0)
darkGreenJet = normRGB(5, 128, 0)
lightGreenJet = normRGB(23, 227, 28)
darkRedJet = normRGB(176, 0, 38)
lightRedJet = normRGB(250, 46, 5)
darkPurpleJet = normRGB(203, 0, 253)
lightPurpleJet = normRGB(203, 108, 253)
darkBrownJet = normRGB(162, 76, 0)
lightBrownJet = normRGB(238, 139, 0)
darkPinkJet =  normRGB(255, 67, 255)
lightPinkJet = normRGB(250, 150, 255)
darkOliveJet = normRGB(148, 148, 0)
lightOliveJet = normRGB(241, 241, 12)
darkCyanJet = normRGB(0, 180, 255)
lightCyanJet = normRGB(123, 246, 255)
darkGrayJet = normRGB(84, 39, 135)
lighGrayJet = normRGB(196, 27, 126)

darkJetColors = [darkBlueJet, darkOrangeJet, darkGreenJet, darkRedJet, darkPurpleJet, darkBrownJet, darkPinkJet, darkOliveJet, darkCyanJet, darkGrayJet]
lightJetColors = [lightBlueJet, lightOrangeJet, lightGreenJet, lightRedJet, lightPurpleJet, lightBrownJet, lightPinkJet, lightOliveJet, lightCyanJet, lighGrayJet]

extraDarkTabColors = [adjust_lightness(color, amount=0.5) for color in darkTabColors]
extraLightTabColors = [adjust_lightness(color, amount=1.6) for color in darkJetColors]

DiscreteColorMapColorList = darkTabColors+lightTabColors+darkJetColors+lightJetColors+extraDarkTabColors

spectrum1 = [adjust_lightness(darkPurpleTab,0.5),darkBlueTab,lightCyanTab,normRGB(255,255,191),myGoldMed,darkRedJet]

###-----------------------------------------

def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # print(colors)
    fig, ax = plt.subplots(1, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax.imshow([colors], extent=[0, 10, 0, 1])

def combineColormaps(cmapList):
    cmap = plt.cm.get_cmap(cmapList[0])
    allColors = cmap(np.arange(cmap.N))[10:-10]
    # print(allColors)
    for thisCmap in cmapList[1:]:
        cmap = plt.cm.get_cmap(thisCmap)
        colors = cmap(np.arange(cmap.N))
        # print(colors)
        allColors = np.concatenate((allColors,colors), axis=0)

def getDiscreteColorMap(numBins=None):
    numBins = len(DiscreteColorMapColorList) if numBins == None else numBins
    if numBins > len(DiscreteColorMapColorList):  ##if more colors are asked for than exist, make more (crappy ones)
        theseColors = DiscreteColorMapColorList + [adjust_lightness(color, amount=0.6) for color in DiscreteColorMapColorList] + [adjust_lightness(color, amount=1.4) for color in DiscreteColorMapColorList]
        if numBins > len(theseColors):
            print("There are",numBins,"categories, but only",len(theseColors),"are supported.")
            return plt.get_cmap(mc.ListedColormap(theseColors), numBins)
        else:
            return plt.get_cmap(mc.ListedColormap(theseColors[:numBins]), numBins)
    else:
        return plt.get_cmap(mc.ListedColormap(DiscreteColorMapColorList[:numBins]), numBins)



####========================================================================
#####=========================== FOR MAPPING =============================
####========================================================================

def makeLegendTicks(numTicks=4, minVal=None, maxVal=None, categorical=False):
    tickList = []
    if categorical:
        if numTicks <= 20:
            tickList = range(numTicks)
        else:
            numberToSkip = makeInt(np.ceil(numTicks/20))
            tickList = range(numTicks)[::numberToSkip]
    else:
        for thisTick in range(numTicks):
            ### If the range is small, just use the values
            if 1 + makeInt(np.floor(maxVal) - np.ceil(minVal)) < numTicks:
                if thisTick == 0:
                    tickList.append(rnd(minVal))
                elif thisTick == numTicks-1:
                    tickList.append(rnd(maxVal))
                else:
                    tickList.append(rnd(scaleDataPoint(thisTick/(numTicks-1), minVal, maxVal)))
            ### If the range is large enough,
            else:
                if thisTick == 0:
                    tickList.append(np.ceil(minVal))
                elif thisTick == numTicks-1:
                    tickList.append(np.floor(maxVal))
                else:
                    if (maxVal - minVal) > 30000:
                        tickList.append(round1000s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                    elif (maxVal - minVal) > 3000:
                        tickList.append(round100s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                    elif (maxVal - minVal) > 300:
                        tickList.append(round10s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                    else:
                        tickList.append(makeInt(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
    return tickList



###==================================================================================================
def makeNetworkDataMap(theNodeData=pd.DataFrame(), theEdgeData=pd.DataFrame(), theNodeVariable=None, theNodeVariableName="none", theEdgeVariable=None, theEdgeVariableName="none", theNodeColormap=None, theEdgeColormap=None, theLegend=None, nodeMinVal=None, nodeMaxVal=None, edgeMinVal=None, edgeMaxVal=None, numTicks=None, categorical=False, fileIndex="", folderName="../Map Images/", figWidth=10, figHeight=7, outputDPI=150, mapBounds=None, nodeSize="none", edgeWidth="none"):

    theData = theEdgeData.copy()  ##-- use the edgeData for bounds if the nodeData is not specified
    ###=== Set defaults for easier plotting with partial info
    if len(theNodeData) > 0:
        if theNodeData.crs.to_string() != mappingCRS:
            theNodeData = theNodeData.to_crs(mappingCRS)
        if theNodeVariable == None:
            theNodeData['nodeConstant'] = 0
            theNodeVariable = 'nodeConstant'
        nodeMinVal = theNodeData[theNodeVariable].min() if nodeMinVal == None else nodeMinVal
        nodeMaxVal = theNodeData[theNodeVariable].max() if nodeMaxVal == None else nodeMaxVal
        theNodeVariableName = theNodeVariable if theNodeVariableName == "none" else theNodeVariableName
        theNodeColormap = makeColorMap([nodeMinVal,nodeMaxVal], [myAlmostBlack,myAlmostBlack]) if theNodeColormap == None else theNodeColormap
        theData = theNodeData.copy()

    if len(theEdgeData) > 0:
        if theEdgeData.crs.to_string() != mappingCRS:
            theEdgeData = theEdgeData.to_crs(mappingCRS)
        if theEdgeVariable == None:
            theEdgeData['edgeConstant'] = 0
            theEdgeVariable = 'edgeConstant'
        edgeMinVal = theEdgeData[theEdgeVariable].min() if edgeMinVal == None else edgeMinVal
        edgeMaxVal = theEdgeData[theEdgeVariable].max() if edgeMaxVal == None else edgeMaxVal
        theEdgeVariableName = theEdgeVariable if theEdgeVariableName == "none" else theEdgeVariableName
        theEdgeColormap = makeColorMap([-1,1], [myAlmostBlack,myAlmostBlack]) if theEdgeColormap == None else theEdgeColormap

    if ((len(theNodeData) == 0) & (len(theEdgeData) == 0)):
        print("You must provide either node or edge data (or both).")
        return None

    nodeSize = 5 if nodeSize == 'none' else nodeSize
    edgeWidth = 1 if edgeWidth == 'none' else edgeWidth


    # print("  -- Number of rows before cleaning", len(theData))
    theData = removeBadGeometries(theData, geomColumn='geometry')
    # print(theData.head())
    # print(list(theData['geometry'])[0:5])
    # print("  -- Number of rows after cleaning", len(theData))
    ###--- bounds need to be in mapping CRS, so this converts them if not already in it
    if not 'geomMap' in list(theData):
        theData['geomMap'] = theData['geometry'].apply(lambda row: convertGeomCRS(row, standardToMapProj))

    # totalBounds = list(theData['geometry'].total_bounds) if mapBounds == None else mapBounds
    # print("totalBounds =", totalBounds)

    # print("length of theData", len(list(theData['geomMap'])))

    if mapBounds == None:  ##-- add a 4% buffer around the data area.
        totalBounds = list(theData['geomMap'].total_bounds)
        # print(totalBounds)
        # minPoint = convertGeomCRS(Point(totalBounds[0], totalBounds[1]), standardToMapProj)
        # maxPoint = convertGeomCRS(Point(totalBounds[2], totalBounds[3]), standardToMapProj)
        # xMin, yMin, xMax, yMax = [minPoint.x, minPoint.y, maxPoint.x, maxPoint.y]
        xMin, yMin, xMax, yMax = totalBounds
        xMin = xMin - (0.04 * (xMax - xMin))
        xMax = xMax + (0.04 * (xMax - xMin))
        yMin = yMin - (0.04 * (yMax - yMin))
        yMax = yMax + (0.04 * (yMax - yMin))
    else:
        if mapBounds[1] > mapBounds[0]:  ###--- check if lon, lat are switched, and fix if not (for Japan area)
            mapBounds = [ mapBounds[1], mapBounds[0], mapBounds[3], mapBounds[2] ]

        if mapBounds[0] < 1000:  ## bounds are in standard CRS, so reproj them
            minPoint = convertGeomCRS(Point(mapBounds[0], mapBounds[1]), standardToMapProj)
            maxPoint = convertGeomCRS(Point(mapBounds[2], mapBounds[3]), standardToMapProj)
            xMin, yMin, xMax, yMax = [minPoint.x, minPoint.y, maxPoint.x, maxPoint.y]
        else:
            xMin, yMin, xMax, yMax = mapBounds

    # print([xMin, yMin, xMax, yMax])

    ###=== restrict plotting area to the bounds
    boundingBox = box(xMin, yMin, xMax, yMax)
    plotArea = boundingBox.area
    print("The surface area of plotting region:", plotArea)
    if mapBounds != None:  ##-- only restrict if we're not plotting the whole DF...this takes time, but should speed up plotting
        theData = theData[theData['geomMap'].intersects(boundingBox)]
        theData['geomMap'] = theData['geomMap'].apply(lambda geom: geom.intersection(boundingBox))

    ###===  Set zoom level based on area covered
    theZoom = 12  ##-- good for 23 wards, which has area 1906886264
    if plotArea < (1906886264 * 0.65):
        theZoom = 13
    if plotArea < (1906886264 * 0.35):
        theZoom = 14

    fig, ax = plt.subplots(1, figsize=(figWidth, figHeight))

    if ((len(theNodeData) > 0) & (len(theEdgeData) > 0)):
        ax = theNodeData.plot(ax=ax, column=theNodeVariable, cmap=theNodeColormap, vmin=nodeMinVal, vmax=nodeMaxVal, alpha=0.6, markersize=nodeSize, marker="o", edgecolor='None')
        theEdgeData.plot(ax=ax, column=theEdgeVariable, cmap=theEdgeColormap, vmin=edgeMinVal, vmax=edgeMaxVal, alpha=0.8, linewidth=edgeWidth)
    elif ((len(theNodeData) == 0) & (len(theEdgeData) > 0)):
        ax = theEdgeData.plot(ax=ax, column=theEdgeVariable, cmap=theEdgeColormap, vmin=edgeMinVal, vmax=edgeMaxVal, alpha=0.8, linewidth=edgeWidth)
    elif ((len(theNodeData) > 0) & (len(theEdgeData) == 0)):
        ax = theNodeData.plot(ax=ax, column=theNodeVariable, cmap=theNodeColormap, vmin=nodeMinVal, vmax=nodeMaxVal, alpha=0.6, markersize=nodeSize, marker="o", edgecolor='None')
    else:
        print("You must provide either node or edge data (or both).")
        return None

    # tileSource = ctx.providers.Stamen.TonerLite
    # tileSource = ctx.providers.CartoDB.Positron
    # tileSource = ctx.providers.CartoDB.Voyager
    tileSource = ctx.providers.Esri.WorldGrayCanvas
    ctx.add_basemap(ax, zoom=theZoom, source=tileSource, attribution_size=6, attribution=u'(C)\u2002Esri\u2002--\u2002Esri,\u2002DeLorme,\u2002NAVTEQ')
    # ctx.add_basemap(ax, zoom=12, source=ctx.providers.CartoDB.PositronOnlyLabels, attribution=u'Tiles\u2002(C)\u2002Esri\u2002--\u2002Esri,\u2002DeLorme,\u2002NAVTEQ\u2002|\u2002Labels\u2002(C)\u2002OpenStreetMap\u2002contributors\u2002(C)\u2002CARTO')
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    ##=== If theLegend==False, then don't put a legend at all.
    ##=== If the theLegendColormap is not specified, then use the same colormap.
    ##=== this is only necessary until we can figure out how to get a legend that supports alpha
    if theLegend != False:
        if theLegend == None:  ##-- if there is an unspecificed legend, and both data are provided, default to nodes
            if len(theNodeData) > 0:
                theLegend = "nodes"
            else:
                theLegend = "edges"

        theLegendColormap = theNodeColormap if theLegend == "nodes" else theEdgeColormap
        minVal = nodeMinVal if theLegend == "nodes" else edgeMinVal
        maxVal = nodeMaxVal if theLegend == "nodes" else edgeMaxVal
        theData = theNodeData if theLegend == "nodes" else theEdgeData
        theVariable = theNodeVariable if theLegend == "nodes" else theEdgeVariable
        theVariableName = theNodeVariableName if theLegend == "nodes" else theEdgeVariableName
        numTicks = 1 + (makeInt(np.floor(maxVal) - np.ceil(minVal))) if numTicks == None else numTicks

        if ((categorical) & (numTicks <= plt.cm.get_cmap(theLegendColormap).N)):
            norm= mc.BoundaryNorm(np.arange(0,numTicks+1)-0.5, numTicks)
            scalarmappable = cm.ScalarMappable(norm=norm, cmap=theLegendColormap)
        else:
            scalarmappable = cm.ScalarMappable(cmap=theLegendColormap)
            scalarmappable.set_clim(minVal, maxVal)
        scalarmappable.set_array(theData[theVariable])
        cbaxes = inset_axes(ax, width=figWidth*0.022, height=figHeight*0.235, loc='lower right', bbox_transform=ax.transAxes, bbox_to_anchor=(0.84, 0.04, 0.1, 0.05), borderpad=0.)
        cbaxes.set_alpha(0.65)
        theTicks = makeLegendTicks(numTicks, minVal, maxVal, categorical)
        cbar = plt.colorbar(scalarmappable, cax=cbaxes, ticks=theTicks, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        cbaxes.yaxis.set_ticks_position("left")
        if categorical:
            if numTicks <= 20:
                cbar.set_ticks(np.arange(0,numTicks))
                cbar.set_ticklabels([str(makeInt(x)) for x in range(numTicks)])
            else:
                numberToSkip = makeInt(np.ceil(numTicks/20))
                cbar.set_ticks(np.arange(0,numTicks)[::numberToSkip])
                cbar.set_ticklabels([str(makeInt(x)) for x in range(numTicks)[::numberToSkip]])

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(theVariableName, rotation=270)

    # fig.suptitle("Total Population Aged 15-65", fontname="Arial", fontsize=18, y=-0.1)
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.show()
    if fileIndex != False:
        theNodeVariableName = '' if theNodeVariableName == 'nodeConstant' else "_"+theNodeVariableName
        theEdgeVariableName = '' if theEdgeVariableName == 'edgeConstant' else "_"+theEdgeVariableName
        fig.savefig(folderName+theNodeVariableName+theEdgeVariableName+"-Map"+fileIndex+".png", dpi=outputDPI, transparent=True, bbox_inches='tight', pad_inches=0)
    #     # fig.savefig(folderName+theVariable+"-Map"+fileIndex+"sm.png", dpi=72, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

###==================================================================================================
def makeGeoDataMap(theData, theVariable, theColormap, theVariableName="poo", theLegendColormap=None, minVal=None, maxVal=None, numTicks=4, categorical=False, fileIndex="", folderName="../Map Images/", figWidth=10, figHeight=7, outputDPI=150, mapBounds=None, outlineColor=None, zoomLevel=None, opacity=0.85):

    fig, ax = plt.subplots(1, figsize=(figWidth, figHeight))
    minVal = theData[theVariable].min() if minVal == None else minVal
    maxVal = theData[theVariable].max() if maxVal == None else maxVal
    outlineColor = outlineColor if outlineColor != None else "face"
    lineWidth = 0 if outlineColor == 'none' else 0.8
    if theData.crs.to_string() != mappingCRS:
        theData = theData.to_crs(mappingCRS)

    if mapBounds == None:  ##-- add a buffer around the data area.
        xMin, yMin, xMax, yMax = list(theData['geometry'].total_bounds)
        xMin = xMin - (0.04 * (xMax - xMin))
        xMax = xMax + (0.04 * (xMax - xMin))
        yMin = yMin - (0.04 * (yMax - yMin))
        yMax = yMax + (0.04 * (yMax - yMin))
    else:
        if mapBounds[1] > mapBounds[0]:  ###--- check if lon, lat are switched, and fix if not (for Japan area)
            mapBounds = [ mapBounds[1], mapBounds[0], mapBounds[3], mapBounds[2] ]

        if mapBounds[0] < 1000:  ## bounds are in standard CRS, so reproj them
            minPoint = convertGeomCRS(Point(mapBounds[0], mapBounds[1]), standardToMapProj)
            maxPoint = convertGeomCRS(Point(mapBounds[2], mapBounds[3]), standardToMapProj)
            xMin, yMin, xMax, yMax = [minPoint.x, minPoint.y, maxPoint.x, maxPoint.y]
        else:
            xMin, yMin, xMax, yMax = mapBounds

    ###=== restrict plotting area to the bounds
    boundingBox = box(xMin, yMin, xMax, yMax)
    plotArea = boundingBox.area
    rect = patches.Rectangle((xMin, yMin), xMax - xMin, yMax - yMin, linewidth=1, edgecolor='none', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

    # print("The surface are of plotting region:", plotArea)
    if mapBounds != None:  ##-- only restrict if we're not plotting the whole DF...this takes time, but should speed up plotting
        theData = theData[theData['geometry'].intersects(boundingBox)]
        theData['geometry'] = theData['geometry'].apply(lambda geom: geom.intersection(boundingBox))

    ###===  Set zoom level based on area covered
    if zoomLevel == None:
        theZoom = 12  ##-- good for 23 wards, which has area 1906886264
        if plotArea < (1906886264 * 0.65):
            theZoom = 13
        if plotArea < (1906886264 * 0.35):
            theZoom = 14
    else:
        theZoom = zoomLevel

    # print("[",xMin,",", yMin,",", xMax,",", yMax,"]")
    # xMin, yMin, xMax, yMax = [15467054.4135, 4232200.62482, 15575946.2476, 4286645.82715] ##itosanken?
    # xMin, yMin, xMax, yMax = [15497318.54462218, 4232088.870202411, 15584674.986741155, 4289158.680551786]  ## Tokyo Main with Elevation
    ax = theData.plot(ax=ax, column=theVariable, cmap=theColormap, vmin=minVal, vmax=maxVal, alpha=opacity, edgecolor=outlineColor, linewidth=lineWidth, antialiased=True)

    # tileSource = ctx.providers.Stamen.TonerLite
    # tileSource = ctx.providers.CartoDB.Positron
    # tileSource = ctx.providers.CartoDB.Voyager
    tileSource = ctx.providers.Esri.WorldGrayCanvas
    ctx.add_basemap(ax, zoom=theZoom, source=tileSource, attribution='')
    ctx.add_basemap(ax, zoom=theZoom, source=ctx.providers.CartoDB.PositronOnlyLabels, attribution=u'Tiles\u2002(C)\u2002Esri\u2002--\u2002Esri,\u2002DeLorme,\u2002NAVTEQ\u2002|\u2002Labels\u2002(C)\u2002OpenStreetMap\u2002contributors\u2002(C)\u2002CARTO')
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    theVariableName = theVariable if theVariableName == "poo" else theVariableName

    ###=== If theLegendColormap==False, then don't put a legend at all.
    ###=== If the theLegendColormap is not specified, then use the same colormap.
    ###=== this is only necessary until we can figure out how to get a legend that supports alpha
    if theLegendColormap != False:
        if theLegendColormap == None:
            theLegendColormap = theColormap
        if ((categorical) & (numTicks <= plt.cm.get_cmap(theLegendColormap).N)):
            norm= mc.BoundaryNorm(np.arange(0,numTicks+1)-0.5, numTicks)
            scalarmappable = cm.ScalarMappable(norm=norm, cmap=theLegendColormap)
        else:
            scalarmappable = cm.ScalarMappable(cmap=theLegendColormap)
            scalarmappable.set_clim(minVal, maxVal)
        scalarmappable.set_array(theData[theVariable])
        cbaxes = inset_axes(ax, width=figWidth*0.018, height=figHeight*0.2, loc='lower right', bbox_transform=ax.transAxes, bbox_to_anchor=(0.84, 0.04, 0.1, 0.05), borderpad=0.)
        cbaxes.set_alpha(0.65)
        theTicks = makeLegendTicks(numTicks, minVal, maxVal, categorical)
        cbar = plt.colorbar(scalarmappable, cax=cbaxes, ticks=theTicks, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        cbaxes.yaxis.set_ticks_position("left")
        if categorical:
            if numTicks <= 20:
                cbar.set_ticks(np.arange(0,numTicks))
                cbar.set_ticklabels([str(makeInt(x)) for x in range(numTicks)])
            else:
                numberToSkip = makeInt(np.ceil(numTicks/20))
                cbar.set_ticks(np.arange(0,numTicks)[::numberToSkip])
                cbar.set_ticklabels([str(makeInt(x)) for x in range(numTicks)[::numberToSkip]])

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(theVariableName, rotation=270)

    # fig.suptitle("Total Population Aged 15-65", fontname="Arial", fontsize=18, y=-0.1)
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.show()
    if fileIndex != False:
        fig.savefig(folderName+theVariableName+"-Map"+fileIndex+".png", dpi=outputDPI, transparent=True, bbox_inches='tight', pad_inches=0)
        # fig.savefig(folderName+theVariable+"-Map"+fileIndex+"sm.png", dpi=72, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




# ####====== Create CSV files to test the data in Kepler
# def createDataForKepler(varList, filename='', dataType='hexData', region=None):
#     if dataType == 'hexData':
#         varList = varList + ['in23Wards', 'inTokyoMain']
#         thisData = getDataForVariables(varList, dataType)

#     if dataType == 'chomeData':
#         varList = varList + ['in23Wards', 'inTokyoMain']
#         thisData = getDataForVariables(varList, dataType)
#         thisData = thisData[thisData['lowestLevel'] == True]

#     if region == 'in23Wards':
#         thisData = thisData[thisData['in23Wards'] == True]
#     if region == 'inTokyoMain':
#         thisData = thisData[thisData['inTokyoMain'] == True]

#     for thisVar in ['geomDist', 'geomAngle', 'geomMap', 'lat', 'lon']:
#         if thisVar in list(thisData):
#             thisData.drop(columns=[thisVar], inplace=True)

#     writeGeoCSV(thisData, "../Data/MapImages/"+filename+'-'+dataType+"-Kepler.csv")



####====== Create CSV files to test the data in Kepler... incorporate more stuff from makeKeplerMap below
def writeDataForKepler(theData, filename='testData', version=None, region=None):

    thisData = theData.copy()

    for thisVar in ['geomDist', 'geomAngle', 'geomMap', 'lat', 'lon', 'point']:
        if thisVar in list(thisData):
            thisData.drop(columns=[thisVar], inplace=True)

    if 'id' in list(thisData):
        thisData['id'] = thisData['id'].astype(str)

    if version == None:
        writeGeoCSV(thisData, "../Data/MapImages/"+filename+"-Kepler.csv")
    else:
        writeGeoCSV(thisData, "../Data/MapImages/"+filename+'-'+version+"-Kepler.csv")


####========================================================================
def makeKeplerMap(thisData, thisVarList, theName="someData", theConfig=None, mappingArea='in23Wards'):
    import keplergl

    theData = thisData.copy()

    dataType = 'Other'
    ###=== Some data we always want to include, depending on the file, then add the variables of interest to the base and reduce
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    if 'addressCode' in list(theData):
        dataType = 'chomeData'
        ###=== Check if theData has the core data, and if not add it
        if 'geometry' not in list(theData):
            theData = pd.merge(loadCoreData(dataType), theData, how='left', on='addressCode')
            theData = theData[theData['lowestLevel'] == True]
            thisVarList.extend(('geometry', 'addressName', 'addressCode', 'landArea'))

    if 'hexNum' in list(theData):
        dataType = 'hexData'
        if 'geometry' not in list(theData):
            theData = pd.merge(loadCoreData(dataType), theData, how='left', on='hexNum')
            thisVarList.extend(('geometry', 'hexNum'))
    else:
        thisVarList.append('geometry')


    ###=== Check if the dataframe is already geopandas; if not, make it one.
    if isinstance(list(theData['geometry'])[0], str):
        theData = convertToGeoPandas(theData)

    ##-- Reduce the data to the geographical area of interest
    if ((mappingArea == 'in23Wards') & ('in23Wards' in list(theData))):
        theData = theData[theData['in23Wards'] == True]
    if ((mappingArea == 'inTokyoMain') & ('inTokyoMain' in list(theData))):
        theData = theData[theData['inTokyoMain'] == True]


    ###==== Check the polygons are all valid, ...some may be made invalid by changing the CRS, so fix them.
    for thisIndex in theData.index.values:
        if theData.at[thisIndex, 'geometry'].is_valid == False:
            theData.at[thisIndex, 'geometry'] = theData.at[thisIndex, 'geometry'].buffer(0)

    for thisVar in thisVarList:
        if thisVar not in list(theData):
            thisVarList.remove(thisVar)

    ###------------------------- Safe up to here
#    print(theData.head())
#    print(thisVarList)

#    theData = theData[thisVarList]  ##-- Reduce the data to the columns of interest  ##-- this generates a recursive depth error.
#    print(theData.head())
#    print(thisVarList)


    if theConfig == None:
        kmap = keplergl.KeplerGl(height=400, data={theName: theData})
    else:
        kmap = keplergl.KeplerGl(height=400, data={theName: theData}, config=theConfig)
#    print("writing html")
    kmap.save_to_html(file_name="../Map Images/"+dataType+"-"+theName+".html")


####=============================================================================
####================= CREATE VARIABLE LISTS FOR VIZENGINE =======================
####=============================================================================

#thisVarList = ['elevationMin', 'elevationMean', 'elevationMax', 'slopeMin', 'slopeMean', 'slopeMedian', 'slopeMax']
#addVarsToLocatorDict(thisVarList, "Geography", dataType='hexData')

#addVarsToLocatorDict('totalPopulation', "Core", dataType='hexData')
##
#print(readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')['hexData'])

#print(list(getDataForTopic("Population")))

#allVarList = getVisVarNames("Crime", dataType='hexData') + getVisVarNames("Economics", dataType='hexData') + getVisVarNames("Environment", dataType='hexData') + getVisVarNames("Geography", dataType='hexData') + getVisVarNames("Population", dataType='hexData') + getVisVarNames("Transportation", dataType='hexData')
#
#print(allVarList)
#print("Num hexData variables for far:", len(allVarList))

#varsToUse = ['Hex_CrimeTotalRate', 'Hex_CrimeFelonyRobberyRate', 'Hex_CrimeFelonyOtherRate', 'Hex_CrimeViolentWeaponsRate', 'Hex_CrimeViolentAssaultRate', 'Hex_CrimeViolentInjuryRate', 'Hex_CrimeViolentIntimidationRate', 'Hex_CrimeViolentExtortionRate', 'Hex_CrimeTheftBurglarySafeRate', 'Hex_CrimeTheftBurglaryEmptyHomeRate', 'Hex_CrimeTheftBurglaryHomeSleepingRate', 'Hex_CrimeTheftBurglaryHomeUnnoticedRate', 'Hex_CrimeTheftBurglaryOtherRate', 'Hex_CrimeTheftVehicleRate', 'Hex_CrimeTheftMotorcycleRate', 'Hex_CrimeTheftPickPocketRate', 'Hex_CrimeTheftPurseSnatchingRate', 'Hex_CrimeTheftBagLiftingRate', 'Hex_CrimeTheftOtherRate', 'Hex_CrimeOtherMoralIndecencyRate', 'Hex_CrimeOtherOtherRate', 'Hex_NumJobs', 'Hex_NumCompanies', 'Hex_GreenArea', 'Hex_NoiseMin', 'Hex_NoiseMean', 'Hex_NoiseMax', 'Hex_PercentCommercial', 'Hex_PercentIndustrial', 'Hex_PercentResidential', 'Hex_MeanPercentLandCoverage', 'Hex_MeanTotalPercentLandCoverage', 'Hex_ElevationMin', 'Hex_ElevationMean', 'Hex_ElevationMax', 'Hex_SlopeMin', 'Hex_SlopeMean', 'Hex_SlopeMedian', 'Hex_SlopeMax', 'Hex_NumHouseholds', 'Hex_Pop_Total_A', 'Hex_Pop_0-19yr_A', 'Hex_Pop_20-69yr_A', 'Hex_Pop_70yr+_A', 'Hex_Pop_20-29yr_A', 'Hex_Pop_30-44yr_A', 'Hex_Pop_percentForeigners', 'Hex_Pop_percentChildren', 'Hex_Pop_percentMale', 'Hex_Pop_percentFemale', 'Hex_Pop_percent30-44yr', 'Hex_TimeToTokyo', 'Hex_TimeAndCostToTokyo']
#
#for thisVar in varsToUse:
#    print('''        "'''+thisVar+'''": {
#            "colors": "white2red",
#            "reverse": false,
#            "type": "standardized",
#            "interpolate": true
#        },''')
#
#
#for thisVar in varsToUse:
#    print("               '"+thisVar+"',")







###=====================================================================================================
###==================================== GET POLYGON FOR NAMED AREA(S) ==================================
###=====================================================================================================

###=== For addresses entered in
def getPolygonForArea(prefName, districtName='', cityName='', oazaName='', chomeName=''):

    if ((prefName == '23Wards') | (prefName == '23区')):
        return readPickleFile('../Data/Polygons/wardsPolygon2.pkl')
    elif prefName == 'tokyoMain':
        return readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')
    elif prefName == 'tokyoArea':
        return readPickleFile('../Data/Polygons/tokyoAreaPolygon2.pkl')
    elif ((prefName == 'japan') | (prefName == 'Japan')):
        return readPickleFile('../Data/Polygons/japanMultiPolygon2.pkl')
    else:
        ###=== get all the data
        adminData = getDataForVariables(['prefName', 'districtName', 'cityName', 'oazaName', 'chomeName'], dataType="chomeData", fillNaN='')
        ###=== filter to data within this prefecture
        adminData = adminData[adminData['prefName'].apply(lambda val: prefName in val )]
        # print(adminData.head())
        if len(adminData) == 0:
            print("  == No prefecture called", prefName, "found.")
            return None
        else:
            if districtName != '':
                adminData = adminData[adminData['districtName'].apply(lambda val: districtName in val )]
                if len(adminData) == 0:
                    print("  == No district called", districtName, "found.")
            if cityName != '':
                adminData = adminData[adminData['cityName'].apply(lambda val: cityName in val )]
                if len(adminData) == 0:
                    print("  == No city called", cityName, "found.")
            if oazaName != '':
                adminData = adminData[adminData['oazaName'].apply(lambda val: oazaName in val )]
                if len(adminData) == 0:
                    print("  == No oaza called", oazaName, "found.")
            if chomeName != '':
                adminData = adminData[adminData['chomeName'].apply(lambda val: chomeName in val )]
                if len(adminData) == 0:
                    print("  == No chome called", chomeName, "found.")

            if len(adminData) == 0:
                print("  == No areas found matching this area/address.")
            elif len(adminData) == 1:
                return list(adminData['geometry'])[0]
            else:
                return unary_union(list(adminData['geometry']))
###--------------------------------------------------------

# thisPolygon = getPolygonForArea('東京', oazaName='神宮前')
# print(thisPolygon)









####========================================================================
#####==================== GETTING ELEVATION DATA ===========================
####========================================================================

###=== Shapely changed the behavior of STRtree to return indices instead of geoms, so this returns the geoms
def getOverlappingGeoms(strTree, queryGeom):
    ###=== get indices of geometries in tree with bounding boxes that intersect this query geom
    overlappingIndices = strTree.query(queryGeom)
    ###=== get the geometries for those indices
    return strTree.geometries.take(overlappingIndices).tolist()

###=== Returns the lowest index closest point's index value from a dataframe of locations to a single point
def getClosestPoint(originLat, originLon, nodeDF):
    dists = np.array([euclideanDistance(originLon, originLat, lon, lat) for lon, lat in zip(nodeDF.lon, nodeDF.lat)])
    return dists.argmin(), dists.min()

def getClosestPointHaversine(originLat, originLon, nodeDF):
    dists = np.array([haversineDist(originLon, originLat, lon, lat) for lon, lat in zip(nodeDF.lon, nodeDF.lat)])
    return dists.argmin(), dists.min()


###=== Uses a folder full of geoTiffs to create a lookup dataframe.
def createRasterTileBoundaryDataframe(rasterFilesDirectory: str = None, targetDirectory = None):
    if rasterFilesDirectory == None:
        print("Please pass a directory containing raster files of the format dddd_dd.tif\n"
              "to createRasterTileBoundaryDataframe")
    if targetDirectory == None: targetDirectory = rasterFilesDirectory
    for root, dirs, files in os.walk(rasterFilesDirectory, topdown=False):
        fileList = files

    ###=== Filter the filelist for appropriate files using the pattern => .tif format, dddd_dd.tif
    p = re.compile(r"\d\d\d\d_\d\d\.tif")
    fileList = [ f for f in fileList if p.match(f)]

    standardToDistProj = createstandardToDistProj()

    ###=== We make the dataframe from a list of dictionaries, one dict for each tile.
    boundaryDictList = []   ##-- first, create empty list
    for fileName in fileList:
        with rasterio.open(rasterFilesDirectory+"/"+fileName) as src:
            maxY = src.bounds.top
            minY = src.bounds.bottom
            minX = src.bounds.left
            maxX = src.bounds.right
        theGeom = box(minX, minY, maxX, maxY)
        theGeomDist = convertGeomCRS(theGeom, standardToDistProj)
        boundaryDict = {"minX":minX, "minY":minY, "maxX":maxX, "maxY":maxY, 'geometry':theGeom, 'geomDist':theGeomDist, "filename":fileName}
        boundaryDictList.append(boundaryDict)  ##-- add this dictionary to the list

    boundaryDataframe = pd.DataFrame(boundaryDictList)  ##-- create the dataframe from the list of dictionaries

    ###=== Save to .csv in the target directory (by default - in the same directory as the raster files)
    if not os.path.exists(targetDirectory):
        os.makedirs(targetDirectory)
    boundaryDataframe.to_csv(targetDirectory+"/"+"boundaryDataframe.csv")
    # boundaryDataframe.to_pickle(targetDirectory + "/" + "boundaryDataframe.pkl")
    return boundaryDataframe



###=== This function returns the filename of the tile that contains the passed coordinates,
###=== found according to the .csv file at bounds_fp. A default filepath for bounds_fp is specified.
def getRasterTileForLatLon(thisLon: float, thisLat: float, thisBoundaryDataframe = None):

    ###=== If we are not passed a boundaryDataframe, we load the backup from the .csv (can change to .pkl later)
    ###=== Should probably use: thisBoundaryDataframe = readGeoPickle('...')
    ###=== This is a dataframe containing columns [minX, maxX, minY, maxY, geometry, filename]
    ###=== Because reading the dataframe is inefficient for multiple calls, it is better to pass it to the function.
    if thisBoundaryDataframe is None:
        thisBoundaryDataframe = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles/boundaryDataFrame.csv')

    ###=== get the row corresponding to the tile
    thisTile = thisBoundaryDataframe[((thisBoundaryDataframe['minX'] < thisLon) &
                                      (thisBoundaryDataframe['minY'] < thisLat) &
                                      (thisBoundaryDataframe['maxX'] > thisLon) &
                                      (thisBoundaryDataframe['maxY'] > thisLat)
                                      )]

    ###=== If no tile found
    if len(thisTile) == 0:
        print("Coordinates are outside our covered area.")
    return thisTile['filename'].iloc[0]

###=== Returns the filename of the tile that contains the passed Point.
def getRasterTileForPoint(point: Point, thisBoundaryDataframe = None):
    # print(f'Reading point object{point} in getRasterTile')
    thisLon = point.x
    thisLat = point.y
    return getRasterTileForLatLon(thisLon, thisLat, thisBoundaryDataframe)

###=== Get the elevation for a lon,lat pair using the 5m GeoTiff data
def getLatLonElevation(thisLon: float, thisLat: float, thisBoundaryDataframe = None, rasterFilesDirectory: str = None):
    ###=== If we are passed no boundaryDataframe, we load the backup from the .csv (can change to .pkl later)
    # should probably use : thisBoundaryDataframe = readGeoPickle('...')
    # this is a dataframe containing columns [minX, maxX, minY, maxY, geometry, filename]
    # Note that reading the dataframe over and over is inefficient and for consecutive calls
    #   it is better to pass it to the function.
    if not isinstance(thisBoundaryDataframe, pd.DataFrame):
        thisBoundaryDataframe = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles/boundaryDataFrame.csv')

    ###=== If we are passed no filepath to the directory with all the raster tiles, we use a default
    if rasterFilesDirectory == None:
        rasterFilesDirectory = '../Data/Altitude/Elevation5mWindowFiles'
        # rasterFilesDirectory = '../Data/Altitude/Elevation5mWindowFilesDistCalc'

    ###=== Get the filepath to the appropriate raster tile
    tile_filename = getRasterTileForLatLon(thisLon, thisLat, thisBoundaryDataframe)
    tile_fp = rasterFilesDirectory + "/" + tile_filename
    with rasterio.open(tile_fp) as src:
        ###=== The .sample() function accepts a list and provides an iterable
        ###=== We write from the iterable to a list, in this case only one point.
        samplePoiList = []
        ###=== Our rasters have only one band, so we don't need to worry about selecting the appropriate one.
        for val in src.sample([(thisLon, thisLat)]):
            samplePoiList.append(val)
        return samplePoiList[0][0]

###=== Convert a point to a lon/lat pair and get the elevation for it.
def getPointElevation(point, thisBoundaryDataframe=None, rasterFilesDirectory:str =None):
    thisLon = point.x
    thisLat = point.y
    return getLatLonElevation(thisLon, thisLat, thisBoundaryDataframe, rasterFilesDirectory)

###=== Input an edge (LineString) geometry and it returns an elevation profile: a list of x,h points.
###=== where x is the distance from the start of the LineString and h is the height at that point.
###=== Note that the geometries should be in GeomDist so that the intervals work properly.
def createElevationProfile(thisGeom, interval=10, thisBoundaryDataFrame=None, thisRasterFilesPath=None):
    # check that object is linestring
    if not isinstance(thisGeom, LineString):
        if isinstance(thisGeom, Point):
            ###--- if the geom is a point, then just return a list of one item with its elevation
            return [(0, getPointElevation(thisGeom, thisBoundaryDataFrame, thisRasterFilesPath))]
        else:
            ###--- If it's a polygon, or not a geometry, then ...
            raise TypeError("The object passed to createElevationProfile is not a valid LineString geometry")
            return False

    # read in the boundarydataframe if not passed one
    # this gets passed down all the way down to getLatLonElevation - if it does not, every step loads it for itself,
    # which is extremely slow.
    if not isinstance(thisBoundaryDataFrame, pd.DataFrame):
        thisBoundaryDataframe = pd.read_csv('../Data/Altitude/Elevation5mWindowFilesDistCalc/boundaryDataFrame.csv')

    if thisRasterFilesPath is None:
        thisRasterFilesPath = '../Data/Altitude/Elevation5mWindowFilesDistCalc'

    # determine the number of points to sample
    numPoints = int(thisGeom.length // interval)

    # generate a list of Point objects and their distances sampled along the geometry, append last point manually
    pointListWithDistances = [
        (thisGeom.interpolate(i * interval, normalized=False), i * interval)
        for i in range(numPoints + 1)
        ]

    # we need to use .coords instead of .boundary since some LineString objects have no boundary:
    # in particular, looped LineStrings.boundary == EMPTY MULTILINESTRING == [].
    pointListWithDistances.append(( Point(thisGeom.coords[-1]), thisGeom.length))

    # add elevations to the points
    profile = [
        (thisPointAndDistance[1],
         getPointElevation(thisPointAndDistance[0], thisBoundaryDataFrame, thisRasterFilesPath))
        for thisPointAndDistance in pointListWithDistances
    ]
    return profile

###=== Enter the profile of something (Like a road edge) and get the "effort" or "effective distance" score for it.
def getSlopeScoreProfile(thisProfile, intensityWeight=1, irregularityWeight=1, interval=10, thisBoundaryDataFrame=None, thisRasterFilesPath=None):
    ###=== effectiveLength = L * (1 + alpha*SUM( h^2/d ) ) * (1 + beta*sigma )
    # where:
    # E = effort
    # L = total length
    # alpha = intensityWeight
    # h,d = chunk height steps/ distances
    # beta = irregularityWeight
    # sigma = variance of inclination in chunks

    theDistance = np.abs((thisProfile[0][0]-thisProfile[-1][0]))
    theLength = 0
    theIntensity = 0

    ###=== thisProfile is an array of structure [(h0,d0), (h1,d1), ...]
    # where h is the height at this point and d is the distance along the line
    # we need to build a list containing data of each interval triangle from point to point, instead of the points themselves
    stepList = [
        ( thisProfile[i+1][0]-thisProfile[i][0], thisProfile[i+1][1]-thisProfile[i][1])
        for i in range(len(thisProfile)-1)
    ]

    segmentSlopeList = []
    ###=== lets compute the total length of the path in 3d space:
    ### L in the above formula ###
    for triangle in stepList:
        # triangle[1] is the delta height
        # triangle[0] is the delta length
        theLength += math.sqrt( triangle[0]**2 + triangle[1]**2 )    ##-- this will be the sum of the hypotenuses of the interval triangles
        theIntensity += triangle[1]**2 / triangle[0]
        segmentSlopeList.append(triangle[1]/triangle[0])  ###-- just using slope variance instead of angle variance

    ###=== lets also compute "intensity":
    ### SUM( h^2/d ) in the above formula ###
    # this is a function of the general inclination of the slopes, but scales hard with steeper slopes
    # for a flat route, this will compute to 0
    # print(stepList)

    ###=== finally, lets compute the variance of the angle of incline over the triangles
    ### sigma in the above formula ###
    # it's kind of unclear whether this is additive in the same way as the other parameters are, but lets go with it for now
    # first, we compute a list of all the angles:
    # inclineAngleList = [
    #     math.atan(triangle[1]/triangle[0]) # in radians
    #     # math.atan(triangle[1]/triangle[0])*180/math.pi # in degrees
    #     for triangle in stepList
    # ]

    ###=== get variance of the slopes with numpy.variance
    theIrregularity = np.var(segmentSlopeList)

    ###=== finally, lets compile some score, using the weight variables passed into the function (default 0)
    thisEffectiveLength = theLength * ( 1 + intensityWeight * theIntensity ) * ( 1 + irregularityWeight * theIrregularity )
    ###=== thisEffort is essentially an effective length for comparison. so,
    thisEffort = thisEffectiveLength / theDistance

    return (theLength, thisEffectiveLength, thisEffort, thisProfile)


###=== This is a function to get data from the old elevation files for speed comparisons
# def getLatLonElevationFromPickleWithRtree(thisLat, thisLon, bounds_df = None):
#     ###=== Swap around the lat/lon if necessary
#     if thisLat > 90:
#         tempLon = thisLat
#         thisLat = thisLon
#         thisLon = tempLon
#     ###=== Load a dataframe default if none is passed
#     if not isinstance(bounds_df, pd.DataFrame):
#         bounds_df = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles_old/boundaryDataFrame.csv')
#     ###=== Retreive the row for the appropriate file
#     thisIndex = bounds_df.index[((bounds_df['minX'] < thisLon) &
#                                       (bounds_df['minY'] < thisLat) &
#                                       (bounds_df['maxX'] > thisLon) &
#                                       (bounds_df['maxY'] > thisLat)
#                                       )]
#     k = thisIndex[0]
#     thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles_old/elevationData-5mGrid-'+str(k)+'.pkl')
#     ###=== instead of using STRtree I will use Rtree instead. There appears to be
#     ###=== a bug in the shapely library that is causing the issue(maybe)

#     # make a point object
#     poi = Point(thisLon, thisLat)
#     gdf = gp.GeoDataFrame(thisBlock, geometry = 'geometry')

#     # make a tiny box for the point for intersection lookup
#     box = poi.bounds

#     # use the indexing provided by geopandas to Rtree the search
#     spatial_index = thisBlock.sindex
#     possible_matches_index = list(spatial_index.intersection(box))
#     possible_matches = thisBlock.iloc[possible_matches_index]
#     precise_matches = possible_matches[possible_matches.intersects(poi)]
#     return precise_matches.iloc[0, 0]


    # gridGeoms = list(thisBlock['geometry'])
    # gridValues = list(thisBlock['elevation'])
    # for index, geom in enumerate(gridGeoms):
    #     gridGeoms[index].idx = gridValues[index]
    # gridTree = STRtree(gridGeoms)
    # print(gridTree)
    # print("Now beginning the query...")
    # thisGrid = gridTree.query(thisPoint)
    # print(thisGrid)


###=== This is a function to get data from the old elevation files for speed comparisons
# def getLatLonElevationFromPickleWithIter(thisLat, thisLon, bounds_df = None):
#     ###=== Swap around the lat/lon if necessary
#     if thisLat > 90:
#         tempLon = thisLat
#         thisLat = thisLon
#         thisLon = tempLon
#     ###=== Load a dataframe default if none is passed
#     if not isinstance(bounds_df, pd.DataFrame):
#         bounds_df = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles_old/boundaryDataFrame.csv')
#     ###=== Retreive the row for the appropriate file
#     thisIndex = bounds_df.index[((bounds_df['minX'] < thisLon) &
#                                       (bounds_df['minY'] < thisLat) &
#                                       (bounds_df['maxX'] > thisLon) &
#                                       (bounds_df['maxY'] > thisLat)
#                                       )]
#     if len(thisIndex) == 0:
#         print("It would appear that the point passed was not found in boundaryDataFrame in getLatLonElevationFromPickleWithIter")
#         print(f"The point was lon={thisLon}, lat={thisLat}")
#     k = thisIndex[0]
#     thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles_old/elevationData-5mGrid-'+str(k)+'.pkl')
#     ###=== instead of using STRtree I will use Rtree instead. There appears to be
#     ###=== a bug in the shapely library that is causing the issue (maybe)

#     # make a point object
#     poi = Point(thisLon, thisLat)

#     # Iterate over all the geometry in the block until .contains == True
#     for i, slice in enumerate(thisBlock['geometry']):
#         if slice.contains(poi):
#             return thisBlock.loc[i, 'elevation']


###=== Get the 5m elevation for a particular point (which can only need a single block even if it's exactly on an edge)

# def getLatLonElevationOld(thisLat, thisLon, thisBoundaryDict=None):
#     if thisBo*undaryDict == None:
#         thisBoundaryDict = readPickleFile('../Data/Altitude/Elevation5mWindowFiles_old/boundaryDict.pkl')
#     thisPoint = Point(thisLon, thisLat)
#     for k,v in thisBoundaryDict.items():
#         if thisPoint.intersects(v["geometry"]):
#             thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles_old/elevationData-5mGrid-'+str(k)+'.pkl')
#             gridGeoms = list(thisBlock['geometry'])
#             gridValues = list(thisBlock['elevation'])
#             for index, geom in enumerate(gridGeoms):
#                 gridGeoms[index].idx = gridValues[index]
#             gridTree = STRtree(gridGeoms)
#             thisGrid = gridTree.query(thisPoint)
#             return thisGrid[0].idx

def addLatLonElevations(theData, thisBoundaryDict=None):
#    print(theData.head(3))
    runStartTime = time.time()
    if thisBoundaryDict == None:
        thisBoundaryDict = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/boundaryDict.pkl')
    theData.loc[:,'thisPoint'] = theData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    theData['elevation'] = [None] * len(theData)
    for k,v in thisBoundaryDict.items():
        runStartTime = printProgress(runStartTime,k,len(thisBoundaryDict))
        thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
        gridGeoms = list(thisBlock['geometry'])
        gridValues = list(thisBlock['elevation'])
        for index, geom in enumerate(gridGeoms):
            gridGeoms[index].idx = gridValues[index]
        gridTree = STRtree(gridGeoms)

        relevantData = theData[theData.apply(lambda row: row['thisPoint'].intersects(v['geometry']), axis=1)]
        if len(relevantData) > 0:
            print("  -- Relevant elevation data found for window",k)
            theseNodes = list(relevantData.index.values)
            for thisNode in theseNodes:
                thisPoint = theData.loc[thisNode,'thisPoint']
                thisGrid = gridTree.query(thisPoint)
                if len(thisGrid) > 0:
                    theData.at[thisNode,'elevation'] = thisGrid[0].idx
                else:
                    print("    --Somehow, node",thisNode,"intersects the geometry but not any grid within it")
                    print("    --ThisGrid=",thisGrid)

    return theData

###=== Add elevation data to all Nodes, or just a subset, by converting to a dataframe, running the elevation assignment based on a window-by-window approach
# def addElevationsToNodesOLD(thisNetwork, theseNodes=None):
#     if theseNodes == None:
#         nodeData = [{'id': n, **attr} for n,attr in thisNetwork.nodes(data=True)]
#     else:
#         nodeData = [{'id': n, **attr} for n,attr in thisNetwork.nodes(data=True) if (n in theseNodes) ]
#     nodeDF = pd.DataFrame.from_dict({'id': [n['id'] for n in nodeData], 'lon': [n['lon'] for n in nodeData], 'lat': [n['lat'] for n in nodeData]}).set_index('id')
#     print(nodeDF.head())
#     print("  -- Starting elevation lookup process")
#     nodeDF = addLatLonElevations(nodeDF, thisBoundaryDict=None)

#     def modifyNode(row):
#         thisNetwork.nodes[row.name]['elevation'] = row['elevation']
# #        thisNetwork.nodes[row.index.values.astype(int)[0]]['elevation'] = row['elevation']
# #        thisNetwork.nodes[row['id']]['elevation'] = row['elevation']

#     nodeDF.apply(modifyNode, axis=1)
#     return thisNetwork

####==== Generate a list of (x,y) values along a straight line connecting two geoPoints
####==== The y-values are the elevations sampled from the 5m grid data at chosen intervals along the line
def getLineElevations(thisLine, pointInterval=10, thisBoundaryDataFrame=None, projection = None):
    # run checks for Point and other unsavory objects
    if isinstance(thisLine, Point):
        # print('Detected POINT object, skipping it...')
        return None

    if len(list(thisLine.boundary)) > 1:
        endPoint = convertGeomCRS(list(thisLine.boundary)[1], projection)
    else:
        # print('Detected POINT object, skipping it...')
        return None

    # read in the boundarydataframe if not passed one
    if not isinstance(thisBoundaryDataFrame, pd.DataFrame):
        thisBoundaryDataframe = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles/boundaryDataFrame.csv')

    # make a projection if none is passed (pass it, this is slow)
    if not isinstance(projection, Transformer):
        projection = createDistToStandardProj()

    # how many points do we slice this edge into?
    numPoints = int(thisLine.length // pointInterval)

    # the profile is stored as [(0, h0), (r1, h1), ....]
    profileList = []

    # we need the points in EPSG:4326 to pass them to the elevation getter
    pathPoints = [thisLine.interpolate(i * pointInterval, normalized=False) for i in range(numPoints + 1)]
    pathPoints = [convertGeomCRS(point, projection) for point in pathPoints]

    # some of the objects may be Point: check for this
    # construct the endpoint separately

    # iterate over the converted points
    for i, thisPoint in enumerate(pathPoints):
        # append the (r, h) tuple to the list
        try:
            profileList.append((i * pointInterval,
                                getLatLonElevation(thisPoint.x, thisPoint.y, thisBoundaryDataFrame)))
        except:
            profileList.append([i * pointInterval,
                                0])
            # Assign a value of 0 when crossing a location not in elevation data
    # Append the last point distance & elevation
    profileList.append([thisLine.length,
                        getPointElevation(endPoint, thisBoundaryDataFrame)])

    # the profile is constructed.
    return profileList


def upgradeNetworkTileWithProfiles(source_fp, out_dir=None, thisBoundaryDataFrame=None,
                                   projection = None):
    # A meaty function that will read in existing simplified tiles,
    # construct elevation profiles for each edge,
    # add them to the network as edge attribute 'profile'
    # and write the new network to a file

    # a default projection
    if not isinstance(projection, Transformer):
        projection = createDistToStandardProj()

    # a default boundary dataframe
    if not isinstance(thisBoundaryDataFrame, pd.DataFrame):
        thisBoundaryDataframe = pd.read_csv('../Data/Altitude/Elevation5mWindowFiles/boundaryDataFrame.csv')

    # a default write directory
    if out_dir == None:
        out_dir = '../Data/OSMData/SimpleWalkabilityNetworkTilesWithProfiles'

    # strip the / or \ on directory end just in case
    if (out_dir[-1] == '/') or (out_dir[-1] == '\\'):
        out_dir = out_dir.pop

    # make the directory if it doesen't exist:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # get the filename of the source file
    source_name = re.search(r'[\/\\](.(?![\/\\]))+$', source_fp).group(0)[1:]

    # declare start of work
    print(f"Beginning upgrade of tile {source_name}...")

    # read in the network
    tile = readPickleFile(source_fp)

    # make a dictionary with profiles
    profile_dict = {}
    for thing in tqdm(tile.edges.data('geomDist')):
        # thing is a 3-tuple of (nodeid1, nodeid2, geomDist)
        profile = getLineElevations(thing[2], 10, thisBoundaryDataFrame=thisBoundaryDataFrame, projection=projection)
        if profile:
            profile_dict[(thing[0], thing[1])] = profile

    # add this dictionary as attribute to the network
    nx.set_edge_attributes(tile, profile_dict, 'profile')

    # write to file
    writePickleFile(tile, out_dir + '/' + source_name)
    print(f'Succesfully wrote tile {source_name}')
    print(f'  --> to {out_dir +"/" + source_name}')


###===================== GET ELEVATION PROFILES FOR ROUTES ====================

###=== Construct a geodataframe containing only the ids and geometries for building strTrees and closest points
def makeNodeGeomDF(thisNetwork, theGeometry = 'geomDist'):
    gdf = gp.GeoDataFrame(list(thisNetwork.nodes.data(theGeometry)), columns = ['id', theGeometry])
    return gdf[['id', theGeometry]]

###=== Feed in a point and a nodeDF made with GetNodeGeoms and it returns the ID and Point of the closest node to that point
def getClosestNodeID(poi, nodeDF):
    dists = np.array([poi.distance(point) for point in nodeDF['geomDist']])
    return nodeDF.iloc[dists.argmin(),0], nodeDF.iloc[dists.argmin(),1]

###=== Find the closest node to a lon/lat of interest.
###=== This uses Euclidean distance, so you should input geoms in distCalcCRS
def findClosestNodeToLatLon(thisLon, thisLat, nodeDF):
    poi = Point(thisLon, thisLat)  ##-- this must be in distCalcCRS
    # poi = convertGeomCRS(poi, standard_to_dist_proj)
    ###=== To reduce the number of distance calculations, iteratively expand the search area
    potentialNearestNodes = nodeDF[nodeDF['geomDist'].within(poi.buffer(10))]  ##-- with buffer 10m
    if len(potentialNearestNodes) == 0:
        potentialNearestNodes = nodeDF[nodeDF['geomDist'].within(poi.buffer(20))]  ##-- if none found, try 20m
    if len(potentialNearestNodes) == 0:
        potentialNearestNodes = nodeDF[nodeDF['geomDist'].within(poi.buffer(100))]  ##-- if none found, try 100m
    ###=== if at least one node is found, get the closest one among the candidates
    if len(potentialNearestNodes) > 0:
        closest_id, closest_point = getClosestNodeID(poi, potentialNearestNodes)
    ###== if still no nodes are found, then just test the whole nodeDF
    else:
        closest_id, closest_point = getClosestNodeID(poi, nodeDF)

    return closest_id, closest_point


###=== For a given start and end location, return the list of nodes along the shortest path between them.
###=== Instead of the Network, this uses a nodeDF greated using makeNodeGeomDF having just ID,geomDist
def getPathListFromNodeDF(startLon, startLat, endLon, endLat, nodeDF, thisNetwork, edgeWeight='timeWeight'):
    ###=== Find closest points in the dataframe to our given coords
    startNodeId, startNodeGeom = findClosestNodeToLatLon(startLon, startLat, nodeDF)
    endNodeId, endNodeGeom = findClosestNodeToLatLon(endLon, endLat, nodeDF)
    # use dijkstras algorithm to get pathList
    return nx.shortest_path(thisNetwork, startNodeId, endNodeId, edgeWeight)


###=== For a given start and end location, return the list of nodes along the shortest path between them.
###=== If you are doing multiple routes, do makeNodeGeomDF once and feed in the nodeDF directly into getPathListFromNodeDF
def getPathListFromNetwork(startLon, startLat, endLon, endLat, thisNetwork, edgeWeight='timeWeight'):
    nodeDF = makeNodeGeomDF(thisNetwork)  ##-- make the nodes into a dataframe of just ID,geomDist
    return getPathListFromNodeDF(startLon, startLat, endLon, endLat, nodeDF, thisNetwork, edgeWeight)


# the idea is to walk a path from node to node in the list,
# get all of the profiles of all of the edges on the way,
# and merge them into one long profile-list
# and return it.
def getPathProfile(pathNodesList, thisNetwork):
    allEdgesTupleList = [(a, b, c) for (a, b, c) in thisNetwork.edges(data='profile', default=None)]
    edgeList = []
    for i in range(len(pathNodesList) - 1):
        thisNode = pathNodesList[i]
        nextNode = pathNodesList[i + 1]
        for edgeTuple in allEdgesTupleList:
            if (thisNode, nextNode) == (edgeTuple[0], edgeTuple[1]):
                # False = edge is not traversed in reverse
                edgeList.append([thisNode, nextNode, False, edgeTuple[2]])
            elif (thisNode, nextNode) == (edgeTuple[1], edgeTuple[0]):
                # True = edge is in reverse.
                profile = edgeTuple[2]
                reversedProfile = [(profile[-1][0] - r, h) for r, h in reversed(profile)]
                edgeList.append([thisNode, nextNode, True, reversedProfile])
    # now we fix the r-distances in the profile -> combine them into one long one
    current_r = 0
    profile = []
    # for edge in edgeList:
    # print(f'---{edge[0],edge[1]}')
    # print(f'{edge[3]}')
    for edge in edgeList:
        # print(f'---{edge[0],edge[1]}')
        for r in edge[3]:
            profile.append((current_r + r[0], r[1]))
        current_r = profile[-1][0]

    return profile


def getPathProfileLonLat(startLon, startLat, endLon, endLat, network):
    # get the path
    pathList = getPathList(startLon, startLat, endLon, endLat, network)
    # get its profile
    return getPathProfile(pathList, network)

# ####==== Generate a list of (x,y) values along a straight line connecting two geoPoints
# ####==== The y-values are the elevations sampled from the 5m grid data at chosen intervals along the line
# def getLineElevations(lat1, lon1, lat2, lon2, thisBoundaryDict, pointInterval = 5):
#     thisLine = LineString([Point(lon1, lat1), Point(lon2, lat2)])
#     allData = gp.GeoDataFrame()
#     ##collect data tiles for all areas where the line goes
#     for k,v in thisBoundaryDict.items():
#         if thisLine.intersects(v["geometry"]):
#             thisData = ('../Data/Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
#             # edges = readCSV("./test_outputs/trainNetwork_nodes-beforeArakawaFix.csv")
#             thisData = thisData[thisData["geometry"].intersects(thisLine)]
#             allData = gp.GeoDataFrame(pd.concat([allData,thisData], ignore_index=True))

#     gc.collect()
#     if len(allData) > 0:
# #        print(len(allData))
#         ###=== Create Rtree from all the needed data tiles
#         gridGeoms = list(allData['geometry'])
#         gridValues = list(allData['elevation'])
#         for index, geom in enumerate(gridGeoms):
#             gridGeoms[index].idx = gridValues[index]
#         gridTree = STRtree(gridGeoms)

#         theDistance = makeInt(haversineDist(lon1, lat1, lon2, lat2))
# #        print("theDistance",theDistance)
#         numPoints = makeInt(theDistance / pointInterval)
#         pathPoints = [thisLine.interpolate(i/float(numPoints - 1), normalized=True) for i in range(numPoints)]
# #        print(pathPoints)

# #        pathGrids = gp.overlay(allData, thisLine, how='intersection')
# #        print(pathGrids)
# #        return pathGrids
#         pathElevations = []
#         for index,thisPoint in enumerate(pathPoints):
#             try:
#                 pathElevations.append([index * pointInterval, np.round(gridTree.query(thisPoint)[0].idx,1)])
#             except:
#                 pathElevations.append([index * pointInterval, 0])  #### Assign a value of 0 when crossing a location not in elevation data
#         return pathElevations
#     else:
#         return [0]

####==== Return the y-values after applying a moving average smoothing operation
def getMovingAverageValues(Xs, Ys, smoothFactor = 5):
    aveSmooth_yValues = np.array(Ys)
    for i in range(smoothFactor,len(aveSmooth_yValues)-smoothFactor):
        aveSmooth_yValues[i] = np.mean(aveSmooth_yValues[i-smoothFactor:i+smoothFactor])
    return aveSmooth_yValues

####==== Generate polynomial fit points that include the first and last point
def getPolynomialFitValues(Xs, Ys, polyRank = 15):
    def polyFit(x, *params):
        return np.poly1d(params)(x)
    sigma = np.ones(len(Xs))   ## Create a sequence of weights to force matching on first/last point
    sigma[[0, -1]] = 0.01           ## Set the weights of the first and last value very small (meaning strong)
    ####=== start with a high rank polynomial, and gradually decrease to linear if optimization doesn't converge
    try:
        polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(polyRank), sigma=sigma, maxfev=5000)
    except:
        try:
            polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(10), sigma=sigma, maxfev=5000)
        except:
            try:
                polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(5), sigma=sigma, maxfev=5000)
            except:
                polyFitFunc = np.polyfit([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], 1)
    return polyFit(Xs, *polyFitFunc)

####==== Calculate the slope in degrees from changes in Y and X values (in the same units, e.g. meters)
def getSlopeAngle(deltaY,deltaX):
    return np.rad2deg(np.arctan2(deltaY, deltaX))

def getSlopeFromElevations(elevation1, elevation2, distance, directed=False):
    if directed == True:
        return np.rad2deg(np.arctan2(elevation2 - elevation1, distance))
    else:
        return abs(np.rad2deg(np.arctan2(elevation2 - elevation1, distance)))

####==== For a list of X and Y values, return the maxSlope, meanSlope, medianSlope, minSlope
def getSlopeStats(Xs, Ys):
    allSlopes = [getSlopeAngle(Ys[i+1] - Ys[i], Xs[i+1] - Xs[i]) for i,x in enumerate(Xs[:-1])]
    return (rnd(np.max(allSlopes)), rnd(np.mean(allSlopes)), rnd(np.median(allSlopes)), rnd(np.min(allSlopes)))


# ####==== Make a straight line between two points, use
# def lineElevationProfile(point1, point2, thisName, saveLoc=None, thisBoundaryDict=None):
#     if thisBoundaryDict == None:
#         thisBoundaryDict = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/boundaryDict.pkl')
#     gc.collect()
#     startTime = time.time()
#     pointInterval = 5    ##distance between point samples in meters
#     elevationList = getLineElevations(point1[0], point1[1], point2[0], point2[1], thisBoundaryDict, pointInterval)
#     #print(elevationList)
#     pathFindTime = np.round((time.time()-startTime),2)

#     ####--- Actual Values
#     xValues, yValues = zip(*elevationList)

#     fig, ax = plt.subplots(figsize=(15, 3.5))
#     plt.plot(xValues, yValues, c=normRGB(169,169,169,0.75), label='Raw Data')  ##gray

#     ####--- Moving window average smoothing
#     plt.plot(xValues, getMovingAverageValues(xValues, yValues), c=normRGB(220,20,60,0.55), label='Moving Average')  ##red

#     ####--- Polynomial Fit Smoothing
#     #poly = np.polyfit(xValues, yValues, 5)
#     #poly_yValues = np.poly1d(poly)(xValues)
#     #plt.plot(xValues, poly_yValues)
#     poly_yValues = getPolynomialFitValues(xValues, yValues)
#     plt.plot(xValues, poly_yValues, c=normRGB(30,144,255,0.75), label='Polynomial Fit')  ##blue

#     ####----Get information about the slopes (using the fitted polynomial)
#     maxSlope, meanSlope, medianSlope, minSlope = getSlopeStats(xValues, poly_yValues)
#     plt.title(thisName+" | Slopes: max="+str(maxSlope)+"  mean="+str(meanSlope)+"  median="+str(medianSlope)+"  min="+str(minSlope), fontsize=16)
#     ####--- Change legend location so it's less likely to cover the data
#     if yValues[-1] > yValues[0]:
#         plt.legend(loc="lower right")
#     else:
#         plt.legend(loc="upper right")
#     plt.xlabel("Distance (m)", fontsize=15)
#     plt.ylabel("Elevation (m)", fontsize=15)
#     plt.show()
#     if saveLoc != False:
#         saveLoc = '../Map Images/' if saveLoc == None else saveLoc
#         fig.savefig(saveLoc+'elevationProfile-'+thisName+'.png', dpi=150, transparent=True, bbox_inches='tight')
#     print("--Found Path Elevation in",pathFindTime,"seconds")




####========================================================================
####========================== NETWORK ALGORITHMS ==========================
####========================================================================

def getAllNodeAttributes(thisNetwork):
    return list(set([item for n,d in thisNetwork.nodes(data=True) for item in list(d.keys()) ]))

def getAllEdgeAttributes(thisNetwork):
    return list(set([item for n1,n2,d in thisNetwork.edges(data=True) for item in list(d.keys()) ]))

def getUniqueNodeAttrValues(thisNetwork, thisVar):
    varVals = nx.get_node_attributes(thisNetwork, thisVar)
    return list(set([v for k,v in varVals.items()]))    ###== return the unique values for this variable

def getUniqueEdgeAttrValues(thisNetwork, thisVar):
    varVals = nx.get_edge_attributes(thisNetwork, thisVar)
    return list(set([v for k,v in varVals.items()]))    ###== return the unique values for this variable

####=== Return a list of nodeNums for nodes matching the conditions
####=== If thisVal == None, then return all nodes that have the attribute
####=== Otherwise return the nodes that have a specific value for that attribute
def getNodesByAttr(thisNetwork, thisVar, thisVal=None):
    if thisVal == None:
        return [node for node,attr in thisNetwork.nodes(data=True) if attr.get(thisVar, None) != None]
    else:
        return [node for node,attr in thisNetwork.nodes(data=True) if attr.get(thisVar, None) == thisVal]

def getEdgesByAttr(thisNetwork, thisVar, thisVal=None):
    if thisVal == None:
        return [(n1,n2) for n1,n2,attr in thisNetwork.edges(data=True) if attr.get(thisVar, None) != None]
    else:
        return [(n1,n2) for n1,n2,attr in thisNetwork.edges(data=True) if attr.get(thisVar, None) == thisVal]

###=== the default subgraph method is weird, and not what anybody wants, this is a substitute
def getSubgraph(thisNetwork, nodesToKeep):
#    nodesToRemove = [node for node in thisNetwork.nodes(data=False) if node not in nodesToKeep]
    nodesToRemove = list(set(thisNetwork.nodes()) - set(nodesToKeep))
    subNetwork = thisNetwork.copy()
    subNetwork.remove_nodes_from(nodesToRemove)
    return subNetwork

####=== I tested whether it's faster to remove nodes from a copy, or build from scratch (as per NetworkX site): the above is 8x faster
#def getSubgraph2(thisNetwork, nodesToKeep):
#    subNetwork = nx.empty_graph(n=0, create_using=nx.Graph())
#    subNetwork.add_nodes_from((n, thisNetwork.nodes[n]) for n in nodesToKeep)
##    subNetwork.add_edges_from((u, v, attr) for u,v,attr in thisNetwork.edges(data=True) if ((u in nodesToKeep) & (v in nodesToKeep)))
#    subNetwork.add_edges_from((n, nbr, d) for n, nbrs in thisNetwork.adj.items() if n in nodesToKeep for nbr, d in nbrs.items() if nbr in nodesToKeep)
#    return subNetwork

####=== Calculate traversal time via walking based on distance in meters and a speed in kph
walkingSpeed = 4.8 #kph
bicycleSpeed = walkingSpeed * 2.5  ## => 12 kph
slowSpeed = walkingSpeed * 0.75  ## => 3.6 kph
# print(walkingSpeed * 1000 / 60)
def metersToMinutes(theDistance, theSpeed=walkingSpeed):
    return theDistance / (theSpeed * 1000 / 60)  ## convert kph into meters/minute

def minutesToMeters(theTime, theSpeed=walkingSpeed):
    return ((theTime / 60) * theSpeed) * 1000   ##




###=== Import a networkX graph and it exports a geopandas dataframe containing both nodes and edges with properties.
def convertGraphToGeopandas(inputNetwork, modesToKeep=None):
    thisNetwork = inputNetwork.copy()
    if modesToKeep != None:   ### Filter to desired node modality if entered
        modesToKeep = [modesToKeep] if isinstance(modesToKeep, str) else modesToKeep ## support entering a single mode as a str
        nodesToKill = [node for node,attr in thisNetwork.nodes(data=True) if attr.get('modality', None) not in modesToKeep]
        #        print(nodesToKill)
        thisNetwork.remove_nodes_from(nodesToKill)  ## this works inplace

    allNodeProperties = list(set([k for n in thisNetwork.nodes for k in thisNetwork.nodes[n].keys()]))
    #    print("allNodeProperties",allNodeProperties)

    nodeData = [{'id': str(n[0]), **n[1]} for n in thisNetwork.nodes(data=True)]  ### Create a list of dictionaries for each row
    #    print(nodeData[0])
    ###--- First create the base geopandas dataframe, then add columns for the other attributes
    nodeData = [n for n in nodeData if n.get('geometry','poo')!='poo']
    # nodeDF = gp.GeoDataFrame({'geometry': [Point(n['lon'], n['lat']) if n.get('geometry',None) == None else n['geometry'] for n in nodeData], 'id': [n['id'] for n in nodeData]})
    nodeDF = gp.GeoDataFrame({'geometry': [n['geometry'] for n in nodeData], 'id': [n['id'] for n in nodeData]}, geometry='geometry')

    nodeDF.crs = standardCRS
    for thisNodeProperty in allNodeProperties:
    #        print("  -- processing property:", thisNodeProperty)
        nodeDF[thisNodeProperty] = [n.get(thisNodeProperty, None) for n in nodeData]
    nodeDF = nodeDF.copy()

    if nx.is_directed(thisNetwork):
        edgeDF = nx.to_pandas_edgelist(thisNetwork, 'source', 'target')
    else:
        edgeDF = nx.to_pandas_edgelist(thisNetwork, 'source', 'target')

    if len(edgeDF) > 0:
        edgeDF = gp.GeoDataFrame(edgeDF, geometry='geometry')
        edgeDF.crs = standardCRS
    else:
        edgeDF = gp.GeoDataFrame(columns=['source', 'target', 'geometry'])

    ##=== These should already be shapely because the networks are stored as pkl
    nodeDF = loadOtherGeoms(nodeDF)
    edgeDF = loadOtherGeoms(edgeDF)

    return (nodeDF, edgeDF)


###=== Create separate node and edge CSV files for visualization in Kepler
###=== by inputing a boundingArea poygon in standardCRS, you can crop the network to create more mangeable files
def convertNetworkToKeplerFiles(thisNetwork=None, filename="networkSample", boundingArea=None, inputGeomDist=False):

    if not isinstance(filename, str):
        filename = str(filename)

    boundingArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl') if boundingArea == '23Wards' else boundingArea
    ###=== get the network for this boundingArea if no network is input
    if ((thisNetwork == None) & (boundingArea != None)):
         thisNetwork = getNetworkTilesForPolygon(boundingArea, networkFolder=filename, cropToPolygon=False) ##-- cropping is done later

    if thisNetwork != None:
        if (len(thisNetwork) == 0):
            print("Network contains no data.")
        else:
            ###=== Reduce the network to the area within the boundingArea if it is specified
            if boundingArea != None:
                nodesToKeep = flattenList([[u,v] for u,v,attr in thisNetwork.edges(data=True) if attr['geometry'].intersects(boundingArea)])
                thisNetwork = getSubgraph(thisNetwork, nodesToKeep)

            print("  -- Number of Nodes:", thisNetwork.number_of_nodes(), "  |  Number of Edges:", thisNetwork.number_of_edges() )
            nodeDF, edgeDF = convertGraphToGeopandas(thisNetwork)
            # nodeDF['id'] = nodeDF['id'].astype('string') # or str
            ###=== remove the distCalcCRS, and lat, lon columns for visualization (takes up memory, but can't be used)
            if len(nodeDF) > 0:
                for thisVar in ["geomDist", "lat", "lon", "geomAngle", "geomMap", "pointMap"]:
                    if thisVar in list(nodeDF):
                        nodeDF = nodeDF.drop(columns=[thisVar])
                if 'id' in list(nodeDF):
                    nodeDF['id'] = nodeDF['id'].astype(str)
                writeGeoCSV(nodeDF, '../Data/MapImages/'+filename+'_nodes.csv')

            if len(edgeDF) > 0:
                for thisVar in ["geomDist", "lat", "lon", "geomAngle", "geomMap", "pointMap"]:
                    if thisVar in list(edgeDF):
                        edgeDF = edgeDF.drop(columns=[thisVar])
                if 'source' in list(edgeDF):
                    edgeDF['source'] = edgeDF['source'].astype(str)
                if 'target' in list(edgeDF):
                    edgeDF['target'] = edgeDF['target'].astype(str)
                writeGeoCSV(edgeDF, '../Data/MapImages/'+filename+'_edges.csv')


# ###=================== NETWORK TILES METHODS ===================
# def getNetworkTile(thisComboIndex, networkFolder="fullNetwork_v1"):
#     try:
#         thisTile = readPickleFile('../Data/OSMData/'+networkFolder+'/networkGrid='+thisComboIndex+'.pkl')
#         return thisTile
#     except:
#         print("Tile for index",thisComboIndex,"could not be found.  Loading empty tile.")
#         thisTile = nx.Graph({})
#         return thisTile


# ###=== Load a network tile and generate csv files fo the nodes and edges for Kepler visualization
# ###=== The networkType can be "Full", "Walkability" or "StationFinder"
# def convertNetworkToKeplerFiles(thisNetwork, filename="temp"):
#     if not isinstance(filename, str):
#         filename = str(filename)
#     if len(thisNetwork) == 0:
#         print(" !! --Network contains no data.")
#     else:
#         nodeDF, edgeDF = convertGraphToGeopandas(thisNetwork)
#         if len(nodeDF) > 0:
#             for thisVar in ["geomDist", "lat", "lon", "geomMap"]:
#                 if thisVar in list(nodeDF):
#                     nodeDF = nodeDF.drop(columns=[thisVar])
#             if 'id' in list(nodeDF):
#                 nodeDF['id'] = nodeDF['id'].astype(str)
#         writeGeoCSV(nodeDF, '../Data/MapImages/network_'+filename+'_nodes.csv')

#         if len(edgeDF) > 0:
#             for thisVar in ["geomDist", "lat", "lon", "geomMap"]:
#                 if thisVar in list(edgeDF):
#                     edgeDF = edgeDF.drop(columns=[thisVar])
#             if 'source' in list(edgeDF):
#                 edgeDF['source'] = edgeDF['source'].astype(str)
#             if 'target' in list(edgeDF):
#                 edgeDF['target'] = edgeDF['target'].astype(str)
#         writeGeoCSV(edgeDF, '../Data/MapImages/network_'+filename+'_edges.csv')


###=== Input a lat/lon and a radius and get a network covering the appropriate area
###=== The networkType can be "Full", "Walkability" or "StationFinder"
def getNetworkTilesForLatLon(thisLon, thisLat, radius=1500, networkFolder="fullNetwork_v1"):
#    startTime = time.time()
    ###=== enter safeguard for switching the order of inputs based on Japan:
    if thisLat > 90:  ## this is always impossible, so they must be switched, but great for Japan because lon is always > 90.
        tempLon = thisLat
        thisLat = thisLon
        thisLon = tempLon

    ###=== Load the network tile lookup file
    gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoArea-1500m.pkl')

#    ###=== get the x,y indices for the entered lon, lat
#    thisPoint = Point(thisLon, thisLat)
#    thisGrid = gridData[gridData.apply(lambda row: row['geometry'].intersects(thisPoint), axis=1)]  0.4s vs 0.04s with the four lat/on comparisons
    # startTime = time.time()
    thisGrid = gridData[((gridData['xMin'] < thisLon) & (gridData['yMin'] < thisLat) & (gridData['xMax'] > thisLon) & (gridData['yMax'] > thisLat))]
#    reportRunTime(startTime, "isolate center tile")
#    print(thisGrid.head())
#    print(thisGrid.xIndex)

#    startTime = time.time()
    if len(thisGrid) == 0:
        print("Coordinates are outside our covered area.")
        if ((networkType=="Walkability") | (networkType=="StationFinder") | (networkType=="SimplifiedWalkability")):
            thisNetwork =  nx.empty_graph(1,create_using=nx.Graph())
        else:
            thisNetwork =  nx.empty_graph(1,create_using=nx.DiGraph())
    else:
        ###=== Use the radius parameter to determine how many tiles in each direction are needed.
        indexDiff = math.ceil(radius / 1500)    ### handle the worse-case scenario, radius=0 gets just the center tile.
        ###=== Get the list of indexDiffs from the radius
        indexDiffsList = list(range(0 - indexDiff, indexDiff+1, 1))
        indexDiffArray = [(i,j) for i in indexDiffsList for j in indexDiffsList]
        indexDiffArray.remove((0,0))
#        print(indexDiffArray)
#        reportRunTime(startTime, "get indexDiffs")

        ###=== Start with the central grid
#        startTime = time.time()
#        print('networkGrid='+str(thisGrid.xIndex.values[0])+"_"+str(thisGrid.yIndex.values[0])+'.pkl')
        thisComboIndex = str(thisGrid.xIndex.values[0])+"_"+str(thisGrid.yIndex.values[0])
        thisNetwork = getNetworkTile(thisComboIndex, networkType=networkType, tileVersion=tileVersion)
#        reportRunTime(startTime, "get center network")

#        startTime = time.time()
        for thisDiff in indexDiffArray:
            thisComboIndex = str(thisGrid.xIndex.values[0] + thisDiff[0])+"_"+str(thisGrid.yIndex.values[0] + thisDiff[1])
            thisNetwork = nx.compose(thisNetwork, getNetworkTile(thisComboIndex, networkFolder=networkFolder))
#        reportRunTime(startTime, "get and merge surrounding tiles")

        return thisNetwork
#####---------------------------------------------------

###=== Input a polygon (such as a city or 23wards or some buffered thing) and get a network covering the appropriate area
###=== You need to specify which tile set to use by inputing the folder name
def getNetworkTilesForPolygon(thisGeom, networkFolder=None, inputGeomDist=True, isDirected=False, cropToPolygon=False):
    ###=== Load the network tile lookup file
    gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoArea-1500m.pkl')

#    ###=== get the list of grid x_y indices for the entered geometry
#    startTime = time.time()
    if inputGeomDist:
        gridData = gridData.loc[gridData['geomDist'].intersects(thisGeom)]
    else:
        gridData = gridData.loc[gridData['geometry'].intersects(thisGeom)]
#    reportRunTime(startTime, "get rows intersecting polygon")
#    print(thisGrid.head())
#    print(thisGrid.xIndex)

    if isDirected:
        thisNetwork =  nx.empty_graph(0,create_using=nx.DiGraph())
    else:
        thisNetwork =  nx.empty_graph(0,create_using=nx.Graph())

    if len(gridData) == 0:
        print("Region is outside our covered area.")
        return thisNetwork
    else:
        for index,thisRow in gridData.iterrows():
#            startTime = time.time()
            thisNetwork = nx.compose(thisNetwork, getNetworkTile(thisRow['comboIndex'], networkFolder=networkFolder))
#            reportRunTime(startTime, "get all network tiles")

        if cropToPolygon:
            nodesToKeep = flattenList([[u,v] for u,v,attr in thisNetwork.edges(data=True) if attr['geometry'].intersects(thisGeom)])
            thisNetwork = getSubgraph(thisNetwork, nodesToKeep)

        return thisNetwork
#####---------------------------------------------------


###=== Input a polygon (such as a ku or tile or some buffered point) and get all its buildings
def getBuildingTilesForGeomOld(thisGeom, inputGeomDist=True, cropToPolygon=False):
    # def csv_to_gdf(path):
    #     df = pd.read_csv(path)
    #     gdf = gp.GeoDataFrame(data=df, crs=standardCRS, geometry=[wkt.loads(g) for g in df['geometry']])
    #     if 'geomDist' in gdf.columns:
    #         gdf['geomDist'] = [wkt.loads(g) for g in df['geomDist']]
    #     return gdf

    ###=== Load the tile lookup file
    try:
        gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoMain-1500m.pkl')
    except:
        gridData = readGeoPandasCSV('../Data/OSMData/networkGridLookup-TokyoMain-1500m.csv')
    ###=== get the list of grid x_y indices for the entered geometry
    if inputGeomDist:
        # gridData = gridData.loc[gridData['geomDist'].intersects(thisGeom)]
        gridData = gridData[gridData['geomDist'].apply(lambda geom: geom.intersects(thisGeom))]
    else:
        gridData = gridData.loc[gridData['geometry'].intersects(thisGeom)]

    if len(gridData) == 0:
        print("== Region is outside our covered area.")
        buildingData = gp.GeoDataFrame()
    else:
        theseTiles = []
        for thisTile in list(gridData['comboIndex']):
        # for index, thisRow in gridData.iterrows():
            # thisTile = gridData.at[index,'comboIndex']
            try:
                thisGDF = readPickleFile("../Data/Kiban/BuildingTiles_Kiban/buildingData-Tile="+thisTile+".pkl")
                # print(thisGDF)
                # thisGDF['geomDist'] = [wkt.loads(g) for g in thisGDF['geomDist']]  ##--> this fixes the error from before
                # thisGDF.crs = None
                theseTiles.append(thisGDF)
                # theseTiles.append(readPickleFile("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".pkl"))
            except:
                ###--- pickle loading failed (pickle might not exist or some other error)
                # theseTiles.append(csv_to_gdf("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".csv"))
                # theseTiles.append(readGeoPandasCSV("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".csv"))
                print(" -- reading kiban building pkl file failed for",thisTile,"- no buildings")

        buildingData = pd.concat([data for data in theseTiles]).reset_index(drop=True)
        # drop duplicates ocurring from concatenating neighbouring buffered tiles
        buildingData = buildingData.drop_duplicates(subset=['geometry'])

        if cropToPolygon:
            if inputGeomDist:
                buildingData = buildingData[buildingData['geomDist'].intersects(thisGeom)]
            else:
                buildingData = buildingData[buildingData['geometry'].intersects(thisGeom)]

    return buildingData
#####---------------------------------------------------


###=== Input a polygon (such as a ku or tile or some buffered point) and get all its buildings
def getBuildingTilesForGeom(thisGeom, inputGeomDist=True, cropToPolygon=False):
    # def csv_to_gdf(path):
    #     df = pd.read_csv(path)
    #     gdf = gp.GeoDataFrame(data=df, crs=standardCRS, geometry=[wkt.loads(g) for g in df['geometry']])
    #     if 'geomDist' in gdf.columns:
    #         gdf['geomDist'] = [wkt.loads(g) for g in df['geomDist']]
    #     return gdf

    ###=== Load the tile lookup file
    try:
        gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoMain-1500m.pkl')
        if isinstance(gridData.iloc[0]['geomDist'], str):
            gridData['geomDist'] = gp.GeoSeries([wkt.loads(g) for g in gridData['geomDist']])
        if not isinstance(gridData['geomDist'], gp.GeoSeries):
            gridData['geomDist'] = gp.GeoSeries([g for g in gridData['geomDist']])
    except:
        gridData = readGeoPandasCSV('../Data/OSMData/networkGridLookup-TokyoMain-1500m.csv')
        if not isinstance(gridData['geomDist'], gp.GeoSeries):
            gridData['geomDist'] = gp.GeoSeries([g for g in gridData['geomDist']])
    ###=== get the list of grid x_y indices for the entered geometry
    if inputGeomDist:
        gridData = gridData.loc[gridData['geomDist'].intersects(thisGeom)]
    else:
        gridData = gridData.loc[gridData['geometry'].intersects(thisGeom)]

    if len(gridData) == 0:
        print("== Region is outside our covered area.")
        buildingData = gp.GeoDataFrame()
    else:
        theseTiles = []
        for index, thisRow in gridData.iterrows():
            thisTile = gridData.at[index,'comboIndex']
            try:
                thisGDF = readPickleFile("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".pkl")
                # if geomDist or geometry are not geometry objects but WKT strings, fix them:
                if isinstance(thisGDF.iloc[0]['geometry'], str):
                    thisGDF['geomDist'] = gp.GeoSeries([wkt.loads(g) for g in thisGDF['geomDist']])
                if isinstance(thisGDF.iloc[0]['geomDist'], str):
                    thisGDF['geomDist'] = gp.GeoSeries([wkt.loads(g) for g in thisGDF['geomDist']])  ##--> this fixes the error from before
                thisGDF.crs = None
                theseTiles.append(thisGDF)
                # theseTiles.append(readPickleFile("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".pkl"))
            except Exception as e:
                ###--- pickle loading failed (pickle might not exist or some other error)
                # theseTiles.append(csv_to_gdf("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".csv"))
                thisGDF = readGeoPandasCSV("../Data/Kiban/BuildingTiles_Full/buildingData-Tile="+thisTile+".csv")
                thisGDF.crs = None
                theseTiles.append(thisGDF)
                # print(f" -- reading building pkl file failed like: {e} --> used csv")

        buildingData = pd.concat([data for data in theseTiles]).reset_index(drop=True)
        # drop duplicates ocurring from concatenating neighbouring buffered tiles
        buildingData = buildingData.drop_duplicates(subset=['geometry'])

        if cropToPolygon:
            if inputGeomDist:
                buildingData = buildingData[buildingData['geomDist'].intersects(thisGeom)]
            else:
                buildingData = buildingData[buildingData['geometry'].intersects(thisGeom)]

    return buildingData
#####---------------------------------------------------



###=== Input a polygon (such as a ku or tile or some buffered point) and get all its buildings
def getKibanTilesForGeom(thisGeom, inputGeomDist=True, cropToPolygon=False):
    ###=== Load the tile lookup file
    try:
        gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoMain-1500m.pkl')
        if isinstance(gridData.iloc[0]['geomDist'], str):
            gridData['geomDist'] = gp.GeoSeries([wkt.loads(g) for g in gridData['geomDist']])
        if not isinstance(gridData['geomDist'], gp.GeoSeries):
            gridData['geomDist'] = gp.GeoSeries([g for g in gridData['geomDist']])
    except:
        gridData = readGeoPandasCSV('../Data/OSMData/networkGridLookup-TokyoMain-1500m.csv')
        if not isinstance(gridData['geomDist'], gp.GeoSeries):
            gridData['geomDist'] = gp.GeoSeries([g for g in gridData['geomDist']])

    ###=== get the list of grid x_y indices for the entered geometry
    if inputGeomDist:
        gridData = gridData.loc[gridData['geomDist'].intersects(thisGeom)]
    else:
        gridData = gridData.loc[gridData['geometry'].intersects(thisGeom)]

    if len(gridData) == 0:
        print("== Region is outside our covered area.")
        kibanData = gp.GeoDataFrame()
    else:
        theseTiles = []
        for index,thisRow in gridData.iterrows():
            thisTile = gridData.at[index,'comboIndex']
            # thisGDF = readPickleFile("../Data/Kiban/KibanTiles/kibanData-Tile="+thisTile+".pkl")
            # print(thisGDF.head())
            # theseTiles.append(thisGDF)
            theseTiles.append(readPickleFile("../Data/Kiban/KibanTiles/kibanData-Tile="+thisTile+".pkl"))

        kibanData = pd.concat([data for data in theseTiles]).reset_index(drop=True)

        if cropToPolygon:
            if inputGeomDist:
                kibanData = kibanData[kibanData['geomDist'].intersects(thisGeom)]
            else:
                kibanData = kibanData[kibanData['geometry'].intersects(thisGeom)]

    return kibanData
#####---------------------------------------------------















###=== This works in place to add geometries for a different CRS, the distance preserving one for now...replaced by using pyProj to convert geometries directly
#def addNewCRSGeoms(thisNetwork, newCRS):
#
#    ###=== Because we know these nodes all have geometries.
#    nodeData = [{'id': n[0], **n[1]} for n in thisNetwork.nodes(data=True)]
#    nodeDF = gp.GeoDataFrame({'geometry': [n['geometry'] for n in nodeData], 'id': [n['id'] for n in nodeData]})
#    nodeDF.crs = standardCRS
#    nodeDF = nodeDF.to_crs(newCRS)
#
#    def modifyNode(row):
#        x, y = list(row['geometry'].coords)[0]
#        nodeID = row['id']
#        thisNetwork.nodes[nodeID]['lonDist'] = x
#        thisNetwork.nodes[nodeID]['latDist'] = y
#        thisNetwork.nodes[nodeID]['geomDist'] = row['geometry']
#
#    nodeDF.apply(modifyNode, axis=1)
##    print("  -- all nodes processed")
#
#    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
#    edgesDF = gp.GeoDataFrame({'geometry': [e['geometry'] for e in edgesData]})
#    edgesDF.crs = standardCRS
#    edgesDF = edgesDF.to_crs(newCRS)
##    edgeBlockSize = makeInt(len(edgesDF) / 10)  ### print statement at 10% increments
#    for i in range(len(edgesDF)):
#        thisGeom = edgesDF.iloc[i, 0]
#        ###=== Some edges have point geometries, so this needs to handle both lines and points.
##        print(thisGeom)
#        try:
#            (x1, y1), (x2, y2) = thisGeom.coords
#        except:
#            x1, y1 = list(thisGeom.coords)[0]
#            x2, y2 = list(thisGeom.coords)[0]
#        edge = edgesData[i]
#        source = edge['source']
#        target = edge['target']
#        thisNetwork.edges[source, target]['x1Dist'] = x1
#        thisNetwork.edges[source, target]['y1Dist'] = y1
#        thisNetwork.edges[source, target]['x2Dist'] = x2
#        thisNetwork.edges[source, target]['y2Dist'] = y2
#        thisNetwork.edges[source, target]['geomDist'] = thisGeom

#        if i % edgeBlockSize == 0:
#            percentDone = rnd(100 * (i / len(edgesDF)), 1)
#            print(f"  -- {percentDone} % edges processed")


####=== This works in place to convert the CRS of a network...replaced by using pyProj to convert geometries directly
#def convertNetworkCRS(thisNetwork, fromCRS, toCRS):
#
#    ###=== Check how geometries are stored, but it really needs to be per node or it's not a real check.
#    allNodeAttributes = getAllNodeAttributes(thisNetwork)
#    if 'geometry' in allNodeAttributes:
#        nodeData = [{'id': n[0], **n[1]} for n in thisNetwork.nodes(data=True)]
#        nodeDF = gp.GeoDataFrame({'geometry': [n['geometry'] for n in nodeData], 'id': [n['id'] for n in nodeData]})
#    elif 'lon' in allNodeAttributes:
#        nodeData = [{'id': n[0], **n[1]} for n in thisNetwork.nodes(data=True)]
#        nodeDF = gp.GeoDataFrame({'geometry': [Point(n['lon'], n['lat']) for n in nodeData], 'id': [n['id'] for n in nodeData]})
#    elif 'x' in allNodeAttributes:
#        nodeData = [{'id': n[0], **n[1]} for n in thisNetwork.nodes(data=True)]
#        nodeDF = gp.GeoDataFrame({'geometry': [Point(n['x'], n['y']) for n in nodeData], 'id': [n['id'] for n in nodeData]})
#    else:
#        print("  !! Nodes have no geometry data.")
#    nodeDF.crs = fromCRS
#    nodeDF = nodeDF.to_crs(toCRS)
#
#    def modifyNode(row):
#        x, y = list(row['geometry'].coords)[0]
#        nodeID = row['id']
#        thisNetwork.nodes[nodeID]['lon'] = x
#        thisNetwork.nodes[nodeID]['lat'] = y
#
#    nodeDF.apply(modifyNode, axis=1)
#    # for i in range(len(nodeDF)):
#    #     x, y = list(nodeDF.iloc[i, 0].coords)[0]
#    #     nodeID = nodeDF.iloc[i, 1]
#    #     thisNetwork.nodes[nodeID]['lon'] = x
#    #     thisNetwork.nodes[nodeID]['lat'] = y
##    print("  -- all nodes processed")
#
#    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
#    # edgesDF = gp.GeoDataFrame([{'geometry': Point(e['x1'], e['y1']), 'target_geom': Point(e['x2'], e['y2'])} for e in edgesData])
#    if 'geometry' in getAllEdgeAttributes(thisNetwork):
#        edgesDF = gp.GeoDataFrame({'geometry': [e['geometry'] for e in edgesData]})
#    elif 'x1' in list(edgesData):
#        edgesDF = gp.GeoDataFrame({'geometry': [LineString([Point(e['x1'], e['y1']), Point(e['x2'], e['y2'])]) for e in edgesData]})
#    else:
#        print("  !! Edges have no geometry data.")
#    edgesDF.crs = fromCRS
#    edgesDF = edgesDF.to_crs(toCRS)
#    edgeBlockSize = makeInt(len(edgesDF) / 10)  ### print statement at 10% increments
#    for i in range(len(edgesDF)):
#        (x1, y1), (x2, y2) = edgesDF.iloc[i, 0].coords
#        edge = edgesData[i]
#        source = edge['source']
#        target = edge['target']
#        thisNetwork.edges[source, target]['x1'] = x1
#        thisNetwork.edges[source, target]['y1'] = y1
#        thisNetwork.edges[source, target]['x2'] = x2
#        thisNetwork.edges[source, target]['y2'] = y2
##        thisNetwork.nodes[source]['lon'] = x1
##        thisNetwork.nodes[source]['lat'] = y1
##        thisNetwork.nodes[target]['lon'] = x2
##        thisNetwork.nodes[target]['lat'] = y2
##
##        if i % edgeBlockSize == 0:
##            percentDone = rnd(100 * (i / len(edgesDF)), 1)
##            print(f"  -- {percentDone} % edges processed")

# Input a network and some point data (in pandas DF) with lat/lon columns
# Return the edge geometry STRTree and the GeoDataFrame with a given buffer
def geomize(thisNetwork, storeData, thisBuffer=125):
    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
    edgesDF = gp.GeoDataFrame({'geometry': [LineString([Point(e['x1'], e['y1']), Point(e['x2'], e['y2'])]) for e in edgesData],
                                                         'source|target': [str(e['source']) + "|" + str(e['target']) for e in edgesData]})

    edgesGeoms = list(edgesDF['geometry'])
    geomsIdx = list(edgesDF['source|target'])
    for index, geom in enumerate(edgesGeoms):
        geom.idx = geomsIdx[index]
    geomTree = STRtree(edgesGeoms)

    storeData.loc[:, 'geometry'] = storeData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    storeData = gp.GeoDataFrame(storeData)
    storeData.crs = standardCRS
    storeData = storeData.to_crs(areaCalcCRS)
    storeData.loc[:, 'geometry'] = storeData.geometry.map(lambda geom: geom.buffer(thisBuffer))

    storeGeoms = createGeomData(storeData)

    storeData.loc[:, 'lon'] = storeData.geometry.map(lambda x: list(x.centroid.coords)[0][0])
    storeData.loc[:, 'lat'] = storeData.geometry.map(lambda x: list(x.centroid.coords)[0][1])

    return geomTree, storeGeoms, storeData

###=== from https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def nearestPointOnLine(startNodePoint, endNodePoint, otherPoint):
    # Returns the nearest point on a given line and its distance
    x1, y1 = startNodePoint
    x2, y2 = endNodePoint
    x3, y3 = otherPoint
    if startNodePoint == endNodePoint:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)
#        return x1, y1, haversineDist(x1, y1, x3, y3)

    ### this may not work here because coords are in degrees, so the delta values for x and y are not commensurable
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = ( (dy * (y3 - y1)) + (dx * (x3 - x1)) ) / det
    # Corner cases
    if a >= 1:
        return x2, y2, euclideanDistance(x2, y2, x3, y3)
#        return x2, y2, haversineDist(x2, y2, x3, y3)
    elif a <= 0:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)
#        return x1, y1, haversineDist(x1, y1, x3, y3)
    newpx = x1 + a * dx
    newpy = y1 + a * dy
    return newpx, newpy, euclideanDistance(newpx, newpy, x3, y3)
#    return newpx, newpy, haversineDist(newpx, newpy, x3, y3)

def findNearestEdge(potentialNearestEdges, otherPoint):
    # Use the min heap to get the nearest edge --> log(N) time query (but still takes ~206 hours)
    otherPoint = list(otherPoint.centroid.coords)[0]
    # start = time.time()
    heap = []
    for edge in potentialNearestEdges:
        startNodeID = edge.idx.split('|')[0]
        try:
            startNodeID = int(startNodeID)  ## some nodeIDs are ints, some are str, so try to convert to int, but if that fails, leave as string.
        except:
            pass

        endNodeID = edge.idx.split('|')[1]
        try:
            endNodeID = int(endNodeID)
        except:
            pass
        startNodeCoords, endNodeCoords = list(edge.coords)
        x3, y3, dist = nearestPointOnLine(startNodeCoords, endNodeCoords, otherPoint)
        heapq.heappush(heap, (dist, [(x3, y3), startNodeCoords, endNodeCoords, startNodeID, endNodeID]))
    try:
        theNode_endpoints = heap[0]
    except Exception as e:
        print(e)
        print(otherPoint)
    return theNode_endpoints


def createEdgeGeomTree(thisNetwork):
    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
    edgesDF = gp.GeoDataFrame({'geometry': [LineString([Point(e['x1'], e['y1']), Point(e['x2'], e['y2'])]) for e in edgesData],
                                                         'source|target': [str(e['source']) + "|" + str(e['target']) for e in edgesData]})
    edgesGeoms = list(edgesDF['geometry'])
    geomsIdx = list(edgesDF['source|target'])
    for index, geom in enumerate(edgesGeoms):
        geom.idx = geomsIdx[index]
    geomTree = STRtree(edgesGeoms)
    return geomTree


#def findNearestEdgeFromNetwork(thisNetwork, storeData):
#    start = time.time()
#
#    # buffer = 125 wasn't big enough...
#    geomTree, storeGeoms, geoStoreData = geomize(thisNetwork, storeData, thisBuffer=100)
#    N = storeData.shape[0]
#    savestep = int(N * 0.01)
#
#    print("Preprocessing done!")
#
#    selectedStores = set()
#    nodeID = max(thisNetwork.nodes()) + 1
#    for i, thisGeom in enumerate(storeGeoms):
#        theStoreCoords = list(thisGeom.centroid.coords)[0]
#        # theStoreData = geoStoreData.loc[(geoStoreData.lon - theStoreCoords[0] < 1e-5) & (geoStoreData.lat - theStoreCoords[1] < 1e-5)]
#        theStoreData = geoStoreData.loc[(~geoStoreData["index"].isin(selectedStores)) & (geoStoreData.lon == theStoreCoords[0]) & (geoStoreData.lat == theStoreCoords[1])]
#        if theStoreData.shape[0] == 0:
#            continue # Skip this iteration since the location was chosen already
#            # print("No store found. Using the index..")
#            # theStoreData = geoStoreData.iloc[thisGeom.idx, :]
#        for idx in theStoreData["index"]:
#            selectedStores.add(idx)
#        # print(f"# of stores at location {theStoreCoords}: {theStoreData.shape[0]}")
#        # Turn the geoDF into a list of dicts
#        theStoreData = theStoreData.drop(columns=['index', 'geometry', 'lat', 'lon']).to_dict('records')
#        theStoreData = {attr: [storeDict[attr] for storeDict in theStoreData] for attr in theStoreData[0].keys()}
#
#        potentialNearestEdges = geomTree.query(thisGeom)
#        if len(potentialNearestEdges) == 0:
#            print("Potential Nearest Edges:", len(potentialNearestEdges))
#            thisGeom = Point(theStoreCoords[0], theStoreCoords[1]).buffer(10000)
#            potentialNearestEdges = geomTree.query(thisGeom)
#            print("Augmented:", len(potentialNearestEdges))
#        storeDist, theNode_endpoints = findNearestEdge(potentialNearestEdges, thisGeom)
#        theNode, startNode, endNode, startNodeID, endNodeID = theNode_endpoints
#
#        # Add the store node
#        storeNodeID = nodeID + 1
#        thisNetwork.add_node(
#            storeNodeID,
#            lon=theStoreCoords[0],
#            lat=theStoreCoords[1],
#            modality='store',
#            storeDist=storeDist,
#            **theStoreData
#        )
#
#        # Add the store access edge
#        if theNode == startNode:
#                thisNetwork.add_edge(startNodeID, storeNodeID,
#                                     modality='store',
#                                     distance=storeDist,
#                                     x1=startNode[0],
#                                     y1=startNode[1],
#                                     x2=theStoreCoords[0],
#                                     y2=theStoreCoords[1],
#                                     elevationGain=0)
#                thisNetwork.add_edge(storeNodeID, startNodeID,
#                                     modality='store',
#                                     distance=storeDist,
#                                     x1=theStoreCoords[0],
#                                     y1=theStoreCoords[1],
#                                     x2=startNode[0],
#                                     y2=startNode[1],
#                                     elevationGain=0)
##                 nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[startNodeID]['elevation']}})
#        elif theNode == endNode:
#                thisNetwork.add_edge(endNodeID, storeNodeID,
#                                     modality='store',
#                                     distance=storeDist,
#                                     x1=endNode[0],
#                                     y1=endNode[1],
#                                     x2=theStoreCoords[0],
#                                     y2=theStoreCoords[1],
#                                     elevationGain=0)
#                thisNetwork.add_edge(storeNodeID, endNodeID,
#                                     modality='store',
#                                     distance=storeDist,
#                                     x1=theStoreCoords[0],
#                                     y1=theStoreCoords[1],
#                                     x2=endNode[0],
#                                     y2=endNode[1],
#                                     elevationGain=0)
##                 nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[endNodeID]['elevation']}})
#        else:
#            # This part still needs updates on distances and elevations
#            thisNetwork.add_node(
#                nodeID,
#                lon=theNode[0],
#                lat=theNode[1],
#                parentSourceID=startNodeID,
#                parentTargetID=endNodeID,
#                parentSource=startNode,
#                parentTarget=endNode
#            )
#            thisNetwork.add_edge(storeNodeID, nodeID,
#                                 modality='store',
#                                 distance=storeDist,
#                                 x1=theStoreCoords[0],
#                                 y1=theStoreCoords[1],
#                                 x2=theNode[0],
#                                 y2=theNode[1],
#                                 elevationGain=0)
#            thisNetwork.add_edge(nodeID, storeNodeID,
#                                 modality='store',
#                                 distance=storeDist,
#                                 x1=theNode[0],
#                                 y1=theNode[1],
#                                 x2=theStoreCoords[0],
#                                 y2=theStoreCoords[1],
#                                 elevationGain=0)
##             nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[endNodeID]['elevation']}})
#        nodeID += 2
#        if i % savestep == 0:
#            print(f"Processed {i} rows. {i//savestep}% done in {time.time() - start} seconds.")
#
#    end = time.time()
#    print(f"Total time: {end - start} seconds")


def mergeTwoNetworks(base_network, G, inplace=True):
    """
    The function to merge two networks, which is faster than nx.compose(G, H).
    But, this function update base_network in-place, so you have to use nx.compose if you want to keep original networks.
    Also, you should use ...
    """
    if inplace is True:
        if base_network.is_multigraph():
            base_network.add_nodes_from(G.nodes(keys=True, data=True))
            base_network.add_edges_from(G.edges(keys=True, data=True))
        else:
            base_network.add_nodes_from(G.nodes(data=True))
            base_network.add_edges_from(G.edges(data=True))
    else:
        return nx.compose(base_network, G)


###=== Get the nodes and edges from the database, and combine them into the networkX network)
def networkFromDataframes(nodeDF, edgeDF, source='source', target='target', nodeID='id', graphType=nx.DiGraph):
    edgeDF[source] = edgeDF[source].map(str)
    edgeDF[target] = edgeDF[target].map(str)
    nodeDF[nodeID] = nodeDF[nodeID].map(str)

    edgeAttrToKeep = [att for att in list(edgeDF) if att not in [source,target] ]

    thisNetwork = nx.from_pandas_edgelist(edgeDF, source=source, target=target, edge_attr=edgeAttrToKeep, create_using=graphType)

    for n1,n2,attr in thisNetwork.edges(data=True):  ## Remove None attributes from edges
        attrToRemove = [att for att in attr if attr.get(att, None) == None]
        for att in attrToRemove:
            attr.pop(att, None)

    # print("Number of nodes from edges:", len(thisNetwork.nodes()))

    nodeDF = nodeDF.drop_duplicates(subset=[nodeID]) ##-- remove duplicate nodes that may be in database
    # print("Number of nodes from nodeDF:", len(nodeDF))

    node_attr = nodeDF.set_index(nodeID).to_dict('index')
    theseNodes = [(node,{k:v for k,v in attr.items() if v != None}) for node,attr in node_attr.items()]
    thisNetwork.add_nodes_from(theseNodes)

    return thisNetwork

####========================================================================
####============================ SCORING HELPERS ===========================
####========================================================================

####=== Function that returns a weighted value that is scaled based on its proportion of the scalingLimit and the curvature.
####=== "value" is the quantity being scaled, "scalingFactor" is the feature that effects scaling (e.g. distance).
####=== When the scaling factor = 0, the weighted value == value, and when the scalingFactor = scalingLimit, weighted value == 0.
####=== When the curavture < 1 the weights decrease sharply, and values > 1 postpone the drop-off.

def weightedValue(scalingFactor, value=1, curvature=1, scalingLimit=60):
    # Use np.where for element-wise comparison and selection
    return np.where(scalingFactor > scalingLimit, 0,
                    value * (1 / (1 + np.exp(curvature * (scalingFactor - scalingLimit/2)))))



def makeFunctionPlot():
    plt.rcParams["font.family"] = "Palatino Linotype"
    fig = plt.figure(figsize=(5,3))
    ax = plt.axes()
    x = np.linspace(0, 60, 1000)
    ax.plot(x, weightedValue(x, curvature = 2), color='#4292c6', label=r'$\lambda = 2$')
    ax.plot(x, weightedValue(x, curvature = 1), color='#969696', label=r'$\lambda = 1$')
    ax.plot(x, weightedValue(x, curvature = 0.5), color='#cb181d', label=r'$\lambda = 0.5$')
    ax.plot(x, weightedValue(x, curvature = 0), color='#939799', label='linear')
    plt.xlabel("time from source (minutes)", fontsize=12)
    plt.ylabel("weight", fontsize=12)
    plt.grid()
    plt.legend()
    fileIndex = 1
    fig.savefig("../DemandEstimation/"+"weightFunction"+"-Plot"+str(fileIndex)+".png", dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0)



# def propertyDiscountFunction(currentAge, currentRent, stabilityAge=25, decayRate=1.0):
#     plt.rcParams["font.family"] = "Palatino Linotype"
#     fig = plt.figure(figsize=(5,3))
#     ax = plt.axes()
#     x = np.linspace(0, 60, 1000)

#     if currentAge >= stabilityAge:
#         # print("Flat line")
#         ax.plot(x, , color='#939799', label='linear')
#     else:











####========================================================================
####============================ ANALYSIS HELPERS ===========================
####========================================================================


####=========================== Correlation and Similarity Analyses ==========================

from difflib import SequenceMatcher

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
# from scipy.stats import rankdata
# import nltk

def safeDivisionLists(n, d, infValue=None):
    n = list(n) if not isinstance(n,list) else n
    d = list(d) if not isinstance(d,list) else d
    if len(n) == len(d):
        return [safeDivision(n[i],d[i],infValue=infValue) for i in range(len(n))]
    else:
        print("  !! Lists must be the same length")



####=== CHECK FOR DIVIDE BY ZERO
def safeDivision(n, d, infValue=None):
    if n == 0:
        return 0
    elif d > 0:
        return n / d
    elif d == 0:
        if infValue != None:
            return infValue
        else:
            return np.inf
    else:
        return 0


###=== a log function that handles corner cases in a dafe way for scoring
def safeLog(values, undef=-0.693):
    if not isinstance(values, list):
        logVal = np.log(values) if values > 0.001 else undef
    else:
        logVal = [np.log(x) if x > 0.001 else undef for x in values]
    return logVal


##== Calculation the Inverse Simpson's Index from a list of group membership values
def effectiveNumberOfGroups(listOfGroupNumbers):
    totalNum = len(listOfGroupNumbers)
    uniqueValues = list(set(listOfGroupNumbers))
    effectiveNumber = 0
    for thisValue in uniqueValues:
        thisNumMembers = sum(1 for p in listOfGroupNumbers if p == thisValue) / totalNum
        effectiveNumber += thisNumMembers ** 2
    return np.round(1 / effectiveNumber, decimals=2)

##== The Pearson's correlation coefficient for two Pandas columns
####-- The adjusted mutual information of two lists of categorical info to measure similarity of groupings/clusterings
####-- This works directly on pandas columns without conversion
#def mutualInfo(list1,list2):
#    return np.round(AMI(list1,list2), decimals=3)

def listDiff(list1, list2):
    return [i for i in list1 + list2 if i not in list1 or i not in list2]

def euclideanVectorDistance(list1, list2):
	return math.sqrt(sum((e1-e2)**2 for e1, e2 in zip(list1, list2)))

##-- Find closest point to a given point among a list of points.
def closest_point(thisPoint, listOfPoints):
    return listOfPoints[cdist([thisPoint], listOfPoints).argmin()]

###--- get the mean squared error of two value lists
def MSE(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    return np.mean([val**2 for val in (list1 - list2)])

def MAPE(trueVals, predVals, decimals=3):
    if not isinstance(trueVals, np.ndarray):
        trueVals = trueVals.detach().numpy()
    if not isinstance(predVals, np.ndarray):
        predVals = predVals.detach().numpy()
    return rnd(100 * mean_absolute_percentage_error(trueVals, predVals), decimals)
    # return rnd(np.mean(np.abs((trueVals - predVals) / trueVals)) * 100, 3)

def MAE(trueVals, predVals, decimals=3):
    if not isinstance(trueVals, np.ndarray):
        trueVals = trueVals.detach().numpy()
    if not isinstance(predVals, np.ndarray):
        predVals = predVals.detach().numpy()
    return rnd(mean_absolute_error(trueVals, predVals), decimals)

## Do regression and plot and GoF analysis?
# from sklearn.linear_model import LinearRegression
def getR2(trueVals, predVals, decimals=3):
    if not isinstance(trueVals, np.ndarray):
        trueVals = trueVals.detach().numpy()
    if not isinstance(predVals, np.ndarray):
        predVals = predVals.detach().numpy()
    return rnd(r2_score(trueVals, predVals), decimals)
    # predCol, inputCols = np.array(predCol).reshape(-1, 1), np.array(inputCols).reshape(-1, 1)
    # regressionEqn = LinearRegression().fit(inputCols, predCol)
    # return np.round(regressionEqn.score(inputCols, predCol), decimals=3)

def getCorrelation(list1, list2):
    if ((isinstance(list1, list)) & (isinstance(list2, list))):
        return np.round(np.corrcoef(list1, list2)[0, 1], decimals=3)
    else:
        return np.round(np.corrcoef(list(list1), list(list2))[0, 1], decimals=3)

def pearsonCorr(list1, list2):
    return np.round(pearsonr(list1, list2), decimals=3)

def spearmanCorr(list1, list2):
    return np.round(spearmanr(list1, list2), decimals=3)

def kendallCorr(list1, list2):
    return np.round(kendalltau(list1, list2), decimals=3)

##--- rank order edit distance
def sequenceMatch(list1, list2):
    return np.round(SequenceMatcher(None, list1, list2).ratio(), decimals=3)


# ##--- rank order edit distance
# def editDist(list1, list2):
#     return np.round(nltk.edit_distance(list1, list2, transpositions=True), decimals=3)

# def editSimilarity(list1, list2):
#     return 1 - (editDist(list1, list2) / len(list1))

####================================ VARIABLE CORRELATION MATRIX ==================================
# Makes an Array plot
def makeArrayPlot(thisData, variableLabels, titleText, figSize=5, filename=None):

    correlationColorMap = makeColorMap([-1,0,1],[normRGB(234,99,99),normRGB(245,245,245),normRGB(99,131,234)])  ##=== from red at -1 to blue at 1
    # correlationColorMap = makeColorMap([-1,.75,1],[normRGB(234,99,99),normRGB(245,245,245),normRGB(99,131,234)])   ##=== show more contrast at high levels
    # correlationColorMap = makeColorMap([-1,.95,1],[normRGB(234,99,99),normRGB(245,245,245),normRGB(99,131,234)])   ##=== show more contrast at high levels

    plt.rcParams["font.family"] = "Palatino Linotype"
    fig, ax = plt.subplots(figsize=(figSize, figSize))
#    ax.matshow(thisData, cmap='coolwarm_r', vmin=-1, vmax=1)
    ax.matshow(thisData, cmap=correlationColorMap, vmin=-1, vmax=1)
    plt.xticks(range(len(variableLabels)), variableLabels, rotation='vertical', fontsize=14)
    plt.yticks(range(len(variableLabels)), variableLabels, fontsize=14)
    plt.title(titleText, fontsize=16, y=-0.13)
    for (i, j), z in np.ndenumerate(thisData):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize=13)
    plt.show()
    if filename != None:
        fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')



###--------------------------------------------------------
def makeScatterPlot(xData, yData, xLabel, yLabel, titleText=None, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=None, diagonalLine=False, plotRange=None, filename=None):
    plt.rcParams["font.family"] = "Palatino Linotype"
    fig, ax = plt.subplots(figsize=(figSize, figSize))
    titleText = titleText if titleText != None else "Relationship of "+xLabel.title()+" and "+yLabel.title()
    colorScheme = colorScheme if colorScheme != None else np.array([normRGB(66,146,198)])

    xMin = np.nanmin(xData)
    xMax = np.nanmax(xData)
    yMin = np.nanmin(yData)
    yMax = np.nanmax(yData)

    minVal = min([xMin,yMin])
    maxVal = max([xMax,yMax])

    ##-- plotrange is a list of four values [xMin, xMax, yMin, yMax].  Setting any value to None will adjust it to the data
    if plotRange != None:
        xMin = xMin if plotRange[0] == None else plotRange[0]
        xMax = xMax if plotRange[1] == None else plotRange[1]
        yMin = yMin if plotRange[2] == None else plotRange[2]
        yMax = yMax if plotRange[3] == None else plotRange[3]

    plt.grid()
    if useCircles:
        plt.scatter(xData, yData, s=dotSize, c=colorScheme, alpha=0.8, marker='o', facecolors='none', edgecolors=colorScheme)
    else:
        plt.scatter(xData, yData, s=dotSize, c=colorScheme, alpha=0.5)
    plt.title(titleText, fontsize=16, y=1)
    plt.xlabel(xLabel, fontsize=14)
    plt.ylabel(yLabel, fontsize=14)
    if scaledAxes:
        # plt.plot([minVal,maxVal],[minVal,maxVal], c=myGrayMed, alpha=0.2)
        plt.axis('square')  ##-- use this to fix the ratio of x and y axes
        ax.set_xlim([minVal, maxVal])
        ax.set_ylim([minVal, maxVal])
    else:
        ax.set_xlim([xMin, xMax])
        ax.set_ylim([yMin, yMax])

    if bestFit != None:
        if not isinstance(bestFit, list):
            bestFit = list(bestFit)
        Xs = np.linspace(xMin,xMax)
        lineColors = [darkRedJet, darkRedTab, darkPinkTab, darkOrangeJet, darkPurpleJet, darkGrayJet, myAlmostBlack]
        for index,degree in enumerate(bestFit):
            lineColor = lineColors[index] if index < len(lineColors) else myAlmostBlack
            params = Polynomial.fit(xData, yData, degree).convert().coef
            Ys = np.sum([(params[i] * Xs**i) for i in range(degree + 1)], axis=0)
            ax.plot(Xs, Ys, c=lineColor, alpha=0.8, lw=0.5, zorder=5)

    if diagonalLine == True:
        Xs = np.linspace(minVal,maxVal)
        ax.plot(Xs, Xs, c='tab:gray', lw=1, zorder=2)

        # ###--- Add lines between points and best fit line (not converted from StackExchange yet
        # from matplotlib import collections  as mc
        # lines = [[(i,j), (i,i*a+b)] for i,j in zip(X,Y)]
        # lc = mc.LineCollection(lines, colors='grey', linewidths=1, zorder=1)
        # ax.add_collection(lc)


    plt.show()
    if filename != None:
        fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')


###--------------------------------------------------------
###--- A density plotfor high density scatterplot needs.  Specify the maxCount to change the colormap towards that value
def makeDensityPlot(xData, yData, xLabel, yLabel, titleText=None, numBins=100, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=None, diagonalLine=False, plotRange=None, addMargin=False, filename=None):

    scaledAxes = True if (diagonalLine == True) else scaledAxes

    xMin = np.nanmin(xData)
    xMax = np.nanmax(xData)
    yMin = np.nanmin(yData)
    yMax = np.nanmax(yData)

    if ((xMin == xMax) | (yMin == yMax)):
        print("    !! Cannot plot because either xMin == xMax or yMin == yMax", xMin, '=', xMax, 'or', yMin, '=',yMax)
    else:
        if addMargin != None:
            xBuffer = (xMax - xMin) * addMargin
            yBuffer = (yMax - yMin) * addMargin
            xMin = xMin - xBuffer
            xMax = xMax + xBuffer
            yMin = yMin - yBuffer
            yMax = yMax + yBuffer

        minVal = min([xMin,yMin])
        maxVal = max([xMax,yMax])

        if scaledAxes:
            histData, xedges, yedges = np.histogram2d(xData, yData, bins=numBins, range=[[minVal, maxVal], [minVal, maxVal]])
        else:
            histData, xedges, yedges = np.histogram2d(xData, yData, bins=numBins)
        histData = histData.T   ##-- does not follow Cartesian convention, therefore transpose H for visualization purposes.
        # print(H)
        maxBinVal = np.max(histData)

        if maxCount == None:
            maxCount = maxBinVal
        else:
            histData = np.clip(histData, 0, maxCount)
            # print(myHist)
            # print(maxBinVal)

        plt.rcParams["font.family"] = "Palatino Linotype"
        fig, ax = plt.subplots(figsize=(figSize, figSize))
        titleText = titleText if titleText != None else "Relationship of "+xLabel.title()+" and "+yLabel.title()

        ###--- this is a colormap for 2D histogram data
        colorInflectionPoint = min([maxCount/4, 50])
        colorScheme = colorScheme if colorScheme != None else makeColorMap([0,1,colorInflectionPoint,maxCount], [myPureWhite,myBlueLight,myBlueMed,myBlueDark], numVals=maxCount)

        X,Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, histData, cmap=colorScheme)

        plt.grid()
        plt.title(titleText, fontsize=16, y=1)
        plt.xlabel(xLabel, fontsize=14)
        plt.ylabel(yLabel, fontsize=14)
        if scaledAxes:
            # plt.plot([minVal,maxVal],[minVal,maxVal], c=myGrayMed, alpha=0.2)
            plt.axis('square')  ##-- use this to fix the ratio of x and y axes
            ax.set_xlim([minVal, maxVal])
            ax.set_ylim([minVal, maxVal])
        else:
            ax.set_xlim([xMin, xMax])
            ax.set_ylim([yMin, yMax])

        ##-- plotrange is a list of four values [xMin, xMax, yMin, yMax].  Setting any value to None will adjust it to the data
        if plotRange != None:
            xMin = xMin if plotRange[0] == None else plotRange[0]
            xMax = xMax if plotRange[1] == None else plotRange[1]
            yMin = yMin if plotRange[2] == None else plotRange[2]
            yMax = yMax if plotRange[3] == None else plotRange[3]

        if bestFit != None:
            if not isinstance(bestFit, list):
                bestFit = list(bestFit)
            Xs = np.linspace(xMin,xMax)
            lineColors = [darkRedJet, darkRedTab, darkPinkTab, darkOrangeJet, darkPurpleJet, darkGrayJet, myAlmostBlack]
            for index,degree in enumerate(bestFit):
                lineColor = lineColors[index] if index < len(lineColors) else myAlmostBlack
                params = Polynomial.fit(xData, yData, degree).convert().coef
                Ys = np.sum([(params[i] * Xs**i) for i in range(degree + 1)], axis=0)
                ax.plot(Xs, Ys, c=lineColor, alpha=0.8, lw=0.5, zorder=5)

        if diagonalLine == True:
            Xs = np.linspace(minVal,maxVal)
            ax.plot(Xs, Xs, c='tab:gray', lw=1, zorder=2)

        plt.show()
        if filename != None:
            fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')




###--------------------------------------------------------
###=== Standard 1D histogram of a list of data
def makeHistogramPlot(theData, xLabel, yLabel=None, titleText=None, numBin=20, colorScheme=None, figSize=5, lowerBound=None, upperBound=None, centerAtZero=False, filename=None):

    plt.rcParams["font.family"] = "Palatino Linotype"
    fig, ax = plt.subplots(figsize=(figSize, figSize))
    titleText = titleText if titleText != None else "Distribution of "+xLabel.title()
    colorScheme = colorScheme if colorScheme != None else myBlueMed

    xMin = min(theData)
    xMax = max(theData)
    lowerBound = xMin if lowerBound==None else lowerBound
    upperBound = xMax if upperBound==None else upperBound

    ###-- center at 0:
    if centerAtZero:
        largestABS = max([abs(lowerBound),abs(upperBound)])
        lowerBound = 0 - largestABS
        upperBound = largestABS

    plt.hist(theData, numBin)

    yLabel = 'count' if yLabel == None else yLabel
    plt.title(titleText, fontsize=16, y=1)
    plt.xlabel(xLabel, fontsize=14)
    plt.ylabel(yLabel, fontsize=14)


    ax.set_xlim([lowerBound, upperBound])

    plt.show()
    if filename != None:
        fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')





# ###--------------------------------------------------------
# ###---- Feed in data as a 2D array: numCategories x numDatapoints (all collections must be of the same length)
# def valueDistPlot(thisData,thisVariable,thisplotTitle=''):
#     fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#     for columnIndex, thisData in enumerate(allData):
#         jitterWidth = 0.6
# #        kde = gaussian_kde(thisData)
# #        density = kde(thisData)     # estimate the local density at each datapoint
#         jitter = np.random.rand(1,len(thisData)) - 0.5 # generate some random jitter between -0.5 and 0.5
#         xvals = columnIndex + (jitter * jitterWidth * 1)     # scale the jitter by the KDE estimate and add it to the centre x-coordinate
# #        xvals = columnIndex + (density * jitter * jitterWidth * 20000)     # scale the jitter by the KDE estimate and add it to the centre x-coordinate
#         ax.scatter(xvals, thisData, marker='o', s=30, facecolors='none', edgecolors='dodgerblue', alpha = 0.65)

#     ax.tick_params(right=True)
#     ax.set_xticks(range(len(allData)))
#     ax.set_xticklabels(networkLabels, fontname="Arial", fontsize=14)
#     ax.set_ylabel(variableLabel(thisVariable), fontname="Arial", fontsize=14)
#     plt.title(radiusLabel(thisRadius), fontname="Arial", fontsize=14)
#     fig.tight_layout()
#     plt.show()
#     fig.savefig("Graph Machine Learning/AnalysisOutput/VariablePlots/variablePlot-" + thisVariable + "-" + thisRadius + ".pdf", dpi=300, transparent=True)



###-----------------------------------------------------------
###=== Sort the rows/columns of a correlation matrix by similarity to find blocks
###--- from https://stackoverflow.com/questions/64248850/sort-simmilarity-matrix-according-to-plot-colors
def sortMatrixByValues(thisMatrix):
    idx = [np.argmax(np.sum(thisMatrix, axis=1))]  # a
    for i in range(1, len(thisMatrix)):
        thisMatrix_i = thisMatrix[idx[-1]].copy()
        thisMatrix_i[idx] = -1
        idx.append(np.argmax(thisMatrix_i))  # b
    return np.array(idx)

def getSimilarityMatrix(thisMatrix):
    return 1 / (1 + distance_matrix(thisMatrix, thisMatrix))
    # return 1 / (np.linalg.norm(thisMatrix[:, np.newaxis] - thisMatrix[np.newaxis], axis=-1) + 1 / len(thisMatrix))

def createCorrelationMatrix(thisData, rowNames, columnNames=None, rowLabels=None, columnLabels=None, sortByVar=None, fileLocation="../Conference Materials/", dataName="", fileIndex=1, fileType="both", xSize=8, ySize=8, numDigits=3):
    columnNames = rowNames if columnNames == None else columnNames
    rowLabels = rowNames if rowLabels == None else rowLabels
    columnLabels = columnNames if columnLabels == None else columnLabels
    dataName = dataName if dataName == "" else dataName+"_"

    ###=== Do Pearson Correlation
    pearsonMatrix = np.array([pearsonCorr(thisData[x], thisData[y])[0] for x in rowNames for y in columnNames]).reshape(len(rowNames), len(columnNames))

    ###=== change this from sorting by chosen column to finding clusters in the data
    ###--https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    if sortByVar != None:

        ###--- sorts by the similarity of the correaltion values (not correlation directly), but results aren't always better
        # pearsonMatrix2 = getSimilarityMatrix(pearsonMatrix)
        # idx2 = sortMatrixByValues(pearsonMatrix2)
        # pearsonMatrix = pearsonMatrix[idx2, :][:, idx2]
        # rowLabels = [rowLabels[i] for i in idx2]
        # columnLabels = [columnLabels[i] for i in idx2]

        ###--- the simple version from SE, but it looks better in some cases
        idx = sortMatrixByValues(pearsonMatrix)
        pearsonMatrix = pearsonMatrix[idx, :][:, idx]
        rowLabels = [rowLabels[i] for i in idx]
        columnLabels = [columnLabels[i] for i in idx]

        ###--- my code that sorts by correlation to a particular variable (used for regression analysis)
        # thisPosition = rowNames.index(sortByVar)
        # thisSorting = np.flip(pearsonMatrix[:, thisPosition].argsort())
        # # print(thisSorting)
        # pearsonMatrix = pearsonMatrix[:, thisSorting]
        # pearsonMatrix = pearsonMatrix[thisSorting, :]
        # rowLabels = [rowLabels[i] for i in thisSorting]
        # columnLabels = [columnLabels[i] for i in thisSorting]
        # columnLabels = columnLabels[thisSorting]

    correlationColorMap = makeColorMap([-1,0,1],[normRGB(234,99,99),normRGB(245,245,245),normRGB(99,131,234)])  ##=== from red at -1 to blue at 1

    plt.rcParams["font.family"] = "Palatino Linotype"
    fig, ax = plt.subplots(figsize=(xSize, ySize))
#    ax.matshow(thisData, cmap='coolwarm_r', vmin=-1, vmax=1)
    ax.matshow(pearsonMatrix, cmap=correlationColorMap, vmin=-1, vmax=1)
    plt.xticks(range(len(columnLabels)), columnLabels, rotation='vertical', fontsize=14)
    plt.yticks(range(len(rowLabels)), rowLabels, fontsize=14)
    titleText = "Pearson Correlation Matrix"
    # plt.title(titleText, fontsize=16, y=.1)
    for (i, j), z in np.ndenumerate(pearsonMatrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=13)
    plt.show()
    if fileType != None:
        # filename = fileLocation+dataName+"arrayPlot_pearson"+"-v"+str(fileIndex)+fileType
        filename = fileLocation+dataName+"arrayPlot_pearson"+"-v"+str(fileIndex)
        if fileType == 'pdf':
            fig.savefig(filename+'.pdf', dpi=300, transparent=True, bbox_inches='tight')
        if fileType == 'png':
            fig.savefig(filename+'.png', dpi=300, transparent=True, bbox_inches='tight')
        if fileType == 'both':
            fig.savefig(filename+'.pdf', dpi=300, transparent=True, bbox_inches='tight')
            fig.savefig(filename+'.png', dpi=300, transparent=True, bbox_inches='tight')




###--------------------------------------------------------
def performSimilarityAnalysis(df, columnNames, variableLabels=None, fileLocation="../Conference Materials/CCS2021-Main/Images/", dataName="", fileIndex=1, fileType=".png"):
    variableLabels = columnNames if variableLabels == None else variableLabels
    dataName = dataName if dataName == "" else dataName+"_"

#    ###=== Only do unique pairs ignoring order: eliminating duplicates yields faster processing, but harder to reshape into array.
#    allUniquePairs = [f(columnNames[p1], columnNames[p2]) for x in range(len(columnNames)) for y in range(x+1,len(columnNames))]

    ###=== Plot values on Scatterplots
#    for index in range(1, len(columnNames)):
#        makeScatterPlot(df[columnNames[0]], df[columnNames[index]], variableLabels[0], variableLabels[index], titleText="Comparison of Walkability Scores", dotSize=1, colorScheme=None, figSize=7.5, filename=fileLocation+"scatterplot-"+columnNames[index]+"-v"+str(fileIndex)+".png")
#    dotSize=1
#    figSize=5
#    ###=== Plot values on Scatterplots
#    plt.rcParams["font.family"] = "Palatino Linotype"
#    fig, ax = plt.subplots(figsize=(figSize, figSize))
#
#    xData = df[columnNames[0]]
#    plt.scatter(xData, df[columnNames[1]], s=dotSize, c=myGrayMed, alpha=0.75, label="unweighted")
#    plt.scatter(xData, df[columnNames[2]], s=dotSize, c=myBlueMed, alpha=0.75, label=r'$\lambda = 2$')
#    plt.scatter(xData, df[columnNames[3]], s=dotSize, c=myRedMed, alpha=0.75, label=r'$\lambda = 1$')
#    plt.scatter(xData, df[columnNames[4]], s=dotSize, c=myGreenMed, alpha=0.75, label=r'$\lambda = 0.5$')
#
#    plt.grid()
#    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
#    plt.title("Comparison of Station Scores" , fontsize=16, y=1.0)
#    plt.xlabel("score for within 1200m", fontsize=14)
#    plt.ylabel("score of station by other score", fontsize=14);
#
#    plt.show()
#    fig.savefig(fileLocation+dataName+"scatterplot-scores-v"+str(fileIndex)+".png", dpi=300, transparent=True, bbox_inches='tight')

    figSize=len(columnNames)-3
    ###=== Do Pearson Correlation
    pearsonMatrix = np.array([pearsonCorr(df[x], df[y])[0] for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
    makeArrayPlot(pearsonMatrix, variableLabels, "Pearson Correlation Matrix", figSize=figSize, filename=fileLocation+dataName+"arrayPlot_pearson"+"-v"+str(fileIndex)+fileType)
    ###=== Do Spearman Correlation
    # spearmanMatrix = np.array([spearmanCorr(df[x], df[y])[0] for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
    # makeArrayPlot(spearmanMatrix, variableLabels, "Spearman Correlation Matrix", figSize=figSize, filename=fileLocation+dataName+"arrayPlot_spearman"+"-v"+str(fileIndex)+fileType)


###--------------------------------------------------------
def rankListSimilarityAnalysis(df, columnNames, variableLabels=None, fileLocation="../Conference Materials/ComplexNetworks2020/Diagrams/", fileIndex=1):
    variableLabels = columnNames if variableLabels == None else variableLabels

    xData = df[columnNames[0]]
    dotSize=1
    figSize=5
    scatterPlotColors = [myGrayMed, myBlueMed, myRedMed, myGreenMed]

    ###=== Plot ranks on Scatterplots
    for index in range(1, len(columnNames)):
        makeScatterPlot(xData, df[columnNames[index]], variableLabels[0], variableLabels[index], titleText="Comparison of Station Ranks", dotSize=1.5, colorScheme=scatterPlotColors[index-1], figSize=figSize, filename=fileLocation+"scatterplot-"+columnNames[index]+"-rank-v"+str(fileIndex)+".png")

    ###=== Plot ranks on Scatterplots
    dotSize=1
    figSize=5
    plt.rcParams["font.family"] = "Palatino Linotype"
    fig, ax = plt.subplots(figsize=(figSize, figSize))

    plt.grid()
    plt.plot([0,490],[0,490], c=myGrayMed, alpha=0.5)
    plt.scatter(xData, df[columnNames[1]], s=dotSize, c=myGrayMed, alpha=0.75, label="unweighted")
    plt.scatter(xData, df[columnNames[2]], s=dotSize, c=myBlueMed, alpha=0.75, label=r'$\lambda = 2$')
    plt.scatter(xData, df[columnNames[3]], s=dotSize, c=myRedMed, alpha=0.75, label=r'$\lambda = 1$')
    plt.scatter(xData, df[columnNames[4]], s=dotSize, c=myGreenMed, alpha=0.75, label=r'$\lambda = 0.5$')

    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.title("Comparison of Station Ranks" , fontsize=16, y=1)
    plt.xlabel("rank of station for within 1250m", fontsize=14)
    plt.ylabel("rank of station by other score", fontsize=14);

    plt.show()
    fig.savefig(fileLocation+"scatterplot-ranks-v"+str(fileIndex)+".png", dpi=300, transparent=True, bbox_inches='tight')

    ###=== Arrayplots
    figSize=4
     ###=== Do Kendal Tau
    kendallMatrix = np.array([kendallCorr(df[x], df[y])[0] for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
    makeArrayPlot(kendallMatrix, variableLabels, "Kendall Correlation Matrix", figSize=figSize, filename=fileLocation+"arrayPlot_kendall"+"-v"+str(fileIndex)+".png")
#    ###=== Do Sequence Distance
#    sequenceMatrix = np.array([sequenceMatch(df[x], df[y]) for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
#    makeArrayPlot(sequenceMatrix, variableLabels, "Sequence Similarity Matrix", figSize=figSize, filename=fileLocation+"arrayPlot_sequenceSim"+"-v"+str(fileIndex)+"b.png")
#    ###=== Do Edit Distance
#    editMatrix = np.array([editSimilarity(df[x], df[y]) for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
#    makeArrayPlot(editMatrix, variableLabels, "Edit Similarity Matrix", figSize=figSize, filename=fileLocation+"arrayPlot_editSim"+"-v"+str(fileIndex)+"b.png")


###--------------------------------------------------------
def nameListSimilarityAnalysis(df, columnNames, variableLabels=None, fileLocation="../Conference Materials/ComplexNetworks2020/Diagrams/", fileIndex=1):
    variableLabels = columnNames if variableLabels == None else variableLabels
    figSize=4

    ###=== Do Sequence Distance
    sequenceMatrix = np.array([sequenceMatch(df[x], df[y]) for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
    makeArrayPlot(sequenceMatrix, variableLabels, "Sequence Similarity Matrix", figSize=figSize, filename=fileLocation+"arrayPlot_sequenceSim"+"-v"+str(fileIndex)+".png")
    ###=== Do Edit Distance
    # editMatrix = np.array([editSimilarity(df[x], df[y]) for x in columnNames for y in columnNames]).reshape(len(columnNames), len(columnNames))
    # makeArrayPlot(editMatrix, variableLabels, "Edit Similarity Matrix", figSize=figSize, filename=fileLocation+"arrayPlot_editSim"+"-v"+str(fileIndex)+".png")




###--------------------------------------------------------
####---- Convert the cluster labels from one list to best match the cluster labels of another
def translateLabels(masterList, listToConvert):
    contMatrix = contingency_matrix(masterList, listToConvert)
    labelMatcher = munkres.Munkres()
    labelTranlater = labelMatcher.compute((contMatrix.max() - contMatrix).tolist())
    uniqueLabels1 = list(set(masterList))
    uniqueLabels2 = list(set(listToConvert))
    tranlatorDict = {}
    for thisPair in labelTranlater:
        tranlatorDict[uniqueLabels2[thisPair[1]]] = uniqueLabels1[thisPair[0]]
    return [tranlatorDict[label] for label in listToConvert]



def runKMeans(dataMatrix, numClusters = 7):
    clusteringResults = KMeans(n_clusters=numClusters).fit(dataMatrix)
    orderedClusterCenters = np.argsort(clusteringResults.cluster_centers_.sum(axis=1))
    orderedClusterLabels = np.zeros_like(orderedClusterCenters)
    orderedClusterLabels[orderedClusterCenters] = np.arange(numClusters)
    return list(orderedClusterLabels[clusteringResults.labels_])

def runHierarchicalClustering(dataMatrix, numClusters = 7, linkage="ward", affinity="euclidean"):
    return (AgglomerativeClustering(n_clusters=numClusters, linkage=linkage, affinity=affinity).fit(dataMatrix)).labels_

def runSpectralClustering(dataMatrix, numClusters = 7):
    return (SpectralClustering(n_clusters=numClusters, affinity='nearest_neighbors').fit_predict(dataMatrix))

def analyzeClusters(resultName, dataHeaders, dataset, numClusters = 7):
    dataMatrix = dataset[dataHeaders].values            ## Convert full DataFrame to a matrix
    kMeansLabels = runKMeans(dataMatrix,numClusters)                ## Get the k-means result labels, ordered by the values of the centers
    dataset[resultName+'-kMeans'+str(numClusters)] = kMeansLabels
    dataset[resultName+'-hier'+str(numClusters)] = translateLabels(kMeansLabels,runHierarchicalClustering(dataMatrix,numClusters))  ## Use the k-means labels as a basis for other labelings
    dataset[resultName+'-spec'+str(numClusters)] = translateLabels(kMeansLabels,runSpectralClustering(dataMatrix,numClusters))

def analyzeClustersAndDifferences(resultName, dataHeaders, dataset, numClusters = 7):
    nClust = str(numClusters)
    dataMatrix = dataset[dataHeaders].values                         ## Convert full DataFrame to a matrix
    meanValues = list(np.mean(dataMatrix, axis = 1))
    dataset[resultName+'-meanValue'] = [(float(i)-min(meanValues))/(max(meanValues)-min(meanValues)) for i in meanValues]
    kMeansLabels = runKMeans(dataMatrix,numClusters)                ## Get the k-means result labels, ordered by the values of the centers
#    print(kMeansLabels)
    dataset[resultName+'-kMeans'+nClust ] = kMeansLabels
    hierarchicalLabels = translateLabels(kMeansLabels,runHierarchicalClustering(dataMatrix,numClusters))  ## Use the k-means labels as a basis for other labelings
#    print(hierarchicalLabels)
    dataset[resultName+'-hier'+nClust ] = hierarchicalLabels
    spectralLabels = translateLabels(kMeansLabels,runSpectralClustering(dataMatrix,numClusters)) ## Use the k-means labels as a basis for other labelings
#    print(spectralLabels)
    dataset[resultName+'-spec'+nClust ] = spectralLabels
#    outputData.append([resultName, mutualInfo(kMeansLabels,hierarchicalLabels), mutualInfo(kMeansLabels,spectralLabels), mutualInfo(spectralLabels,hierarchicalLabels)])
#    print("Mutual Info:",mutualInfo(kMeansLabels,hierarchicalLabels), mutualInfo(kMeansLabels,spectralLabels), mutualInfo(spectralLabels,hierarchicalLabels))
#    dataset[resultName+'-kMeans'+nClust +'|hier'+nClust +'-Diff'] = labelSimilarity(kMeansLabels,hierarchicalLabels)
#    dataset[resultName+'-kMeans'+nClust +'|spec'+nClust +'-Diff'] = labelSimilarity(kMeansLabels,spectralLabels)
#    dataset[resultName+'-spec'+nClust +'|hier'+nClust +'-Diff'] = labelSimilarity(spectralLabels,hierarchicalLabels)
#    print("Label Similarity:",labelSimilarity(kMeansLabels,hierarchicalLabels), labelSimilarity(kMeansLabels,spectralLabels), labelSimilarity(spectralLabels,hierarchicalLabels))
#    dataset[resultName+'-kMeans'+nClust +'|hier'+nClust +'-Rand'] = randIndex(kMeansLabels,hierarchicalLabels)
#    dataset[resultName+'-kMeans'+nClust +'|spec'+nClust +'-Rand'] = randIndex(kMeansLabels,spectralLabels)
#    dataset[resultName+'-spec'+nClust +'|hier'+nClust +'-Rand'] = randIndex(spectralLabels,hierarchicalLabels)



###----------------------------------------------------
###=== Calculate the approximate width and length of a polygon for a polygon
def getLengthAndWidth(thisPoly):
    thisArea = thisPoly.area
    thisPerimeter = thisPoly.length
    thisWidth = 1
    if ((thisPerimeter * thisPerimeter) - (16 * thisArea)) > 0:
        thisWidth = (thisPerimeter - math.sqrt((thisPerimeter * thisPerimeter) - (16 * thisArea))) / 4
        thisWidth = 1 if thisWidth < 1 else thisWidth
    thisLength = thisArea / thisWidth
    return (thisLength, thisWidth)


def getPolygonWidth(thisPoly):
    thisArea = thisPoly.area
    thisPerimeter = thisPoly.length
    thisWidth = 1
    if ((thisPerimeter * thisPerimeter) - (16 * thisArea)) > 0:
        thisWidth = (thisPerimeter - math.sqrt((thisPerimeter * thisPerimeter) - (16 * thisArea))) / 4
        thisWidth = 1 if thisWidth < 1 else thisWidth
    return thisWidth






###=============================== NETWORK ANALYSIS HELPERS ===============================


def getPathCost(thisNetwork, thisPath):
    totalCost = 0
    for thisEdge in list(pairwise(thisPath)):
        attr = thisNetwork.get_edge_data(thisEdge[0],thisEdge[1])
        totalCost += attr['costWeight']
    return makeInt(totalCost)










###=============================== OSM DATA PULLS ===============================

###=== make it simpler to query overpy by allowing full queries to be built from just the meat part.
def buildOverpassQuery(queryBodyInput, getByID=False):
    areaMinX, areaMinY, areaMaxX, areaMaxY = (139.22574925811375, 35.11217077290081, 140.23614041333477, 36.09344272613344)  ###--- bounding box for tokyoMain
    if getByID:
        pass
    elif isinstance(queryBodyInput, list):
        queryBodyInput = [query+'({yMin},{xMin},{yMax},{xMax});' for query in queryBodyInput]
        # print(queryBodyInput)
        queryBodyInput = '\n         '.join(queryBodyInput)
        # print(queryBodyInput)
        # queryBodyInput = ''.join(str(queryBodyInput))[1:-1].replace(",", " ")  ##-- convert the list of strings into a single string
    else:
        queryBodyInput = queryBodyInput+'({yMin},{xMin},{yMax},{xMax});'

    queryBodyOutput = """
        [out:json][timeout:900];
        ({queryBody}
        );
        out body;
        (._;>;);
        out meta;
        """.format(queryBody=queryBodyInput).format(xMin=str(areaMinX), yMin=str(areaMinY), xMax=str(areaMaxX), yMax=str(areaMaxY))
    # print(queryBodyOutput)
    return queryBodyOutput

###--- the overpy library has to be imported in the code that calls this command
def pullDataFromOverpass(thisQuery, sleepTime=15, numAttempts=5):
    import overpy
    api = overpy.Overpass()
    queryResults = []
    if not 'out:json' in thisQuery:
        thisQuery = buildOverpassQuery(thisQuery)  ##-- support entering complete queries or just query bodies
    overpassLookupFailed = True
    overassAttemps = 0
    while ((overpassLookupFailed == True) & (overassAttemps < numAttempts)):
        try:
            queryResults = api.query(thisQuery)
            overpassLookupFailed = False
        except:
            print("  -! Overpass data pull failed, trying Again.")
            overassAttemps += 1
            time.sleep(sleepTime)
    if isinstance(queryResults,list):
        print("  -- Query returned no data or failed.")
        raise RuntimeError("  -- Failed pulling data from Overpass, cannot proceed.")
    return queryResults

###-----------------------------------






###=== process geometries to remove garbage leftover from dissolving imperfectly aligned geom
###--- threshold is in meters, so the crs must be converted
##-- from: https://gis.stackexchange.com/questions/409340/removing-small-holes-from-the-polygon
def cleanSlivers(geom, threshold):
    geom = convertGeomCRS(geom, standardToAreaProj)

    if isinstance(geom, MultiPolygon):
        list_parts = []
        for polygon in geom.geoms:
            list_interiors = []
            for interior in polygon.interiors:
                p = Polygon(interior)
                if p.area > threshold:
                    list_interiors.append(interior)

            list_parts.append(Polygon(polygon.exterior.coords, holes=list_interiors))

        geom = MultiPolygon(list_parts)  ##-- or unary_union(list_parts)
        geom = geom.buffer(1).buffer(-1) ##-- that fills holes, this fills boundary slivers

    elif isinstance(geom, Polygon):
        list_interiors = []
        for interior in geom.interiors:
            p = Polygon(interior)
            if p.area > threshold:
                list_interiors.append(interior)

        geom = Polygon(geom.exterior.coords, holes=list_interiors)
        ##-- that fills holes, this fills boundary slivers (aka cracks),
        geom = geom.buffer(1).buffer(-1)
        ##-- tried using mitered corners to eliminate smoothing, but it doesn't and adds pointy bits at corners
        # geom = geom.buffer(1, cap_style=3, join_style=2).buffer(-1, cap_style=3, join_style=2)

    else:
        print(" !! geometry is a",type(geom),"so no garbage processing")

    geom = convertGeomCRS(geom, areaToStandardProj)
    return geom



###=== Converts a linestring into a list of simple two-point segments
def getSegments(curve):
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


def getLonLatFromGeom(geom):
    if isinstance(geom, Point):
        lon = geom.x
        lat = geom.y
    if isinstance(geom, LineString):
        midPoint = geom.interpolate(0.5, normalized=True)
        lon = midPoint.x
        lat = midPoint.y
    if isinstance(geom, MultiLineString):
        theCenter = geom.representative_point()
        lon = theCenter.x
        lat = theCenter.y
    if isinstance(geom, Polygon):
        theCenter = geom.representative_point()
        lon = theCenter.x
        lat = theCenter.y
    if isinstance(geom, MultiPolygon):
        theCenter = geom.representative_point()
        lon = theCenter.x
        lat = theCenter.y

    return [lon, lat]


def getGeomType(geom):
    if isinstance(geom, Point):
        return "Point"
    elif isinstance(geom, LineString):
        return "LineString"
    elif isinstance(geom, MultiLineString):
        return "MultiLineString"
    elif isinstance(geom, Polygon):
        return "Polygon"
    elif isinstance(geom, MultiPolygon):
        return "MultiPolygon"
    ###-------------------------------------------------------




###============================================================================
###============================= TRAVERSAL FUNCTIONS ==========================
###============================================================================

###=== only use the walking network (INCLUDE time from road network to hex center)
def walkWeight(u,v,attr):
    if attr.get('modality','poo') in ['hex', 'hexStation', 'stationToPlatform', 'transfer', 'platformToStation', 'train', 'through', 'interStation']:
        return None
    else:
        return attr.get('walkTime',None)

# ###=== only use the walking network (EXCLUDE time from road network to hex center)
# def walkWeight2(u,v,attr):
#     if attr.get('modality','poo') in ['hex', 'hexStation', 'stationToPlatform', 'transfer', 'platformToStation', 'train', 'through', 'interStation']:
#         return None
#     elif attr.get('modality','poo') in ['hexAccess']:
#         return 0
#     return attr.get('walkTime',None)

###=== traversals across the hex network by walking only
def hexWeight(u,v,attr):
    if attr.get('modality','poo') in ['hexStation','hex']:
        return attr.get('walkTime',None)
    else:
        return None

###=== traversals across the hex network by walking and train
def hexTrainWeight(u,v,attr):
    if ((attr.get('timeWeight',None) != None) & (attr.get('walkTime',None) != None)):
        # print("both weights exist in",attr.get('modality'),"edge: using timeWeight =", attr.get('timeWeight',None), "instead of walkTime =", attr.get('walkTime',None))
        return attr.get('timeWeight',None)
    elif attr.get('timeWeight',None) != None:
        return attr.get('timeWeight',None)
    elif attr.get('walkTime',None) != None:
        return attr.get('timeWeight',None)
    else:
        return None

    # if attr.get('modality','poo') in ['hexStation','hex']:
    #     return attr.get('walkTime',None)
    # elif attr.get('modality','poo') in ['stationToPlatform', 'transfer', 'platformToStation', 'train', 'through', 'interStation']:
    #     return attr.get('timeWeight',None)
    # else:
    #     return None



###=== get the appropriate point geometry for creating lines from servies of points
def getPointGeom(node, thisNetwork):
    thisGeom = thisNetwork.nodes[node].get('geometry','poo')
    if isinstance(thisGeom, Point):
        return thisGeom
    elif thisNetwork.nodes[node].get('point','poo') != 'poo':
        return thisNetwork.nodes[node].get('point','poo')
    elif thisNetwork.nodes[node].get('lat','poo') != 'poo':
        return Point(thisNetwork.nodes[node].get('lon',0), thisNetwork.nodes[node].get('lat',0))
    else:
        print("  -- cannot find a point geometry.")
        return None




# G = nx.random_regular_graph(3, 8, seed=None)
# for (start, end) in G.edges:
#     G.edges[start, end]['myAttr'] = np.random.randint(1,10)

# pth = nx.shortest_path(G, source=0, target=7, weight='myAttr')
# print(pth)
# edgeAttrList= [G[pth[i]][pth[i+1]]['myAttr'] for i in range(len(pth[:-1]))]
# print(edgeAttrList)

# edgeAttrList= [G[u][v]['myAttr'] for u,v in zip(pth, pth[1:])]
# print(edgeAttrList)

# edgeAttrList = [G[u][v].get('myAttr') for u,v in zip(pth, pth[1:]) if G[u][v].get('myAttr','foo') != 'foo']
# print(edgeAttrList)









#####======================================== END OF FILE ===========================================
