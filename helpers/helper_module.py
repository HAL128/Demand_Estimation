# -*- coding: utf-8 -*-


###=== Module List ============================================================
import ast
import codecs
import collections
import colorsys
import contextily as ctx  ## needed for plotting on basemaps
import csv
import fiona
import fugashi
import gc
import geopandas as gp
import geopy
import geopy.distance
import glob
import heapq
import json
import keplergl
import math
import matplotlib
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import momepy
import munkres
import networkx as nx
import nltk
import numbers
import numpy as np
import os
import osm_loader
import overpy
import pandas as pd
import pickle
import pke
import pke.unsupervised
import psycopg2 as pg
import random
import rasterio
import rasterio.env
import re
import requests
import spacy
import statsmodels.api as sm
import string
import sys
import s3fs
import time
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import unicodedata
import urllib.parse
import urllib.request, urllib.error
import warnings



from ast import literal_eval
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from datetime import date
from dateutil.relativedelta import relativedelta
from difflib import SequenceMatcher
from geoalchemy2 import Geometry
from geojson import Feature, FeatureCollection, dump
from geopy import Point
from itertools import permutations
from janome.tokenizer import Tokenizer
from matplotlib import cm
from matplotlib import collections  as mc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool, Manager, cpu_count, Value
from networkx.utils import pairwise
from numpy.polynomial import Polynomial
from psycopg2.extras import RealDictCursor
from pyproj import Proj, Transformer
from rasterio.features import shapes
from scipy.optimize import curve_fit  ## for the elevation profile approximation
from scipy.spatial import cKDTree
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import rankdata
from scipy.stats import spearmanr
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from shapely import wkt
from shapely.affinity import translate
from shapely.affinity import scale
from shapely.geometry import box
from shapely.geometry import LinearRing
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from shapely.ops import nearest_points
from shapely.ops import split
from shapely.ops import transform
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import explain_validity
from shapely.wkb import loads as wkbloads
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI
from sklearn.metrics.cluster import adjusted_rand_score as randIndex
from sklearn.metrics.cluster import contingency_matrix
from spacy.lang.ja.stop_words import STOP_WORDS as JA_STOP_WORDS
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.types import String
from statistics import mode
from statistics import stdev
from tqdm import tqdm



###=== End of File ============================================================