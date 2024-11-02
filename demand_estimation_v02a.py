# -*- coding: utf-8 -*-
from Codebase.helpers.helper_functions import *
from helpers.database_helpers import *



dbConnection = DatabaseConnInfo(username='data_warehouse_owner', password='3x4mp13us3r')
###============================================================================

###=== get the centroid of a polygon or other geometry using the angle-preserving CRS and recast back to standard CRS
def getCentroid(geom):
   return convertGeomCRS((convertGeomCRS(geom, standardToMapProj)).centroid, mapToStandardProj)

###===========================================================================



# ###===========================================================================================
# ###================================= PROCESS RENT PRICES =====================================
# ###===========================================================================================

# filename = '../Data/warehouse_mansion_research_rent_story_normalized.csv'

# # ###=== Get a sample of data to confirm the column names
# # roomData = pd.read_csv(filename, encoding='utf-8', nrows=10).fillna('')
# # print(list(roomData))
# # print(roomData.head())

# # ###=== Confirm that I can load it using just those column names.
# # useCols=['id','[最低賃料]','築年月（西暦）','使用部分面積','lat','lon']
# # roomData = pd.read_csv(filename, encoding='utf-8', usecols=useCols, nrows=10).fillna('')
# # print(roomData.head())

# ###=== Read the full dataset, but only select columns
# useCols=['id','[最低賃料]','築年月（西暦）','使用部分面積','lat','lon']
# roomData = pd.read_csv(filename, encoding='utf-8', usecols=useCols).fillna('')

# # chunksize = 100000
# # with pd.read_csv(filename, encoding='utf-8', chunksize=chunksize, usecols=useCols) as reader:
# #     for chunk in reader:
# #         process(chunk)

# print("-- Adding geometry column to data")
# roomData['geometry'] = [Point(xy) for xy in zip(roomData.lon, roomData.lat)]
# roomData = gp.GeoDataFrame(roomData, geometry='geometry')
# roomData.crs = standardCRS

# print("-- Filtering to Tokyo Main")
# analysisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon.pkl')
# roomData = roomData[roomData['geometry'].intersects(analysisArea)]

# print("-- Refining Data Columns")
# roomData.rename(columns={'[最低賃料]': 'rent', '築年月（西暦）': 'built_yyyymm', '使用部分面積': 'surface_area'}, inplace=True)
# roomData['built_year'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[0:4]))
# roomData['built_month'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[4:6]))
# roomData.drop(columns=['built_yyyymm', 'lat', 'lon'], inplace=True)

# print("-- Saving Unfiltered Rooms")
# writePickleFile(roomData,'../Demand Estimation/roomData_Full_TokyoMain.pkl')

# ###-------------------------------------------------------------
# print("-- Filtering Rooms to Desired Subset")
# roomData = roomData[((roomData['built_year'] < 2017) & (roomData['built_year'] >= 2003))]
# roomData = roomData[((roomData['surface_area'] < 30) & (roomData['surface_area'] >= 25))]

# print("-- Saving Filtered Rooms")
# writePickleFile(roomData,'../Demand Estimation/roomData_Filtered_TokyoMain.pkl')
# writeGeoCSV(roomData, '../Demand Estimation/roomData_Filtered_TokyoMain.csv')
# ###============================================================================




# ###===========================================================================================
# ###================================= SETUP SOURCE DEMAND =====================================
# ###===========================================================================================

# print("== Starting Setup of Source Demand")
# thisArea = getPolygonForArea('tokyoMain')

# ##load the hex network and job data, and resample the latter intot he former.
# hexNetwork = readPickleFile('../Data/GridData/hexNetwork_v1d.pkl')

# hexNodes = [node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('connected','poo') == True))]
# print("  -- Number of connected hex nodes:", len(hexNodes))  ##==> 126440  --> 126865

# hexNodeData,hexEdgeData = convertGraphToGeopandas(hexNetwork)
# hexNodeData = hexNodeData[hexNodeData['modality']=='hex']
# hexNodeData = hexNodeData[hexNodeData['connected']==True]
# hexNodeData = hexNodeData.reset_index(drop=True)
# print(hexNodeData.head())
# print("  -- Number of connected hex nodes:", len(hexNodeData))  ##==> 126440  --> 126865◘ (confirmed, the same number)

# jobsData = get_data_for_geom(dbConnection, table="economics", geom=thisArea)
# print(jobsData.head())
# jobsData['centroid'] = jobsData['geometry'].apply(lambda geom: getCentroid(geom))

# for thisHex in hexNodes:
#     hexNetwork.nodes[thisHex]['sourceJobs'] = 0
#     hexNetwork.nodes[thisHex]['demand'] = 0

# ###=== Assign initial demand from jobs grid to hex grid
# for index,row in jobsData.iterrows():
#     theseJobs = row['num_employees']
#     if theseJobs > 0:
#         thisPoint = row['centroid']
#         theseHexes = hexNodeData[hexNodeData['geometry'].intersects(thisPoint)]
#         ##-- in most cases, exactly one hex should intersect the centroid of each grid.
#     ##-- if there are multiple matches, then it's exactly on a boundary, so just do the first (vs random) one.
#         if len(theseHexes) >= 1:
#             thisHex = list(theseHexes['id'])[0]
#             hexNetwork.nodes[thisHex]['sourceJobs'] += theseJobs
#         ##-- if there is no connected hex here, then we really want the closest one.
#         else:
#             # print("  !! There is no hex intersecting the jobs grid", row['mesh_code'])    ##-- happens a lot!
#             rowIdx,_ = getClosestPoint(thisPoint.y, thisPoint.x, hexNodeData)
#             thisHex = hexNodeData.at[rowIdx,'id']
#             hexNetwork.nodes[thisHex]['sourceJobs'] += theseJobs

# ##=== setting a common edge weight
# for n1,n2,attr in hexNetwork.edges(data=True):
#     if ((attr.get('timeWeight',None) != None) & (attr.get('walkTime',None) != None)):
#         print("both weights exist, using timeWeight")
#         hexNetwork[n1][n2]['weight'] = attr.get('timeWeight',None)
#     elif attr.get('timeWeight',None) != None:
#         hexNetwork[n1][n2]['weight'] = attr.get('timeWeight',None)
#     elif attr.get('walkTime',None) != None:
#         hexNetwork[n1][n2]['weight'] = attr.get('walkTime',None)
#     else:
#         print("some other weight exists")

# convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v1d+jobs-source")

# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v1d+jobs-source.pkl')
# ###============================================================================



# ###===========================================================================================
# ###================================= PROCESS DEMAND FLOW =====================================
# ###===========================================================================================

###=== For a given source amount and time from source, return the time-discounted amount for the chosen functional form
def weightedDemand(sourceValue, time, function='linear', maxLimit=60, curvature=1):
    if time > maxLimit:
        return 0
    elif function == 'linear':
        return (1 - (time / maxLimit)) * sourceValue
    elif function == 'sCurve':
        return weightedValue(time, value=sourceValue, curvature=curvature, scalingLimit=maxLimit)
    else:
        print("  !! Undefined weighting function requested.")
        return 0



# makeFunctionPlot()





# print("== Starting Demand Flow Process")

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v1d+jobs-source.pkl')
# hexNodes = [node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('connected',False) == True))]
# print("  == Number of hexes:", len(hexNodes))  ##--> 126440 --> 126865
# sourceNodes = sorted([node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('sourceJobs',0) > 0))])
# print("  == Number of source hex nodes:", len(sourceNodes))  ##--> 17201  --> 17206
# numSources = len(sourceNodes)

# # print(getAllEdgeAttributes(hexNetwork))
# ###==> ['timeWeight', 'lineName', 'weight', 'routeID', 'length', 'modality', 'geomDist', 'color', 'walkTime', 'walkingSpeed', 'direction', 'distance', 'lineType', 'geomMap', 'lineGeom', 'geometry', 'throughService']

# # listOfWeightings = ['linear-60', 'linear-90']
# listOfWeightings = ['linear-60', 'sCurve-60_05', 'sCurve-60_10', 'sCurve-60_20', 'linear-90', 'sCurve-90_05', 'sCurve-90_10', 'sCurve-90_20']
# #weightList = [0, 0.5, 1, 2]
# #weightNames = ["00", "05", "10", "20"]

# ###=== Setup all the nodes to have zero values for all weights.
# for thisHex,hexAttr in hexNetwork.nodes(data=True):
#     for thisWeight in listOfWeightings:
#         hexNetwork.nodes[thisHex]['demand_'+thisWeight] = 0

# ###=== For each source node, propagate the demand to all the other nodes.
# runStartTime = time.time()
# for index,source in enumerate(sourceNodes):
#     runStartTime = printProgress(runStartTime, index, numSources)
#     ###=== The original demand from this source
#     sourceDemand = hexNetwork.nodes[source]['sourceJobs']
#     ###=== Get the time to all other nodes withing 90 minutes
#     allTimes = nx.single_source_dijkstra_path_length(hexNetwork, source, cutoff=90, weight='weight')
#     ###=== Now accumulate the weighted demand to all reachable hexes based on each weighting apporach
#     for node,time in allTimes.items():
#         hexNetwork.nodes[node]['demand_linear-90'] += weightedDemand(sourceDemand, time, function='linear', maxLimit=90)
#         hexNetwork.nodes[node]['demand_sCurve-90_05'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=0.5)
#         hexNetwork.nodes[node]['demand_sCurve-90_10'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=1.0)
#         hexNetwork.nodes[node]['demand_sCurve-90_20'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=2.0)
#         if time <= 60:
#             hexNetwork.nodes[node]['demand_linear-60'] += weightedDemand(sourceDemand, time, function='linear', maxLimit=60)
#             hexNetwork.nodes[node]['demand_sCurve-60_05'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=0.5)
#             hexNetwork.nodes[node]['demand_sCurve-60_10'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=1.0)
#             hexNetwork.nodes[node]['demand_sCurve-60_20'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=2.0)

# ###=== Save the results for further analysis
# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v01d+jobs.pkl')

# ###=== Output for visualization in Kepler
# convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v01d+jobs")
###============================================================================








# ##===========================================================================================
# ##================================= ANALYZE DEMAND AND RENT =================================
# ##===========================================================================================

# print("== Starting Analysis of Demand and Rent")

# roomData = readPickleFile('../Demand Estimation/roomData_Filtered_TokyoMain.pkl')

# print("-- Filtering Rooms to Desired Subset")
# roomData = roomData[((roomData['built_year'] >= 2010) & (roomData['built_year'] <= 2014))]
# roomData = roomData[((roomData['surface_area']  >= 25) & (roomData['surface_area'] <= 28))]
# # print(roomData.head())
# print("Number of selected rooms in area:", len(roomData))  ##==> 702,005  --> 356,744

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v01d+jobs.pkl')
# nodeDF,edgeDF = convertGraphToGeopandas(hexNetwork)
# hexData = nodeDF[nodeDF['modality']=='hex']
# print("Number of hexes in hexData:", len(hexData))  ##==> 126448
# # print(hexData.head())

# # stationData = nodeDF[nodeDF['modality']=='station']
# # print("Number of stations in hexData:", len(stationData))  ##==> 1468

# # print(hexData.head())

# def addRoomPricesToHex(hexGeom, roomDF):
#     theseRooms = roomDF[roomDF['geometry'].intersects(hexGeom)]
#     return list(theseRooms['rent'])

# def getStdDev(listOfVals):
#     if len(listOfVals) > 2:
#         return stdev(listOfVals)
#     elif len(listOfVals) == 1:
#         return 0
#     else:
#         return None

# hexData['prices'] = hexData['geometry'].apply(lambda geom: addRoomPricesToHex(geom, roomData))
# hexData['price_mean'] = hexData['prices'].apply(lambda val: np.mean(val) if len(val) > 0 else None)
# hexData['price_stddev'] = hexData['prices'].apply(lambda val: getStdDev(val))

# writePickleFile(hexData,'../Demand Estimation/hexData_with_prices_v01d.pkl')
# writeGeoCSV(hexData, '../Demand Estimation/hexData_with_prices_v01d.csv')
# ##===========================================================================================

# hexData = readPickleFile('../Demand Estimation/hexData_with_prices_v01d.pkl')

# priceData = hexData[hexData['prices'].apply(lambda val: val != [])]
# # print("Number of hexes with prices:", len(priceData))  ##==> 12542  about 10%
# print("Number of hexes with prices:", len(priceData))  ##==> 8412  about 6.65%
# # print(priceData.head())

# priceData['price_mean_log'] = priceData['price_mean'].apply(lambda val: safeLog(val) )

# ###==== Regressions on price
# listOfWeightings = ['linear-60', 'sCurve-60_05', 'sCurve-60_10', 'sCurve-60_20', 'linear-90', 'sCurve-90_05', 'sCurve-90_10', 'sCurve-90_20']
# listOfLabels = ['linear-60', 'curved-60 0.5', 'curved-60 1.0', 'curved-60 2.0', 'linear-90', 'curved-90 0.5', 'curved-90 1.0', 'curved-90 2.0']

# for index,thisWeight in enumerate(listOfWeightings):

#     thisWeightVar = 'demand_'+thisWeight
#     thisWeightLab = listOfLabels[index]

#     predVals = list(priceData[thisWeightVar])
#     truthVals = list(priceData['price_mean'])

#     thisCorrelation = getCorrelation(predVals, truthVals)
#     print("== Pearson Correlation Coefficient for", thisWeight,":", thisCorrelation)

#     makeScatterPlot(predVals, truthVals, xLabel='estimated demand', yLabel='mean rent', titleText=' '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_1d_results2/price_vs_'+thisWeight+'.png')

#     ###=== Get the residual values and plot them as a function of X
#     regressionModel = LinearRegression().fit(np.array(predVals).reshape(-1, 1), truthVals)
#     prediction = regressionModel.predict(np.array(predVals).reshape(-1, 1))
#     residual_values = (truthVals - prediction)

#     # OLS_model = sm.OLS(truthVals,predVals).fit()  # training the model
#     # predicted_values = OLS_model.predict()  # predicted values
#     # residual_values = OLS_model.resid # residual values

#     makeScatterPlot(predVals, residual_values, xLabel='estimated demand', yLabel='residual', titleText='residuals '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_1d_results2/price_vs_'+thisWeight+'_residuals.png')

#     makeHistogramPlot(residual_values, 'residual values', 'count', titleText='Residuals '+thisWeightLab, colorScheme=None, figSize=5, lowerBound=None, upperBound=None, filename='../Demand Estimation/hex_network_1d_results2/price_vs_'+thisWeight+'_residual_histogram.png')

# ####--------------------------------------------

# ###==== Regressions on log of price
# listOfWeightings = ['linear-60', 'sCurve-60_05', 'sCurve-60_10', 'sCurve-60_20', 'linear-90', 'sCurve-90_05', 'sCurve-90_10', 'sCurve-90_20']
# listOfLabels = ['linear-60', 'curved-60 0.5', 'curved-60 1.0', 'curved-60 2.0', 'linear-90', 'curved-90 0.5', 'curved-90 1.0', 'curved-90 2.0']

# for index,thisWeight in enumerate(listOfWeightings):

#     thisWeightVar = 'demand_'+thisWeight
#     thisWeightLab = listOfLabels[index]

#     predVals = list(priceData[thisWeightVar])
#     truthVals = list(priceData['price_mean_log'])

#     thisCorrelation = getCorrelation(predVals, truthVals)
#     print("== Pearson Correlation Coefficient for", thisWeight,":", thisCorrelation)

#     makeScatterPlot(predVals, truthVals, xLabel='estimated demand', yLabel='log mean rent', titleText=' '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_1d_results2/log-price_vs_'+thisWeight+'.png')

#     ###=== Get the residual values and plot them as a function of X
#     regressionModel = LinearRegression().fit(np.array(predVals).reshape(-1, 1), truthVals)
#     prediction = regressionModel.predict(np.array(predVals).reshape(-1, 1))
#     residual_values = (truthVals - prediction)

#     # OLS_model = sm.OLS(truthVals,predVals).fit()  # training the model
#     # predicted_values = OLS_model.predict()  # predicted values
#     # residual_values = OLS_model.resid # residual values

#     makeScatterPlot(predVals, residual_values, xLabel='estimated demand', yLabel='residual', titleText='residuals '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_1d_results2/log-price_vs_'+thisWeight+'_residuals.png')

#     makeHistogramPlot(residual_values, 'residual values', 'count', titleText='Residuals '+thisWeightLab, colorScheme=None, figSize=5, lowerBound=None, upperBound=None, filename='../Demand Estimation/hex_network_1d_results2/log-price_vs_'+thisWeight+'_residual_histogram.png')
####--------------------------------------------


# == Pearson Correlation Coefficient for linear-60 :    0.825
# == Pearson Correlation Coefficient for sCurve-60_05 : 0.794
# == Pearson Correlation Coefficient for sCurve-60_10 : 0.809
# == Pearson Correlation Coefficient for sCurve-60_20 : 0.821

# == Pearson Correlation Coefficient for linear-90 :    0.805
# == Pearson Correlation Coefficient for sCurve-90_05 : 0.83
# == Pearson Correlation Coefficient for sCurve-90_10 : 0.822
# == Pearson Correlation Coefficient for sCurve-90_20 : 0.793

# == Pearson Correlation Coefficient for linear-60 :    0.813
# == Pearson Correlation Coefficient for sCurve-60_05 : 0.766
# == Pearson Correlation Coefficient for sCurve-60_10 : 0.786
# == Pearson Correlation Coefficient for sCurve-60_20 : 0.807

# == Pearson Correlation Coefficient for linear-90 :    0.821
# == Pearson Correlation Coefficient for sCurve-90_05 : 0.827
# == Pearson Correlation Coefficient for sCurve-90_10 : 0.829
# == Pearson Correlation Coefficient for sCurve-90_20 : 0.812



###================================== ADD TIME TO TOKYO AND CALCULATE ITS CORRELATIONS ======================

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v01d+jobs.pkl')
# tokyoStation = getNodesByAttr(hexNetwork, 'stationNameEN', thisVal='Tokyo')[0]
# print(tokyoStation)  ##--> '6193732167'

# timesToTokyo = nx.single_source_dijkstra_path_length(hexNetwork, '6193732167', weight='weight')
# nx.set_node_attributes(hexNetwork, timesToTokyo, "timeToTokyo")

# hexData = readPickleFile('../Demand Estimation/hexData_with_prices_v01d.pkl')
# hexData['timeToTokyo'] = hexData['id'].map(timesToTokyo)
# print(hexData.head())

# priceData = hexData[hexData['prices'].apply(lambda val: val != [])]
# priceData['price_mean_log'] = priceData['price_mean'].apply(lambda val: safeLog(val) )
# print("Number of hexes with prices:", len(priceData))  ##==> 8412  about 6.65%

# thisCorrelation = getCorrelation(list(priceData['price_mean']), list(priceData['timeToTokyo']))
# print("== Pearson Correlation Coefficient for", 'price_mean',":", thisCorrelation)

# thisCorrelation = getCorrelation(list(priceData['price_mean_log']), list(priceData['timeToTokyo']))
# print("== Pearson Correlation Coefficient for", 'price_mean_log',":", thisCorrelation)

# == Pearson Correlation Coefficient for price_mean and timeToTokyo: -0.714
# == Pearson Correlation Coefficient for price_mean_log and timeToTokyo: -0.74
####--------------------------------------------





##================================== PROCESS OTHER STUFF FOR PRESENTATION & PAPER ======================

# thisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')
# thisAreaArea = convertGeomCRS(thisArea, standardToAreaProj)
# print(thisAreaArea.area)  ##==> 4893420593 m2 ==> 4893.420593 km2

# tokyoArea = readPickleFile('../Data/Polygons/tokyoAreaPolygon2.pkl')
# tokyoAreaArea = convertGeomCRS(tokyoArea, standardToAreaProj)
# print(tokyoAreaArea.area)  ##==> 13624284233 m2 ==> 13624.284233 km2

# print(thisAreaArea.area / tokyoAreaArea.area)  ##==> 0.359169 -> 36%


# populationData = get_columns_for_geom(dbConnection, table='adminarea_data_old', geom=thisArea, columns=['totalPopulation','lowestLevel','geometry'])
# populationData = populationData[populationData['lowestLevel']==True]
# totalPopulation = populationData['totalPopulation'].sum()
# print(totalPopulation)  ##--> 32,197,448


# employeeData = get_columns_for_geom(dbConnection, table='economics', geom=thisArea, columns=['num_employees','geometry'])
# totalEmployees = employeeData['num_employees'].sum()
# print(totalEmployees)  ##--> 15,080,305


# trainNetwork = readPickleFile('../Data/trainNetworks/trainNetwork_v8f.pkl')
# stationNodes = [node for node,attr in trainNetwork.nodes(data=True) if ((attr.get('modality')=='station'))]
# print(len(stationNodes))  ##--> 1441


# ###-----------------------
# # hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v2c+jobs-source.pkl')
# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores2.pkl')
# nonHexEdges = [(u,v) for u,v,attr in hexNetwork.edges(data=True) if not attr["modality"] in ['hexStation', 'hex']]
# hexOnlyNetwork = hexNetwork.copy()
# hexOnlyNetwork.remove_edges_from(nonHexEdges)

# print(hexOnlyNetwork.number_of_nodes())   ##--> 144,849  (129,793 are connected)
# print(hexOnlyNetwork.number_of_edges())   ##--> 2,163,600

# print("Number of hexes with source jobs:", len([node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('sourceJobs',0)>0))])) ##--> 17433




# ###--------------------- WITH NEW HEX NETWORK ------------------
# # == Pearson Correlation Coefficient for linear-60 :    0.825
# # == Pearson Correlation Coefficient for sCurve-60_05 : 0.788
# # == Pearson Correlation Coefficient for sCurve-60_10 : 0.804
# # == Pearson Correlation Coefficient for sCurve-60_20 : 0.82

# # == Pearson Correlation Coefficient for linear-90 :    0.81
# # == Pearson Correlation Coefficient for sCurve-90_05 : 0.832
# # == Pearson Correlation Coefficient for sCurve-90_10 : 0.826
# # == Pearson Correlation Coefficient for sCurve-90_20 : 0.8

# # == Pearson Correlation Coefficient for linear-60 :    0.81
# # == Pearson Correlation Coefficient for sCurve-60_05 : 0.758
# # == Pearson Correlation Coefficient for sCurve-60_10 : 0.779
# # == Pearson Correlation Coefficient for sCurve-60_20 : 0.802

# # == Pearson Correlation Coefficient for linear-90 :    0.825
# # == Pearson Correlation Coefficient for sCurve-90_05 : 0.828
# # == Pearson Correlation Coefficient for sCurve-90_10 : 0.832
# # == Pearson Correlation Coefficient for sCurve-90_20 : 0.819




# ###==============================================================================================
# ###================================== PERFORM SIMPLE HEDONIC PRICING MODEL ======================
# ###==============================================================================================


# ###================================= REPROCESS RENT PRICES =====================================

# filename = '../Data/warehouse_mansion_research_rent_story_normalized.csv'

# ###=== Get a sample of data to confirm the column names
# roomData = pd.read_csv(filename, encoding='utf-8', nrows=10).fillna('')
# # print(list(roomData))
# # print(roomData.head())

# # for idx,varName in enumerate(list(roomData)):
# #     print(idx, varName)

# japaneseHeaders = {'jp'+str(idx):varName for idx,varName in enumerate(list(roomData))}
# # print(japaneseHeaders)

# englishHeaders = {'jp0': 'id', 'jp1': 'Data type', 'jp2': 'Property type', 'jp3': 'Property type', 'jp4': 'Registration date', 'jp5' : 'Date of modification', 'jp6': 'Date of contract conclusion', 'jp7': 'Expiry date of transaction terms', 'jp8': 'Prefecture name', 'jp9': 'City name ', 'jp10': 'Town name', 'jp11': 'Street name', 'jp12': 'Street number', 'jp13': 'Entire address', 'jp14': 'Building name', 'jp15': ' Room number', 'jp16': 'Other location display', 'jp17': 'Building number', 'jp18': '[Nearest station] Real Estate Distribution Promotion Center (block_line_code)', 'jp19': '[Nearest station] Station] Line name', 'jp20': '[Nearest station] Real Estate Distribution Promotion Center (block_line_station_code)', 'jp21': '[Nearest station] Station name', 'jp22': '[Nearest station] Walking ( ', 'jp23': '[Nearest station] walk (m)', 'jp24': 'Bus time required', 'jp25': 'Bus route name', 'jp26': 'Bus stop name', 'jp27': 'Stopping (minutes)', 'jp28': 'Stopping (m)', 'jp29': 'Car (km)', 'jp30': 'Current status', 'jp31': 'Current status planned year Month', 'jp32': 'Move-in period', 'jp33': 'Move-in date (Western calendar)', 'jp34': 'Move-in period', 'jp35': 'Transaction type', 'jp36': 'Transaction type Flag (seller)', 'jp37': 'Transaction type flag (general)', 'jp38': 'Transaction type flag (exclusive)', 'jp39': 'Transaction type flag (exclusive)', 'jp40': ' Transaction type flag (agent)', 'jp41': 'Transaction type flag (brokerage)', 'jp42': 'Remuneration type', 'jp43': '[Minimum rent]', 'jp44': '[Minimum rent] Price per tsubo', 'jp45': '[Minimum rent] Price per square meter', 'jp46': '[Minimum rent] Rent only', 'jp47': '[Minimum rent] Management fee only', 'jp48': '[ Maximum rent]', 'jp49': '[Maximum rent] Price per tsubo', 'jp50': '[Maximum rent] Price per square meter', 'jp51': '[Maximum rent] Rent only', 'jp52': '[ Maximum rent] Management fee only', 'jp53': 'Contracted rent', 'jp54': 'Contracted price per tsubo', 'jp55': 'Contracted price per square meter', 'jp56': '[Contracted rent] Only rent', 'jp57': '[Contracted rent] Management fee only', 'jp58': 'Rent before contract', 'jp59': 'Price per tsubo before contract', 'jp60': 'Price per square meter before contract', 'jp61': '[Pre-contract] Rent only', 'jp62': '[Pre-contract] Management fee only', 'jp63': 'Deposit 1 (amount)', 'jp64': 'Deposit 2 (months)', 'jp65' : 'Entitlement money 1 (amount)', 'jp66': 'Entitlement money 2 (months)', 'jp67': 'Key money 1 (amount)', 'jp68': 'Key money 2 (months)', 'jp69' : 'Deposit 1 (amount)', 'jp70': 'Deposit 2 (months)', 'jp71': 'Used area', 'jp72': 'Balcony (terrace) area', 'jp73': 'Private garden Area', 'jp74': 'Use area (1)', 'jp75': 'Use area (2)', 'jp76': 'Registered land area', 'jp77': 'Current land area', 'jp78': 'Urban planning', 'jp79': 'Building coverage ratio', 'jp80': 'Floor area ratio', 'jp81': 'Regional district', 'jp82': 'Optimal use', 'jp83': 'Topography', 'jp84': 'Building lease classification', 'jp85': 'Building lease period', 'jp86': 'Building lease renewal', 'jp87': 'Management fee', 'jp88': 'Presence of management association', 'jp89' : 'Common service fee', 'jp90': 'Renewal category', 'jp91': 'Renewal fee (amount)', 'jp92': 'Renewal fee (month)', 'jp93': 'Insurance obligation', 'jp94': 'Insurance name', 'jp95': 'Insurance premium', 'jp96': 'Insurance period', 'jp97': 'Floor plan type (1)', 'jp98': 'Number of rooms with floor plan (1) )', 'jp99': 'Room location', 'jp100': 'Number of units', 'jp101': 'Parking lot availability', 'jp102': 'Parking lot monthly', 'jp103': 'Parking lot deposit (amount)', 'jp104': 'Parking deposit (months)', 'jp105': 'Parking key money (amount)', 'jp106': 'Parking key money (months)', 'jp107': 'Building Structure', 'jp108': 'Building construction method', 'jp109': 'Building type', 'jp110': 'Above ground floor', 'jp111': 'Underground floor', 'jp112': 'Location floor', 'jp113': 'Building date (Western calendar)', 'jp114': 'New construction flag', 'jp115': 'Balcony direction (1)', 'jp116': 'Number of days from sale to final supplement', 'jp117': 'Number of branch data', 'jp118': 'Number of real estate companies handled', 'jp119': 'Final real estate companies handled', 'jp120': 'Number of price revisions', 'jp121': 'Price change rate', 'jp122': 'Price change amount', 'jp123': 'Registration date and time', 'jp124': 'Last updated date and time', 'jp125': 'Apartment id', 'jp126': 'input_file_name', 'jp127': '_right_ _Building name #0', 'jp128': 'normalized_name', 'jp129': '_right__Entire address #1', 'jp130': 'matching_status', 'jp131': 'path', 'jp132': 'paired_address ', 'jp133': 'normalized_address', 'jp134': 'prefecture', 'jp135': 'city_ward', 'jp136': 'town', 'jp137': 'subtown', 'jp138': 'street', 'jp139': 'block', 'jp140': 'number', 'jp141': 'subnumber', 'jp142': 'lat', 'jp143': 'lon'}

# headerTranslator = {japaneseHeaders[k]:englishHeaders[k] for k,v in japaneseHeaders.items()}
# # print(headerTranslator)

# headerTranslator = {'id': 'id', 'データ種類': 'Data type', '物件種別': 'Property type', '物件種目': 'Property type', '登録年月日': 'Registration date', '変更年月日': 'Date of modification', '成約年月日': 'Date of contract conclusion', '取引条件の有効期限': 'Expiry date of transaction terms', '都道府県名': 'Prefecture name', '市区町村名': 'City name ', '町名': 'Town name', '丁目名': 'Street name', '番地': 'Street number', '住所全体': 'Entire address', '建物名': 'Building name', '部屋番号': ' Room number', 'その他所在地表示': 'Other location display', '棟番号': 'Building number', '[最寄駅]不動産流通推進センター（block_line_code）': '[Nearest station] Real Estate Distribution Promotion Center (block_line_code)', '[最寄駅]沿線名': '[Nearest station] Station] Line name', '[最寄駅]不動産流通推進センター（block_line_station_code）': '[Nearest station] Real Estate Distribution Promotion Center (block_line_station_code)', '[最寄駅]駅名': '[Nearest station] Station name', '[最寄駅]徒歩（分）': '[Nearest station] Walking ( ', '[最寄駅]徒歩（m）': '[Nearest station] walk (m)', 'バス所要時間': 'Bus time required', 'バス路線名': 'Bus route name', 'バス停名称': 'Bus stop name', '停歩（分）': 'Stopping (minutes)', '停歩（m）': 'Stopping (m)', '車（km）': 'Car (km)', '現況': 'Current status', '現況予定年月': 'Current status planned year Month', '入居時期': 'Move-in period', '入居年月（西暦）': 'Move-in date (Western calendar)', '入居旬': 'Move-in period', '取引態様': 'Transaction type', '取引態様フラグ（売主）': 'Transaction type Flag (seller)', '取引態様フラグ（一般）': 'Transaction type flag (general)', '取引態様フラグ（専属）': 'Transaction type flag (exclusive)', '取引態様フラグ（専任）': 'Transaction type flag (exclusive)', '取引態様フラグ（代理）': ' Transaction type flag (agent)', '取引態様フラグ（仲介）': 'Transaction type flag (brokerage)', '報酬形態': 'Remuneration type', '[最低賃料]': '[Minimum rent]', '[最低賃料]坪単価': '[Minimum rent] Price per tsubo', '[最低賃料]平米単価': '[Minimum rent] Price per square meter', '[最低賃料]賃料のみ': '[Minimum rent] Rent only', '[最低賃料]管理費のみ': '[Minimum rent] Management fee only', '[最高賃料]': '[ Maximum rent]', '[最高賃料]坪単価': '[Maximum rent] Price per tsubo', '[最高賃料]平米単価': '[Maximum rent] Price per square meter', '[最高賃料]賃料のみ': '[Maximum rent] Rent only', '[最高賃料]管理費のみ': '[ Maximum rent] Management fee only', '成約賃料': 'Contracted rent', '成約坪単価': 'Contracted price per tsubo', '成約㎡単価': 'Contracted price per square meter', '[成約賃料]賃料のみ': '[Contracted rent] Only rent', '[成約賃料]管理費のみ': '[Contracted rent] Management fee only', '成約前賃料': 'Rent before contract', '成約前坪単価': 'Price per tsubo before contract', '成約前㎡単価': 'Price per square meter before contract', '[成約前]賃料のみ': '[Pre-contract] Rent only', '[成約前]管理費のみ': '[Pre-contract] Management fee only', '保証金1（額）': 'Deposit 1 (amount)', '保証金2（ヶ月）': 'Deposit 2 (months)', '権利金1（額）': 'Entitlement money 1 (amount)', '権利金2（ヶ月）': 'Entitlement money 2 (months)', '礼金1（額）': 'Key money 1 (amount)', '礼金2（ヶ月）': 'Key money 2 (months)', '敷金1（額）': 'Deposit 1 (amount)', '敷金2（ヶ月）': 'Deposit 2 (months)', '使用部分面積': 'Used area', 'バルコニー（テラス）面積': 'Balcony (terrace) area', '専用庭面積': 'Private garden Area', '用途地域（1）': 'Use area (1)', '用途地域（2）': 'Use area (2)', '登記簿地目': 'Registered land area', '現況地目': 'Current land area', '都市計画': 'Urban planning', '建ぺい率': 'Building coverage ratio', '容積率': 'Floor area ratio', '地域地区': 'Regional district', '最適用途': 'Optimal use', '地勢': 'Topography', '建物賃貸借区分': 'Building lease classification', '建物賃貸借期間': 'Building lease period', '建物賃貸借更新': 'Building lease renewal', '管理費': 'Management fee', '管理組合有無': 'Presence of management association', '共益費': 'Common service fee', '更新区分': 'Renewal category', '更新料（額）': 'Renewal fee (amount)', '更新料（ヶ月）': 'Renewal fee (month)', '保険加入義務': 'Insurance obligation', '保険名称': 'Insurance name', '保険料': 'Insurance premium', '保険期間': 'Insurance period', '間取タイプ（1）': 'Floor plan type (1)', '間取部屋数（1）': 'Number of rooms with floor plan (1) )', '部屋位置': 'Room location', '納戸数': 'Number of units', '駐車場在否': 'Parking lot availability', '駐車場月額': 'Parking lot monthly', '駐車場敷金（額）': 'Parking lot deposit (amount)', '駐車場敷金（ヶ月）': 'Parking deposit (months)', '駐車場礼金（額）': 'Parking key money (amount)', '駐車場礼金（ヶ月）': 'Parking key money (months)', '建物構造': 'Building Structure', '建物工法': 'Building construction method', '建物形式': 'Building type', '地上階層': 'Above ground floor', '地下階層': 'Underground floor', '所在階': 'Location floor', '築年月（西暦）': 'Building date (Western calendar)', '新築フラグ': 'New construction flag', 'バルコニー方向（1）': 'Balcony direction (1)', '売出〜最終補足までの日数': 'Number of days from sale to final supplement', 'branchデータ数': 'Number of branch data', '取扱不動産会社数': 'Number of real estate companies handled', '最終取扱不動産会社': 'Final real estate companies handled', '価格改定数': 'Number of price revisions', '価格騰落率': 'Price change rate', '価格騰落額': 'Price change amount', '登録日時': 'Registration date and time', '最終更新日時': 'Last updated date and time', 'マンションid': 'Apartment id', 'input_file_name': 'input_file_name', '_right__建物名#0': '_right_ _Building name #0', 'normalized_name': 'normalized_name', '_right__住所全体#1': '_right__Entire address #1', 'matching_status': 'matching_status', 'path': 'path', 'paired_address': 'paired_address ', 'normalized_address': 'normalized_address', 'prefecture': 'prefecture', 'city_ward': 'city_ward', 'town': 'town', 'subtown': 'subtown', 'street': 'street', 'block': 'block', 'number': 'number', 'subnumber': 'subnumber', 'lat': 'lat', 'lon': 'lon'}


# headerTranslator = {'[最低賃料]': 'rent', '築年月（西暦）': 'built_yyyymm', '使用部分面積': 'surface_area', '[最低賃料]平米単価': 'rent_per_square_meter', '物件種別': 'property_type1', '物件種目': 'property_type2', '[最寄駅]沿線名': 'nearest_station_line_name', '[最寄駅]駅名': 'nearest_station',  '[最寄駅]徒歩（分）': 'nearest_station_walking_minutes', '[最寄駅]徒歩（m）': 'nearest_station_walking_meters', '礼金1（額）': 'key_money_amount', '礼金2（ヶ月）': 'key_money_months', '敷金1（額）': 'deposit_amount', '敷金2（ヶ月）': 'deposit_months', '間取タイプ（1）': 'floor_plan_type', '建物形式': 'building_type', '納戸数': 'building_number_of_units', '地上階層': 'floors_above_ground', '地下階層': 'basement_floors', '所在階': 'floor_of_property'}

# # ###=== Read the full dataset, but only select columns
# useCols=['id','lat','lon', 'prefecture', 'city_ward', 'town', 'subtown', 'street', 'block', 'number', 'subnumber']+[k for k,v in headerTranslator.items()]
# roomData = pd.read_csv(filename, encoding='utf-8', usecols=useCols).fillna('')

# roomData.rename(columns=headerTranslator, inplace=True)
###---------------------------------------------------

# # print("-- Adding geometry column to data")
# # roomData['geometry'] = [Point(xy) for xy in zip(roomData.lon, roomData.lat)]
# # roomData = gp.GeoDataFrame(roomData, geometry='geometry')
# # roomData.crs = standardCRS

# # print("-- Filtering to Tokyo Main")
# # analysisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')
# # roomData = roomData[roomData['geometry'].intersects(analysisArea)]

# # print("-- Refining Data Columns")
# # roomData['built_year'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[0:4]))
# # roomData['built_month'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[4:6]))
# # roomData.drop(columns=['built_yyyymm', 'lat', 'lon'], inplace=True)


# roomData['age_in_months'] = roomData.apply(lambda row: makeInt((12 - row['built_month']) + (12 * (2023 - row['built_year']))), axis=1)
# roomData['age_in_years'] = roomData['built_year'].apply(lambda val: 2023 - row['built_year'] ))

#     def getDepositAmount(rowData):
#     if isNumber(rowData['deposit_amount']):
#         return int(rowData['deposit_amount'])
#     else isNumber(rowData['deposit_months']):
#         return int(rowData['deposit_months'] * rent)
#     else:
#         return 0

# def getKeyMoneyAmount(rowData):
#     if isNumber(rowData['key_money_amount']):
#         return int(rowData['key_money_amount'])
#     else isNumber(rowData['key_money_months']):
#         return int(rowData['key_money_months'] * rent)
#     else:
#         return 0

# roomData['deposit_amount'] = roomData.apply(lambda row: getDepositAmount(row), axis=1)
# roomData['key_money_amount'] = roomData.apply(lambda row: getKeyMoneyAmount(row), axis=1)

# def getHexID(theHexData, pointGeom):
#     theHexData = theHexData[theHexData['geometry'].intersects(pointGeom)]
#     return list(theHexData['id'])[0]

# roomData['hex_ID'] = roomData['geometry'].apply(lambda geom: getHexID(hexData, geom), axis=1)



# # print(roomData.head())

# # print("-- Saving Unfiltered Rooms")
# # writePickleFile(roomData,'../Demand Estimation/roomData_Full_TokyoMain_2.pkl')

# # print(len(roomData))  ##--> 6,921,693



# ###-------------------------------------------------------------
# print("-- Filtering Rooms to Desired Subset")
# roomData = roomData[((roomData['built_year'] < 2017) & (roomData['built_year'] >= 2003))]
# roomData = roomData[((roomData['surface_area'] < 30) & (roomData['surface_area'] >= 25))]

# print("-- Saving Filtered Rooms")
# writePickleFile(roomData,'../Demand Estimation/roomData_Filtered_TokyoMain.pkl')
# writeGeoCSV(roomData, '../Demand Estimation/roomData_Filtered_TokyoMain.csv')
# ###============================================================================





# # ###==============================================================================================
# # ###================================== ADD TIME TO STATIONS TO HEXES =============================
# # ###==============================================================================================

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v01d+jobs.pkl')

# stationNodes = getNodesByAttr(hexNetwork, 'modality', thisVal='station')
# hexNodes = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex'))]

# # print(getUniqueEdgeAttrValues(hexNetwork, 'modality'))
# stationsNoNameEN = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'station') & (attr.get('stationNameEN','poo') == 'poo'))]
# for thisStation in stationsNoNameEN:
#     print(hexNetwork.nodes[thisStation].get('name','poo'))
#     # print("--------------------")


# nonHexEdges = [(u,v) for u,v,attr in hexNetwork.edges(data=True) if not attr["modality"] in ['stationLink', 'hexLink']]
# hexOnlyNetwork = hexNetwork.copy()
# hexOnlyNetwork.remove_edges_from(nonHexEdges)
# # print(hexNetwork.number_of_edges())  ##--> 2,227,486
# # print(hexOnlyNetwork.number_of_edges())  ##--> 2,155,170 ##-- after removing the train related edges

# allTimesToStations = {n:{} for n in hexNodes}

# print("== Getting all times to stations.")
# # for thisStation in stationNodes[0:1]:
# for thisStation in stationNodes:
#     print("  -- This station:", hexNetwork.nodes[thisStation]['stationNameEN'])
#     timesToHexes = nx.single_source_dijkstra_path_length(hexOnlyNetwork, thisStation, cutoff=15, weight='weight')
#     # print(" -- timesToHexes:", timesToHexes)
#     # timesToHexes = {node:time for node,time in timesToHexes.items() if node in hexNodes}
#     timesToStations = {node:{thisStation:rnd(time,2)} for node,time in timesToHexes.items() if node in hexNodes}
#     # print(" -- timesToStations:", timesToStations)
#     theseNodes = [node for node,stationTimes in timesToStations.items()]
#     for thisNode in theseNodes:
#         for station,time in timesToStations[thisNode].items():
#             allTimesToStations[thisNode][station] = time


# hexesWithTimes = [n for n,attr in hexNetwork.nodes(data=True) if attr.get('times_to_stations',{}) != {}]
# for thisHex in hexesWithTimes:
#     hexNetwork.nodes[thisHex]['times_to_stations'] = sortDictByValue(allTimesToStations[thisHex], largerFirst=False)
#     hexNetwork.nodes[thisHex]['num_nearby_stations'] = len(allTimesToStations[thisHex])
#     closest_station_id = min(allTimesToStations[thisHex], key=allTimesToStations[thisHex].get)
#     hexNetwork.nodes[thisHex]['closest_station_id'] = closest_station_id
#     hexNetwork.nodes[thisHex]['closest_station_name'] = hexNetwork.nodes[closest_station_id].get('stationNameEN','error')
#     hexNetwork.nodes[thisHex]['closest_station_time'] = allTimesToStations[thisHex][closest_station_id]

# # print([(n,attr) for n,attr in hexNetwork.nodes(data=True) if attr.get('times_to_stations',{}) != {}])


# print("== Filling in times to hexes without nearby stations.")
# hexesTooFar = getNodesByAttr(hexNetwork, 'times_to_stations', thisVal={})
# print("  -- Hexes with no stations within 15 minutes", len(hexesTooFar))

# for thisHex in hexesTooFar:
#     timesToStations = nx.single_source_dijkstra_path_length(hexOnlyNetwork, thisHex, cutoff=90, weight='weight')
#     timesToStations = {node:time for node,time in timesToStations.items() if node in stationNodes}
#     # timesToStations = sortDictByValue(timesToStations, largerFirst=False)
#     closest_station_id = min(timesToStations, key=timesToStations.get)
#     hexNetwork.nodes[thisHex]['times_to_stations'] = {closest_station_id:timesToStations[closest_station_id]}
#     hexNetwork.nodes[thisHex]['num_nearby_stations'] = 0
#     closest_station_id = min(allTimesToStations[thisHex], key=allTimesToStations[thisHex].get)
#     hexNetwork.nodes[thisHex]['closest_station_id'] = closest_station_id
#     hexNetwork.nodes[thisHex]['closest_station_name'] = hexNetwork.nodes[closest_station_id].get('stationNameEN','error')
#     hexNetwork.nodes[thisHex]['closest_station_time'] = allTimesToStations[thisHex][closest_station_id]


# hexesTooFar = getNodesByAttr(hexNetwork, 'times_to_stations', thisVal={})
# print("  -- Remaining hexes with no station", len(hexesTooFar))

# for thisHex in hexesTooFar:
#     hexNetwork.nodes[thisHex]['times_to_stations'] = {}
#     hexNetwork.nodes[thisHex]['num_nearby_stations'] = 0
#     hexNetwork.nodes[thisHex]['closest_station_id'] = None
#     hexNetwork.nodes[thisHex]['closest_station_name'] = None
#     hexNetwork.nodes[thisHex]['closest_station_time'] = None




# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v01d+jobs+stationTimes.pkl')

# convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v01d+jobs+stationTimes")

# # nodeDF,edgeDF = convertGraphToGeopandas(hexNetwork)
# # hexData = nodeDF[nodeDF['modality']=='hex']
# # print(hexData.head())
# # print(edgeDF.head())




# # ###==============================================================================================
# # ###================================== HEDONIC PRICING ANALYSIS =============================
# # ###==============================================================================================

def getDeposit(rowData):
    if isNumber(rowData['deposit_amount']):
        if rowData['deposit_amount'] > 0:
            return makeInt(rowData['deposit_amount'])
    elif isNumber(rowData['deposit_months']):
        if rowData['deposit_months'] > 0:
            return makeInt(rowData['deposit_months'] * rowData['rent'])
    else:
        return 0

def getKeyMoney(rowData):
    if isNumber(rowData['key_money_amount']):
        if rowData['key_money_amount'] > 0:
            return makeInt(rowData['key_money_amount'])
    elif isNumber(rowData['key_money_months']):
        if rowData['key_money_months'] > 0:
            return makeInt(rowData['key_money_months'] * rowData['rent'])
    else:
        return 0

###== this function distrubutes the key money over two years to get a better measure of renter cost
def getAdjustedRent(row):
    if row['key_money'] > 0:
        return makeInt(row['rent'] + (row['key_money'] / 24))
    else:
        return makeInt(row['rent'])


###---------------------------
hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores2.pkl')
nodeDF,edgeDF = convertGraphToGeopandas(hexNetwork)

# print(type(nodeDF.at[1,'closest_station_name']))
# print(uniqueItems(list(nodeDF['closest_station_name'])))

hexData = nodeDF.loc[((nodeDF['modality']=='hex') & (nodeDF['connected']==True))].copy()
hexData = hexData.loc[hexData['closest_station_name'].apply(lambda val: True if isinstance(val,str) else False)].copy()
# print(hexData.head())
# print(list(hexData))
# print("Number of hexes with data in Tokyo Main:", len(hexData))  ##--> 127,259

hexData = hexData[['geometry', 'closest_station_name', 'closest_station_nameEN', 'reachable_stations', 'closest_station_id', 'closest_station_time', 'num_nearby_stations', 'reachable_hexes_15min', 'total_stores', 'total_relevant_stores', 'store_score', 'demand_sCurve-90_05', 'demand_sCurve-90_10']]


roomData = readPickleFile('../Demand Estimation/roomData_Full_TokyoMain_2.pkl')
# print(roomData.head())
# print(list(roomData))
# print("Number of unfiltered rooms in Tokyo Main:", len(roomData))  ##--> 6,921,693

# roomData['surface_area'] = roomData['surface_area'].fillna(0)
# print(len(roomData.loc[roomData['surface_area'] == 0]))
###===> ALl rooms have rent and month and surface_area

# roomData['key_money_amount'] = roomData['key_money_amount'].fillna(0)
# roomData['key_money_months'] = roomData['key_money_months'].fillna(0)
roomData['key_money'] = roomData.apply(lambda row: getKeyMoney(row), axis=1)

# roomData['deposit_amount'] = roomData['deposit_amount'].fillna(0)
# roomData['deposit_months'] = roomData['deposit_months'].fillna(0)
roomData['deposit'] = roomData.apply(lambda row: getKeyMoney(row), axis=1)

roomData['adj_rent'] = roomData.apply(lambda row: getAdjustedRent(row), axis=1)
roomData['adj_rent_per_square_meter'] = roomData.apply(lambda row: row['adj_rent'] / row['surface_area'], axis=1)
roomData['log_adj_rent_per_square_meter'] = roomData.apply(lambda row: safeLog(row['adj_rent_per_square_meter']), axis=1)

roomData['age_in_months'] = roomData.apply(lambda row: makeInt((12 - row['built_month']) + (12 * (2023 - row['built_year']))), axis=1)
roomData['age_in_years'] = roomData['built_year'].apply(lambda val: 2023 - val)

roomData = roomData[['geometry', 'id', 'rent', 'adj_rent', 'rent_per_square_meter',  'adj_rent_per_square_meter', 'key_money', 'deposit', 'surface_area', 'floor_plan_type', 'floor_of_property', 'built_year', 'built_month', 'age_in_years', 'age_in_months', 'prefecture', 'city_ward', 'town', 'subtown', 'street', 'block', 'number', 'subnumber', 'property_type1', 'property_type2', 'nearest_station', 'nearest_station_line_name']]

# roomData['hex_ID'] = roomData['geometry'].apply(lambda geom: getHexID(hexData, geom), axis=1)

# print("Number of unfiltered rooms in Tokyo Main:", len(roomData))  ##--> 6,921,693
# roomData2 = roomData[isNumber(roomData['log_adj_rent_per_square_meter'])]
# roomData2 = roomData2[roomData2['log_adj_rent_per_square_meter'] > 0]
# print("Number of rooms with valid log adjusted prices per sq meter:", len(roomData2))  ##-->

writePickleFile(roomData,'../Demand Estimation/roomData_Full_TokyoMain_3.pkl')

joinedData = roomData.sjoin(hexData, how="left")

writePickleFile(joinedData,'../Demand Estimation/room+hex_data_TokyoMain_2c.pkl')


print(joinedData.head())















###======================================== END OF FILE ===========================================