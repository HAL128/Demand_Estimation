# -*- coding: utf-8 -*-
from Codebase.helpers.helper_functions import *
from helpers.database_helpers import *



dbConnection = DatabaseConnInfo(username='data_warehouse_owner', password='3x4mp13us3r')
###============================================================================

###=== get the centroid of a polygon or other geometry using the angle-preserving CRS and recast back to standard CRS
def getCentroid(geom):
   return convertGeomCRS((convertGeomCRS(geom, standardToMapProj)).centroid, mapToStandardProj)

###============================================================================


# category ['eatery', 'healthcare', 'lodging', 'police', 'retail', 'fire_station', 'education', 'social', 'cultural', 'parking', 'vending_machine', 'entertainment']

# subcategory ['shopping', 'pharmacy', 'leisure', 'recreation', 'gas_station', 'convenience_store', 'hospital', 'bicycle_parking', 'police', 'religion', 'pub', 'restaurant', 'cafe', 'post_office', 'fitness', 'beauty', 'cinema', 'service', 'motorcycle_parking', 'car_parking', 'science', 'bar', 'community', 'primary-school', 'park', 'specialty_food', 'art', 'laundry', 'post_secondary_school', 'clinic', 'spa', 'bank', 'fire_station', 'hotel', 'supermarket', 'gambling', 'vending_machine', 'smoking_area', 'amenity']

# storeType ['optician', 'greengrocer', 'playground', 'doctor', 'travel_agency', 'gas_station', 'garden', 'convenience_store', 'police', 'marina', 'cafe', 'seafood', 'golf_course', 'home_center', 'community_center', 'car_repair', 'park', 'alcohol', 'camp_site', 'bank', 'sports_area', 'car_dealership', 'sporting_goods', 'attraction', 'car_rental', 'kindergarten', 'bakery', 'pet', 'bicycle_shop', 'gift_shop', 'massage', 'college', 'gym', 'hospital', 'bicycle_parking', 'pub', 'fast_food', 'pastry', 'bar', 'miniature_golf', 'bowling_alley', 'laundry', 'swimming_pool', 'butcher', 'supermarket', 'game_center', 'deli', 'school', 'library', 'pharmacy', 'electronics', 'religion', 'gardening', 'restaurant', 'post_office', 'zoo', 'fishing', 'beach_resort', 'cinema', 'confectionery', 'picnic_site', 'artwork', 'museum', 'car_parts', 'furniture', 'vending_machine', 'townhall', 'food_court', 'motorcycle', 'dentist', 'childcare', 'books', 'charging_station', 'clothing_store', 'dry_cleaning', 'motorcycle_parking', 'veterinary', 'car_parking', 'antiques', 'pachinko', 'bicycle_rental', 'clinic', 'atm', 'hotel', 'fire_station', 'salon', 'smoking_area']



# livableTypes = ['optician', 'greengrocer', 'playground', 'doctor', 'travel_agency', 'convenience_store', 'cafe', 'seafood', 'golf_course', 'home_center', 'community_center', 'park', 'alcohol', 'camp_site', 'bank', 'sports_area', 'sporting_goods', 'attraction', 'kindergarten', 'bakery', 'pet', 'bicycle_shop', 'gift_shop', 'massage', 'college', 'gym', 'hospital', 'pub', 'fast_food', 'pastry', 'bar', 'miniature_golf', 'bowling_alley', 'laundry', 'swimming_pool', 'butcher', 'supermarket', 'game_center', 'deli', 'school', 'library', 'pharmacy', 'electronics', 'religion', 'gardening', 'restaurant', 'post_office', 'zoo', 'fishing', 'beach_resort', 'cinema', 'confectionery', 'picnic_site', 'museum', 'car_parts', 'furniture', 'townhall', 'food_court', 'dentist', 'childcare', 'books', 'clothing_store', 'dry_cleaning', 'veterinary', 'antiques', 'pachinko', 'clinic', 'salon']

# landUseCats = ['agriculture', 'forest', 'sea', 'high-rise building', 'low-rise dense', 'rice field', 'beach', 'park', 'water', 'low-rise sparse', 'facility', 'wasteland', 'golf', 'rail', 'vacant', 'factory', 'road']

# zoningCatsDict = {'商業地域':'Commercial', '工業地域':'industrial', '工業専用地域':'industrial', '準住居地域':'residential_1', '準工業地域':'industrial', '第一種中高層住居専用地域':'residential_1', '第一種低層住居専用地域':'residential_1', '第一種住居地域':'residential_1', '第二種中高層住居専用地域':'residential_2', '第二種低層住居専用地域':'residential_2', '第二種住居地域':'residential_2', '近隣商業地域':'commercial'}

# zoningCatsTrans = {'商業地域':'commercial', '工業地域':'industrial', '工業専用地域':'industrial_specific', '準住居地域':'semi-residential', '準工業地域':'semi-industrial', '第一種中高層住居専用地域':'residential_class1_highrise', '第一種低層住居専用地域':'residential_class1_lowrise', '第一種住居地域':'residential_class1', '第二種中高層住居専用地域':'residential_class2_highrise', '第二種低層住居専用地域':'residential_class2_lowrise', '第二種住居地域':'residential_class2', '近隣商業地域':'neighborhood_commercial'}
    
# vegCatTrans = {'その他':'other', '公園・墓地':'park_cemetery', '工場地帯':'factory', '市街地':'urban', '樹林・草地':'forest_grassland', '田畑':'field', '空地':'vacant', '緑の多い住宅地':'green_residential', '芝地':'lawn', '造成地':'development'}



### transform json file into dataframe ===========================================

# ### import univeristy data from json file =======================================
# universityData = pd.read_json('../Data/TokyoMain_Hirata/university_uc.json')
# geometry = [Point(xy) for xy in zip(universityData['longitude'], universityData['latitude'])]
# universityData = gp.GeoDataFrame(universityData, geometry=geometry)
# universityData.set_crs(epsg=4326, inplace=True)

# ### import stadium data from json file =======================================
# stadiumData = pd.read_json('../Data/TokyoMain_Hirata/stadium.json')
# geometry = [Point(xy) for xy in zip(stadiumData['longitude'], stadiumData['latitude'])]
# stadiumData = gp.GeoDataFrame(stadiumData, geometry=geometry)
# stadiumData.set_crs(epsg=4326, inplace=True)

# ### import shinkansen data from json file =======================================
# shinkansenData = pd.read_json('../Data/TokyoMain_Hirata/shinkansen.json')
# geometry = [Point(xy) for xy in zip(shinkansenData['longitude'], shinkansenData['latitude'])]
# shinkansenData = gp.GeoDataFrame(shinkansenData, geometry=geometry)
# shinkansenData.set_crs(epsg=4326, inplace=True)

# ### import movie_theater data from json file =======================================
# movietheaterData = pd.read_json('../Data/TokyoMain_Hirata/movie_theater.json')
# geometry = [Point(xy) for xy in zip(movietheaterData['longitude'], movietheaterData['latitude'])]
# movietheaterData = gp.GeoDataFrame(movietheaterData, geometry=geometry)
# movietheaterData.set_crs(epsg=4326, inplace=True)

# ### import airport data from json file =======================================
# airportData = pd.read_json('../Data/TokyoMain_Hirata/airport.json')
# geometry = [Point(xy) for xy in zip(airportData['longitude'], airportData['latitude'])]
# airportData = gp.GeoDataFrame(airportData, geometry=geometry)
# airportData.set_crs(epsg=4326, inplace=True)

### import hospitals data from json file =======================================
hospitalsData = pd.read_json('../Data/TokyoMain_Hirata/hospital.json')
geometry = [Point(xy) for xy in zip(hospitalsData['longitude'], hospitalsData['latitude'])]
hospitalsData = gp.GeoDataFrame(hospitalsData, geometry=geometry)
hospitalsData.set_crs(epsg=4326, inplace=True)



###===========================================================================================
###================================= SETUP SOURCE DEMAND =====================================
###===========================================================================================

# print("== Starting Setup of Source Demand")
thisArea = getPolygonForArea('tokyoMain')

##load the hex network and job data, and resample the latter into the former.
hexNetwork = readPickleFile('../Data/hexNetwork_v02c+jobs+stationTimes+stores+other3.pkl')

hexNodes = [node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('connected','poo') == True))]
# print("  -- Number of connected hex nodes:", len(hexNodes))  ##==> 126440  --> 126865  --> 129793

hexNodeData,hexEdgeData = convertGraphToGeopandas(hexNetwork)
hexNodeData = hexNodeData[hexNodeData['modality']=='hex']
hexNodeData = hexNodeData[hexNodeData['connected']==True]
hexNodeData = hexNodeData.reset_index(drop=True)
# print(hexNodeData.head())
# print("  -- Number of connected hex nodes:", len(hexNodeData))  ##--> confirmed, the same number


### hospitalsData ============================================================================

for thisHex in hexNodes:
    hexNetwork.nodes[thisHex]['sourceHospitals'] = 0
    hexNetwork.nodes[thisHex]['demand'] = 0


###=== Assign initial demand from jobs grid to hex grid
for index, row in hospitalsData.iterrows():
    num_doctors = strToNum(row['doctor'])
    if num_doctors > 0:
        thisPoint = row['geometry']  # Correctly get the geometry of the current hospital
        theseHexes = hexNodeData[hexNodeData['geometry'].intersects(thisPoint)]
        # print(theseHexes)
        
        ##-- in most cases, exactly one hex should intersect the centroid of each grid.
        ##-- if there are multiple matches, then it's exactly on a boundary, so just do the first (vs random) one.
        if len(theseHexes) >= 1:
            thisHex = list(theseHexes['id'])[0]
            hexNetwork.nodes[thisHex]['sourceHospitals'] += num_doctors
        ##-- if there is no connected hex here, then we really want the closest one.
        else:
            print("  !! There is no hex intersecting the hospital. Finding the nearest hex.")
            distances = hexNodeData['geometry'].apply(lambda geom: geom.distance(thisPoint))
            nearest_hex_index = distances.idxmin()
            nearest_hex = hexNodeData.loc[nearest_hex_index]
            hexNetwork.nodes[nearest_hex['id']]['sourceHospitals'] += num_doctors
            print(f"  Added {num_doctors} doctors to the nearest hex: {nearest_hex['id']}")
            




# ### jobsData ============================================================================

# jobsData = get_data_for_geom(dbConnection, table="economics2016", geom=thisArea)
# print(jobsData.head())
# jobsData['centroid'] = jobsData['geometry'].apply(lambda geom: getCentroid(geom))

# for thisHex in hexNodes:
#     hexNetwork.nodes[thisHex]['sourceJobs'] = 0
#     hexNetwork.nodes[thisHex]['demand'] = 0

# ###=== Assign initial demand from jobs grid to hex grid
# for index,row in jobsData.iterrows():
#     theseJobs = strToNum(row['employees_AR_all_industries'])
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

# ###=== setting a common edge weight
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

# convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v2c+jobs-source")

# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v2c+jobs-source.pkl')
# ###============================================================================



###===========================================================================================
###================================= PROCESS DEMAND FLOW =====================================
###===========================================================================================

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





print("== Starting Demand Flow Process")



# # ##=== setting a common edge weight ==== This should already be done in the network you have.
# for n1,n2,attr in hexNetwork.edges(data=True):
#     ###--- first set all the edges to and from disconnected hex nodes to None
#     if ((hexNetwork.nodes[n1].get('connected',True) == False) | (hexNetwork.nodes[n2].get('connected',True) == False)):
#         hexNetwork[n1][n2]['weight'] = None 
#     else:  ###--- then set NaN weights for either weight type to None
#         if np.isnan(attr.get('timeWeight',np.nan)):
#             hexNetwork[n1][n2]['timeWeight'] = None        
#         if np.isnan(attr.get('walkTime',np.nan)):
#             hexNetwork[n1][n2]['walkTime'] = None        
        
#         ###-- Then test for which weight actuallu has a value, and use that one
#         if ((attr.get('timeWeight',None) != None) & (attr.get('walkTime',None) != None)):
#             print("both weights exist in edge type", attr.get('modality'), "- using timeWeight:", attr.get('timeWeight',None), "instead of walkTime:", attr.get('walkTime',None) )
#             hexNetwork[n1][n2]['weight'] = attr.get('timeWeight',None)
#         elif attr.get('timeWeight',None) != None:
#             hexNetwork[n1][n2]['weight'] = attr.get('timeWeight',None)
#         elif attr.get('walkTime',None) != None:
#             hexNetwork[n1][n2]['weight'] = attr.get('walkTime',None)
#         else:
#             print("some other weight exists for", attr.get('modality'), "edge with timeWeight =", attr.get('timeWeight',None), ", walkTime =", attr.get('walkTime',None), ", and weight =", attr.get('weight',None))
        


# print("both", len( [u for u,v,.get('modality') in hexNetwork.edges(data=True) if ((attr.get('timeWeight',None) != None) & (attr.get('walkTime',None) != None))])) 

numEdges = len([u for u,v,a in hexNetwork.edges(data=True)])
numNone = len([u for u,v,a in hexNetwork.edges(data=True) if a.get('weight',None) is None])
numNumber = len([u for u,v,a in hexNetwork.edges(data=True) if isNumber(a.get('weight',None))])

print("Number of edges", numEdges)  ##--> 2,374,291 edges in total

print("weight is None", len([u for u,v,a in hexNetwork.edges(data=True) if a.get('weight',None) == None]))
print("weight is None", numNone)  ##-- 
print("numNumber", numNumber, "+ numNone", numNone, "=", numNumber+numNone, "out of", numEdges, "total edges"  )



hexNodes = [node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('connected',False) == True))]
print("  == Number of hexes:", len(hexNodes))  ##--> 126440 --> 126865
sourceNodes = sorted([node for node,attr in hexNetwork.nodes(data=True) if ((attr.get('modality') == 'hex') & (attr.get('sourceHospitals',0) > 0))])
print("  == Number of source hex nodes:", len(sourceNodes))  ##--> 17201  --> 17206
numSources = len(sourceNodes)

print(getAllEdgeAttributes(hexNetwork))
###==> ['timeWeight', 'lineName', 'weight', 'routeID', 'length', 'modality', 'geomDist', 'color', 'walkTime', 'walkingSpeed', 'direction', 'distance', 'lineType', 'geomMap', 'lineGeom', 'geometry', 'throughService']

listOfWeightings = ['sCurve-30_10', 'linear-60', 'sCurve-60_05', 'sCurve-60_10', 'sCurve-60_20', 'linear-90', 'sCurve-90_05', 'sCurve-90_10', 'sCurve-90_20']  ## 'linear-60', 'sCurve-60_05', 'sCurve-60_10', 'sCurve-60_20', 'linear-90', 'sCurve-90_05', 'sCurve-90_10', 'sCurve-90_20'
weightList = [0, 0.5, 1, 2]  ## 0, 0.5, 1, 2
weightNames = ["00", "05", "10", "20"]  ## "00", "05", "10", "20"

###=== Setup all the nodes to have zero values for all weights.
for thisHex,hexAttr in hexNetwork.nodes(data=True):
    for thisWeight in listOfWeightings:
        # hexNetwork.nodes[thisHex]['demand_'+thisWeight] = 0
        hexNetwork.nodes[thisHex]['demandHospital_'+thisWeight] = 0  ## 追加

###=== For each source node, propagate the demand to all the other nodes.
runStartTime = time.time()
for index,source in enumerate(sourceNodes):
    runStartTime = printProgress(runStartTime, index, numSources)
    ###=== The original demand from this source
    sourceDemand = hexNetwork.nodes[source]['sourceHospitals']
    ###=== Get the time to all other nodes withing 90 minutes
    allTimes = nx.single_source_dijkstra_path_length(hexNetwork, source, cutoff=90, weight='weight')
    ###=== Now accumulate the weighted demand to all reachable hexes based on each weighting apporach
    for node,time in allTimes.items():
        hexNetwork.nodes[node]['demandHospital_sCurve-30_10'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=30)
        hexNetwork.nodes[node]['demandHospital_linear-90'] += weightedDemand(sourceDemand, time, function='linear', maxLimit=90)
        hexNetwork.nodes[node]['demandHospital_sCurve-90_05'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=0.5)
        hexNetwork.nodes[node]['demandHospital_sCurve-90_10'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=1.0)
        hexNetwork.nodes[node]['demandHospital_sCurve-90_20'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=90, curvature=2.0)
        # if time <= 60:
        hexNetwork.nodes[node]['demandHospital_linear-60'] += weightedDemand(sourceDemand, time, function='linear', maxLimit=60)
        hexNetwork.nodes[node]['demandHospital_sCurve-60_05'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=0.5)
        hexNetwork.nodes[node]['demandHospital_sCurve-60_10'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=1.0)
        hexNetwork.nodes[node]['demandHospital_sCurve-60_20'] += weightedDemand(sourceDemand, time, function='sCurve', maxLimit=60, curvature=2.0)

###=== Save the results for further analysis
writePickleFile(hexNetwork,'../Data/Output/hexNetwork_v02c+hospitals.pkl')

###=== Output for visualization in Kepler
convertNetworkToKeplerFiles(hexNetwork, filename="hexNetwork_v02c+hospitals")
##============================================================================


print('Done!')




# ###==============================================================================================
# ###================================== ADD TIME TO STATIONS TO HEXES =============================
# ###==============================================================================================

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs.pkl')

# hexNodes = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex'))]
# for node in hexNodes:
#     hexNetwork.nodes[node]['nearest_stations'] = ast.literal_eval(hexNetwork.nodes[node].get('nearest_stations','{}'))
    
# # print("== Connected Hexes in area", len(connectedNodes))  ##--> 137221
    
# nodesWithStations = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo')=='hex') & (attr.get('nearest_stations',{}) != {}))]
# print("  -- Hex nodes with stations", len(nodesWithStations))  ##--> 43232

# nodesWithoutStations = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo')=='hex') & (attr.get('nearest_stations',{})=={}))]
# print("  -- Hex nodes without stations", len(nodesWithoutStations))  ##--> 93989

# # ###--- confirm that the hex distances vary, and I found that zero-distance links remain
# # hexDistances = [convertNanToNone(attr.get('distance',None)) for u,v,attr in hexNetwork.edges(data=True) if attr["modality"] == 'hex']
# # hexDistances = [val for val in hexDistances if not val is None]
# # print(len(hexDistances))
# # print(hexDistances[:100])

# ###--- remove the stupid zero-length edges from the hex network
# zeroDistHexEdges = [(u,v) for u,v,attr in hexNetwork.edges(data=True) if ((attr["modality"] == 'hex') & (convertNanToNone(attr.get('distance',None))==0))]
# hexNetwork.remove_edges_from(zeroDistHexEdges)

# # ###--- Confirm the weights have been assigned correctly (they have)
# # hexWeights = [convertNanToNone(attr.get('weight',None)) for u,v,attr in hexNetwork.edges(data=True) if attr["modality"] == 'hex']
# # hexWeights = [val for val in hexWeights if not val is None]
# # print(hexWeights[:100])
# # hexWalkTimes = [convertNanToNone(attr.get('walkTime',None)) for u,v,attr in hexNetwork.edges(data=True) if attr["modality"] == 'hex']
# # hexWalkTimes = [val for val in hexWalkTimes if not val is None]
# # print(hexWalkTimes[:100])


# ### if code does not run due to this definition, delete this
# def convertNanToNone(value):
#     if isinstance(value, (float, np.float64)) and np.isnan(value):
#         return None
#     return value



# ###--- because hex edge lengths/distances vary in this projection, get the average time per meter instead of just time
# def getTimePerMeter(att):
#     thisWeight = convertNanToNone(att.get('weight',None))
#     thisDistance = convertNanToNone(att.get('distance',None))
#     if thisWeight is None:
#         return None
#     elif thisDistance is None:
#         print("   ++ Edge has no distance")
#         return None
#     elif thisWeight == 0:
#         return None
#     elif thisDistance == 0:
#         return None
#     else:        
#         return thisWeight / thisDistance


# hexEdgeTimePerMeter = [getTimePerMeter(attr) for u,v,attr in hexNetwork.edges(data=True) if attr["modality"] == 'hex']
# hexEdgeTimePerMeter = [val for val in hexEdgeTimePerMeter if not val is None]
# print("  -- hex edges", len(list(hexNetwork.edges())))  ##--> 2237070
# print("  -- hex edges with numerical weights", len(hexEdgeTimePerMeter))  ##--> 2152810
# print("  -- default walking minutes per meter", 1 / 80)  ##--> 0.0125
# print("  -- min hex edge minutes per meter", min(hexEdgeTimePerMeter))  ##--> 0.01248
# print("  -- mean hex edge minutes per meter", np.mean(hexEdgeTimePerMeter))  ##--> 0.021333280992133396
# print("  -- median hex edge minutes per meter", np.median(hexEdgeTimePerMeter))  ##--> 0.01916
# print("  -- max hex edge minutes per meter", max(hexEdgeTimePerMeter))  ##-->  0.06962818548948614

# # print(hexEdgeTimePerMeter[:100])

# makeHistogramPlot(hexEdgeTimePerMeter, 'minutes per meter', 'count', titleText='minutes per meter for hex links', numBin=40, colorScheme=None, figSize=5, lowerBound=0, upperBound=None, filename='../Demand Estimation/histogram of minutes per meter for hex links.png')

# medianMinutesPerMeter = np.median(hexEdgeTimePerMeter)


# # print(type(hexNetwork.nodes[nodesWithStations[10]]['nearest_stations']))
# # print(hexNetwork.nodes[nodesWithStations[10]]['nearest_stations'])

# # print(getUniqueEdgeAttrValues(hexNetwork, 'modality'))


# # stationsNoNameEN = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'station') & (attr.get('stationNameEN','poo') == 'poo'))]
# # for thisStation in stationsNoNameEN:
# #     print(hexNetwork.nodes[thisStation].get('name','poo'))
# #     # print("--------------------")


# # for node,attr in hexNetwork.nodes(data=True):
# #     hexNetwork.nodes[thisHex]['times_to_stations'] = {}

# nonHexEdges = [(u,v) for u,v,attr in hexNetwork.edges(data=True) if not attr["modality"] in ['hexStation', 'hex']]
# hexOnlyNetwork = hexNetwork.copy()
# hexOnlyNetwork.remove_edges_from(nonHexEdges)
# # print(hexNetwork.number_of_edges())  ##--> 2,227,486
# # print(hexOnlyNetwork.number_of_edges())  ##--> 2,155,170 ##-- after removing the train related edges

# ###=== Assign weight value to those non-connected hex edges to train time assignments
# for n1,n2,attr in hexOnlyNetwork.edges(data=True):
#     thisDistance = convertNanToNone(attr.get('distance',None))
#     if not isNumber(attr.get('weight',None)):
#         hexNetwork[n1][n2]['weight'] = None
#         hexNetwork[n1][n2]['approxTime'] = medianMinutesPerMeter * thisDistance
#         hexOnlyNetwork[n1][n2]['weight'] = medianMinutesPerMeter * thisDistance        
# ###-------------- weights assigned to virtual links -------------------        
        
# ###=== I created a variable called 'nearest_stations' that stores a dictionary of station names to times
# ###-- It is based on the path on the road network, but it only has one station by name, and some are multiple (like Shibuya)

# ###=== For each station, get the time from each hex to that station within 90 minutes
# print("== Getting times from all stations to all hexes within 15 minutes.")
# stationNodes = getNodesByAttr(hexNetwork, 'modality', thisVal='station')
# hexNodes = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex'))]
# allTimesToStations = {n:{} for n in hexNodes}
# # for thisStation in stationNodes[0:1]:
# for thisStation in stationNodes:
#     # print("  -- This station:", hexNetwork.nodes[thisStation]['stationNameEN'])
#     timesToHexes = nx.single_source_dijkstra_path_length(hexOnlyNetwork, thisStation, cutoff=90, weight='weight')
#     # print(" -- timesToHexes:", timesToHexes)
#     # timesToHexes = {node:time for node,time in timesToHexes.items() if node in hexNodes}
#     timesToStations = {node:{thisStation:rnd(time,2)} for node,time in timesToHexes.items() if node in hexNodes}
#     # print(" -- timesToStations:", timesToStations)
#     theseNodes = [node for node,stationTimes in timesToStations.items()]
#     for thisNode in theseNodes:
#         for station,time in timesToStations[thisNode].items():
#             allTimesToStations[thisNode][station] = time

# ###----------------------------------------------------------
# ###=== Now distribute and filter those times station times depending on the situation
# stationNodes = {n:attr for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'station'))}
# hexNodes = [(n,attr) for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex'))]

# for thisHex,attr in hexNodes:
    
#     times_to_stations = sortDictByValue(allTimesToStations.get(thisHex,{}), largerFirst=False)
#     if times_to_stations == {}:
#         hexNetwork.nodes[thisHex]['nearby_stations'] = {}
#         hexNetwork.nodes[thisHex]['num_nearby_stations'] = 0
#         hexNetwork.nodes[thisHex]['closest_station_id'] = None
#         hexNetwork.nodes[thisHex]['closest_station_name'] = None
#         hexNetwork.nodes[thisHex]['closest_station_nameEN'] = None
#         hexNetwork.nodes[thisHex]['closest_station_time'] = None
#     else:
#         hexNetwork.nodes[thisHex]['nearby_stations'] = times_to_stations
#         hexNetwork.nodes[thisHex]['num_nearby_stations'] = len([k for k,v in times_to_stations.items() if v <= 15])
#         closest_station_id = min(times_to_stations, key=times_to_stations.get)
#         hexNetwork.nodes[thisHex]['closest_station_id'] = closest_station_id
#         hexNetwork.nodes[thisHex]['closest_station_name'] = stationNodes[closest_station_id].get('stationName','error')
#         hexNetwork.nodes[thisHex]['closest_station_nameEN'] = stationNodes[closest_station_id].get('stationNameEN','error')
#         hexNetwork.nodes[thisHex]['closest_station_time'] = times_to_stations[closest_station_id]
    
    
# hexesWithTimes = [n for n,attr in hexNetwork.nodes(data=True) if attr.get('nearby_stations',{}) != {}]
# print("  -- Hexes with times:", len(hexesWithTimes))
# hexesWithoutTimes = [n for n,attr in hexNetwork.nodes(data=True) if attr.get('nearby_stations',{}) == {}]
# print("  -- Hexes without times:", len(hexesWithoutTimes))
   

# writePickleFile(hexNetwork,'../Data/Output/hexNetwork_v02c+hospitals.pkl')
# convertNetworkToKeplerFiles(hexNetwork, filename="hexNetwork_v02c+hospitals")


# ###===============================================================










# ###----------------------------------------------------------
# ##--- Compare the hex times to stations to the road times to stations:
    



















# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes.pkl')
# connectedNodes = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex') & (attr.get('connected',False)==True))]
# someNode = connectedNodes[0]
# print(someNode)
# print(hexNetwork.nodes[someNode]['closest_station_name'])

# nodesWithStations = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('nearest_stations',{}) != {}))]
# print(type(hexNetwork.nodes[nodesWithStations[10]]['nearest_stations']))
# print(len(nodesWithStations))

# nodesWithoutStations = [n for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo')=='hex') & (attr.get('nearest_stations','poo')=='poo'))]
# print(len(nodesWithoutStations))






# # nodeDF,edgeDF = convertGraphToGeopandas(hexNetwork)
# # hexData = nodeDF[nodeDF['modality']=='hex']
# # print(hexData.head())
# # print(edgeDF.head())



















# ##===========================================================================================
# ##================================= ANALYZE DEMAND AND RENT =================================
# ##===========================================================================================

# print("== Starting Analysis of Demand and Rent")

# roomData = readPickleFile('/Users/h_hirata/Library/CloudStorage/GoogleDrive-h_hirata@ga-tech.co.jp/.shortcut-targets-by-id/1cU31L7UYory6l84YC2Zknd66dAZUtzOh/Demand_Estimation_2/Data/roomData_TokyoMain_4a_fullAddress.csv')

# print("-- Filtering Rooms to Desired Subset")
# roomData = roomData[((roomData['built_year'] >= 2010) & (roomData['built_year'] <= 2014))]
# roomData = roomData[((roomData['surface_area']  >= 25) & (roomData['surface_area'] <= 28))]
# # print(roomData.head())
# print("Number of selected rooms in area:", len(roomData))  ##==> 702,005  --> 356,744

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes.pkl')
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

# writePickleFile(hexData,'../Demand Estimation/hexData_with_prices_v02c.pkl')
# writeGeoCSV(hexData, '../Demand Estimation/hexData_with_prices_v02c.csv')
# ##===========================================================================================















# # ##===========================================================================================
# hexData = readPickleFile('../Demand Estimation/hexData_with_prices_v02c.pkl')

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

#     makeScatterPlot(predVals, truthVals, xLabel='estimated demand', yLabel='mean rent', titleText=' '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_2c_results/price_vs_'+thisWeight+'.png')

#     ###=== Get the residual values and plot them as a function of X
#     regressionModel = LinearRegression().fit(np.array(predVals).reshape(-1, 1), truthVals)
#     prediction = regressionModel.predict(np.array(predVals).reshape(-1, 1))
#     residual_values = (truthVals - prediction)

#     # OLS_model = sm.OLS(truthVals,predVals).fit()  # training the model
#     # predicted_values = OLS_model.predict()  # predicted values
#     # residual_values = OLS_model.resid # residual values

#     makeScatterPlot(predVals, residual_values, xLabel='estimated demand', yLabel='residual', titleText='residuals '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_2c_results/price_vs_'+thisWeight+'_residuals.png')

#     makeHistogramPlot(residual_values, 'residual values', 'count', titleText='Residuals '+thisWeightLab, colorScheme=None, figSize=5, lowerBound=None, upperBound=None, filename='../Demand Estimation/hex_network_2c_results/price_vs_'+thisWeight+'_residual_histogram.png')

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

#     makeScatterPlot(predVals, truthVals, xLabel='estimated demand', yLabel='log mean rent', titleText=' '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_2c_results/log-price_vs_'+thisWeight+'.png')

#     ###=== Get the residual values and plot them as a function of X
#     regressionModel = LinearRegression().fit(np.array(predVals).reshape(-1, 1), truthVals)
#     prediction = regressionModel.predict(np.array(predVals).reshape(-1, 1))
#     residual_values = (truthVals - prediction)

#     # OLS_model = sm.OLS(truthVals,predVals).fit()  # training the model
#     # predicted_values = OLS_model.predict()  # predicted values
#     # residual_values = OLS_model.resid # residual values

#     makeScatterPlot(predVals, residual_values, xLabel='estimated demand', yLabel='residual', titleText='residuals '+thisWeightLab, dotSize=1, useCircles=False, colorScheme=None, figSize=5, scaledAxes=False, bestFit=True, filename='../Demand Estimation/hex_network_2c_results/log-price_vs_'+thisWeight+'_residuals.png')

#     makeHistogramPlot(residual_values, 'residual values', 'count', titleText='Residuals '+thisWeightLab, colorScheme=None, figSize=5, lowerBound=None, upperBound=None, filename='../Demand Estimation/hex_network_2c_results/log-price_vs_'+thisWeight+'_residual_histogram.png')

# print("_______________________________________")
# ###--------------------------------------------


# ###--------------------- WITH OLD HEX NETWORK ------------------
# # == Pearson Correlation Coefficient for linear-60 :    0.825
# # == Pearson Correlation Coefficient for sCurve-60_05 : 0.794
# # == Pearson Correlation Coefficient for sCurve-60_10 : 0.809
# # == Pearson Correlation Coefficient for sCurve-60_20 : 0.821

# # == Pearson Correlation Coefficient for linear-90 :    0.805
# # == Pearson Correlation Coefficient for sCurve-90_05 : 0.83
# # == Pearson Correlation Coefficient for sCurve-90_10 : 0.822
# # == Pearson Correlation Coefficient for sCurve-90_20 : 0.793

# # == Pearson Correlation Coefficient for linear-60 :    0.813
# # == Pearson Correlation Coefficient for sCurve-60_05 : 0.766
# # == Pearson Correlation Coefficient for sCurve-60_10 : 0.786
# # == Pearson Correlation Coefficient for sCurve-60_20 : 0.807

# # == Pearson Correlation Coefficient for linear-90 :    0.821
# # == Pearson Correlation Coefficient for sCurve-90_05 : 0.827
# # == Pearson Correlation Coefficient for sCurve-90_10 : 0.829
# # == Pearson Correlation Coefficient for sCurve-90_20 : 0.812


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



# ##================================== ADD TIME TO TOKYO AND CALCULATE ITS CORRELATIONS ======================

# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes.pkl')
# tokyoStation = getNodesByAttr(hexNetwork, 'stationNameEN', thisVal='Tokyo')[0]
# # print(tokyoStation)  ##--> '6193732167'

# timesToTokyo = nx.single_source_dijkstra_path_length(hexNetwork, '6193732167', weight='weight')
# nx.set_node_attributes(hexNetwork, timesToTokyo, "timeToTokyo")

# hexData = readPickleFile('../Demand Estimation/hexData_with_prices_v02c.pkl')
# hexData['timeToTokyo'] = hexData['id'].map(timesToTokyo)
# hexData = hexData[hexData['connected'] == True]
# print(hexData.head())

# priceData = hexData.copy()
# priceData = priceData[priceData['prices'].apply(lambda val: val != [])]
# ##-- there seems to be exactly one hex that has roads (connected) but not connected to the rest of the hex network
# priceData = priceData[priceData['timeToTokyo'].apply(lambda val: not isNan(val))]  
# priceData['price_mean_log'] = priceData['price_mean'].apply(lambda val: safeLog(val) )
# print("Number of hexes with prices:", len(priceData))  ##==> 8426  about  %
# print(priceData.head())

# # print([val for val in list(priceData['timeToTokyo']) if not isNumber(val)])
# # print(len([val for val in list(priceData['price_mean']) if not isNumber(val)]))
# # print(len([val for val in list(priceData['price_mean_log']) if not isNumber(val)]))

# thisCorrelation = getCorrelation(list(priceData['price_mean']), list(priceData['timeToTokyo']))
# print("== Pearson Correlation Coefficient for", 'price_mean',":", thisCorrelation)

# thisCorrelation = getCorrelation(list(priceData['price_mean_log']), list(priceData['timeToTokyo']))
# print("== Pearson Correlation Coefficient for", 'price_mean_log',":", thisCorrelation)


# # == Pearson Correlation Coefficient for price_mean and timeToTokyo: -0.714  --> -0.722
# # == Pearson Correlation Coefficient for price_mean_log and timeToTokyo: -0.74  --> -0.749
# ####--------------------------------------------





# ##================================== PROCESS OTHER STUFF FOR PRESENTATION & PAPER ======================

# thisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')
# thisAreaArea = convertGeomCRS(thisArea, standardToAreaProj)
# print(thisAreaArea.area)  ##==> 4893420593 m2 ==> 4893.420593 km2

# tokyoArea = readPickleFile('../Data/Polygons/tokyoAreaPolygon2.pkl')
# tokyoAreaArea = convertGeomCRS(tokyoArea, standardToAreaProj)
# print(tokyoAreaArea.area)  ##==> 13624284233 m2 ==> 13624.284233 km2

# print(thisAreaArea.area / tokyoAreaArea.area)  ##==> 0.359169 -> 36%


# # populationData = get_columns_for_geom(dbConnection, table='adminarea_data_old', geom=thisArea, columns=['totalPopulation','lowestLevel','geometry'])
# # populationData = populationData[populationData['lowestLevel']==True]
# # totalPopulation = populationData['totalPopulation'].sum()
# # print(totalPopulation)  ##--> 32,197,448


# employeeData = get_columns_for_geom(dbConnection, table='economics', geom=thisArea, columns=['num_employees','geometry'])
# totalEmployees = employeeData['num_employees'].sum()
# print(totalEmployees)  ##--> 15,080,305


# trainNetwork = readPickleFile('../Data/trainNetworks/trainNetwork_v8f.pkl')
# stationNodes = [node for node,attr in trainNetwork.nodes(data=True) if ((attr.get('modality')=='station'))]
# print(len(stationNodes))  ##--> 1441






    
# ###------------------------ CLEAN THE DATA A LITTLE-----------------------
# print("== Loading data and preparing grids")
# # hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes.pkl')
# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores.pkl')

# # print(sorted(getAllNodeAttributes(hexNetwork)))

# attrToConvertToInt = ['demand_linear-60', 'demand_linear-90', 'demand_sCurve-60_05', 'demand_sCurve-60_10', 'demand_sCurve-60_20', 'demand_sCurve-90_05', 'demand_sCurve-90_10', 'demand_sCurve-90_20']

# attrToRemove = ['demand', 'geomDist', 'geomMap',]

# for node,attr in hexNetwork.nodes(data=True):
#     reachableStations = attr.pop('nearby_stations',None)
#     nearbyStations = attr.pop('nearest_stations',None)
#     attr["reachable_stations"] = reachableStations
#     attr["nearby_stations"] = nearbyStations
    
#     for thisAtt in attrToRemove:
#         if thisAtt in list(attr.keys()):
#             del attr[thisAtt]
            
#     for thisAtt in attrToConvertToInt:
#         if thisAtt in list(attr.keys()):
#             attr[thisAtt] = makeInt(attr[thisAtt])
            
# # convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v02c+jobs+stationTimes+stores+other1")
# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores2.pkl')
            



# ###==============================================================================================
# ###===================================== ADD MORE DATA TO HEXES =================================
# ###==============================================================================================

    
    
    
# ###-------------------------------------------------------------------
# print("== Loading data and preparing grids")
# # hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes.pkl')
# hexNetwork = readPickleFile('../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores2.pkl')

# ###-----------------------
# nonHexEdges = [(u,v) for u,v,attr in hexNetwork.edges(data=True) if not attr["modality"] in ['hexStation', 'hex']]
# hexOnlyNetwork = hexNetwork.copy()
# hexOnlyNetwork.remove_edges_from(nonHexEdges)

# gridData = readGeoPandasCSV('../Data/GridData/meshData_lvl2.csv')  ##-- all level 2 meshes for the whole country
# # analysisArea = getPolygonForArea('tokyoMain')
# analysisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')  ##-- hopefully there won't be a problem reading this pkl
# gridData = gridData[gridData['geometry'].intersects(analysisArea)]
# print("== Number of grids in analysis area:", len(gridData))  ##==> 73 in tokyoMain
# # writeDataForKepler(gridData, "gridData_tokyoMain")  ##-- coverage confirmed
# # print(gridData.head())
# gridData['L2MeshCode'] = gridData['L2MeshCode'].map(str)  ##--> the gridNums should be strings
# # print(gridData.dtypes)
# # print(gridData.at[1924,'geometry'])  ##-- this is a closed polygon

# gridGeomLookup = dictFromColumns(gridData, 'L2MeshCode', 'geometry')
# gridNums = sorted([str(val) for val in gridData['L2MeshCode'].values])

# # gridNums = getUnmadeGrids(gridNums, gridGeomLookup)  ##-- check the database for tiles that have already been done

# # gridNums = ['533936']    ##-- area including eastern Minato-ku, Obaiba, and garbage island
# # gridNums = ['533945']    ##-- area including shinjuku, nakano, toshima kus and surrounding areas
# # gridNums = ['523954']    ##-- tip of Miura hanto
# # gridNums = ['523965']    ##-- the first tile with stations
# # gridNums = ['523972']    ##-- the second tile with stations, had nodes without geometry attr
# # print(gridNums.index("533962"))
# # print(gridNums[48])
# # gridNums = gridNums[48:]

# numFiles = len(gridNums)
# print("== Number of tiles to process:", numFiles)

# import warnings
# warnings.filterwarnings('ignore', '.*The array interface is deprecated.*', )
# warnings.filterwarnings('ignore', '.*parts of a multi-part geometry.*', )
# warnings.filterwarnings('ignore', '.*invalid value encountered in intersects*', )
# warnings.filterwarnings('ignore', '.*will attempt to set the values*', )
# warnings.filterwarnings('ignore', '.*will attempt to set the values*', )

# startTime = time.time()
# rowNum = 1

# ###------------------------------------------------------------

# for index,gridNum in enumerate(gridNums):
#     print("== Processing grid",rowNum,"of",numFiles,':',gridNum)
#     rowNum += 1
    
#     thisGridGeom = bufferGeometry(gridGeomLookup[gridNum], 125)
#     bufferedGeom = bufferGeometry(thisGridGeom, 1500)
        
#     hexNodeAttrs = {n:attr for n,attr in hexNetwork.nodes(data=True) if ((attr.get('modality','poo') == 'hex') & (attr.get('geometry','poo').intersects(bufferedGeom)))}
#     gridHexNodeAttrs = {n:attr for n,attr in hexNodeAttrs.items() if attr.get('geometry','poo').intersects(thisGridGeom)}
#     numberOfHexes = len(gridHexNodeAttrs.keys())
    
#     # ###--------------------------------------- STORE DATA -------------------------------------------------
#     # ###=== Process the store data     
#     # # storeData = get_data_for_geom(dbConnection, table="stores", geom=bufferedGeom)
#     # # print(storeData.head())
#     # # print('storeType', uniqueItems(list(storeData['storeType'])))
#     # # print('category', uniqueItems(list(storeData['category'])))
#     # # print('subcategory', uniqueItems(list(storeData['subcategory'])))    
    
#     # ###--- first get the number of stores IN each hex
#     # for thisHex,thisAttr in hexNodeAttrs.items():
#     #     hexNetwork.nodes[thisHex]["total_stores"] = len(storeData.loc[storeData.intersects(thisAttr['geometry'])])
#     #     relevantStores = storeData.loc[storeData.intersects(thisAttr['geometry'])]
#     #     relevantStores = relevantStores.loc[relevantStores['subcategory'].apply(lambda val: val in livableTypes)]
#     #     hexNetwork.nodes[thisHex]["total_relevant_stores"] = len(relevantStores)
#     #     hexNetwork.nodes[thisHex]["constantDiffusion"] = 100

#     # ###--- then get the store score based on times to surrpounding hexes' stores 
#     # for thisHex,thisAttr in gridHexNodeAttrs.items():
#     #     timesToNeighbors = nx.single_source_dijkstra_path_length(hexOnlyNetwork, thisHex, cutoff=15, weight='weight')
#     #     thisStoreScore = 0
#     #     neighborhoodHexes = 0
#     #     constantDiffusion = 0
#     #     for otherHex,timeToHex in timesToNeighbors.items():
#     #         thisStoreScore += weightedValue(timeToHex, value=hexNetwork.nodes[thisHex]["total_relevant_stores"], curvature=1, scalingLimit=15)
#     #         constantDiffusion += weightedValue(timeToHex, value=hexNetwork.nodes[thisHex]["constantDiffusion"], curvature=1, scalingLimit=15)
#     #         neighborhoodHexes += 1
#     #     hexNetwork.nodes[thisHex]["store_score"] = thisStoreScore
#     #     hexNetwork.nodes[thisHex]["constantDiffusion"] = constantDiffusion
#     #     hexNetwork.nodes[thisHex]["reachable_hexes_15min"] = neighborhoodHexes
#     #     ###---------------------------------------------------------
        
#     ###--------------------------------------- OTHER DATA -------------------------------------------------
#     landUseData = get_data_for_geom(dbConnection, table="land_use", geom=bufferedGeom)
#     zoningData = get_data_for_geom(dbConnection, table="zoning", geom=bufferedGeom)
#     zoningData['zoningCat'] = zoningData['zoneType'].apply(lambda val: zoningCatsDict[val])
#     zoningData['zoneTypeEn'] = zoningData['zoneType'].apply(lambda val: zoningCatsTrans[val])
#     vegData = get_data_for_geom(dbConnection, table="vegetation", geom=bufferedGeom)
#     vegData['vegCatEn'] = vegData['category'].apply(lambda val: vegCatTrans[val])
#     buildingData = get_data_for_geom(dbConnection, table="kiban_buildings", geom=bufferedGeom)
#     # print(thisHexLandUseData.head(20))
    
#     for thisHex,thisAttr in gridHexNodeAttrs.items():
#         bufferedHexGeom = bufferGeometry(thisAttr.get('geometry'), 1000)
#         bufferedHexGeomArea = convertGeomCRS(bufferedHexGeom, standardToAreaProj).area
        
#         ###---------------- LAND USE --------------------
#         thisHexLandUseData = landUseData.loc[landUseData.intersects(bufferedHexGeom)].copy()
#         # thisHexLandUseData['geometry'] = thisHexLandUseData['geometry'].apply(lambda geom: geom.intersection(bufferedHexGeom))
#         thisHexLandUseData['landUsePercent'] = thisHexLandUseData['geometry'].apply(lambda geom: convertGeomCRS(geom.intersection(bufferedHexGeom), standardToAreaProj).area / bufferedHexGeomArea)
#         thisHexLandUseData = thisHexLandUseData[['landUseEn','landUsePercent']]
#         thisHexLandUseData = thisHexLandUseData.groupby(by=["landUseEn"], as_index=False).sum()
#         landUseDict = dictFromColumns(thisHexLandUseData, 'landUseEn', 'landUsePercent')
#         hexNetwork.nodes[thisHex]["land_uses"] = landUseDict        
        
#         ###---------------- ZONING --------------------
#         thisHexZoningData = zoningData.loc[zoningData.intersects(bufferedHexGeom)].copy()
#         thisHexZoningData['zoningPercent'] = thisHexZoningData['geometry'].apply(lambda geom: convertGeomCRS(geom.intersection(bufferedHexGeom), standardToAreaProj).area / bufferedHexGeomArea)
#         thisHexZoningData1 = thisHexZoningData[['zoningCat','zoningPercent']]
#         thisHexZoningData1 = thisHexZoningData1.groupby(by=["zoningCat"], as_index=False).sum()
#         zoningDict = dictFromColumns(thisHexZoningData1, 'zoningCat', 'zoningPercent')
#         hexNetwork.nodes[thisHex]["zoneCats"] = zoningDict                
        
#         # thisHexZoningData = zoningData.loc[zoningData.intersects(bufferedHexGeom)].copy()
#         # thisHexZoningData['zoningPercent'] = thisHexZoningData['geometry'].apply(lambda geom: convertGeomCRS(geom.intersection(bufferedHexGeom), standardToAreaProj).area / bufferedHexGeomArea)
#         thisHexZoningData2 = thisHexZoningData[['zoneTypeEn','zoningPercent']]
#         thisHexZoningData2 = thisHexZoningData2.groupby(by=["zoneTypeEn"], as_index=False).sum()
#         zoningDict = dictFromColumns(thisHexZoningData2, 'zoneTypeEn', 'zoningPercent')
#         hexNetwork.nodes[thisHex]["zoneTypes"] = zoningDict              
        
#         ###---------------- VEGETATION --------------------
#         thisHexVegData = vegData.loc[vegData.intersects(bufferedHexGeom)].copy()
#         thisHexVegData['vegPercent'] = thisHexVegData['geometry'].apply(lambda geom: convertGeomCRS(geom.intersection(bufferedHexGeom), standardToAreaProj).area / bufferedHexGeomArea)
#         thisHexVegData = thisHexVegData[['vegCatEn','vegPercent']]
#         thisHexVegData = thisHexVegData.groupby(by=["vegCatEn"], as_index=False).sum()
#         vegDict = dictFromColumns(thisHexVegData, 'vegCatEn', 'vegPercent')
#         hexNetwork.nodes[thisHex]["vegCats"] = vegDict                
        
#         ###---------------- BUILDINGS --------------------
#         thisBuildingData = buildingData.loc[buildingData.intersects(bufferedHexGeom)].copy()
#         thisBuildingData['surfaceArea'] = thisBuildingData['geometry'].apply(lambda geom: convertGeomCRS(geom.intersection(bufferedHexGeom), standardToAreaProj).area)
#         hexNetwork.nodes[thisHex]["numBuildings"] = len(thisBuildingData)
#         hexNetwork.nodes[thisHex]["meanBuildingSurfaceArea"] = np.mean(np.array(thisBuildingData['surfaceArea']))
#         hexNetwork.nodes[thisHex]["percentBuildingSurfaceArea"] = np.sum(np.array(thisBuildingData['surfaceArea'])) / bufferedHexGeomArea
        
#         ###---------------- POPULATION --------------------
#         ##-- is not clearly useful for the hedonic pricing and requires complicated interpolation, so skip for now
        
        
        
#     startTime = reportProgressTime(startTime,"Completed grid with "+str(numberOfHexes)+" hexes in")
        
    
# ###--- for testing one tile
# # hexNodes = [n for n,attr in gridHexNodeAttrs.items()]
# # convertNetworkToKeplerFiles(getSubgraph(hexNetwork, hexNodes), filename="hex_network_v02c+jobs+stationTimes+stores_other1-test")
        
# ###--- the first pass doing just stores
# # writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores.pkl')
# # convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v02c+jobs+stationTimes+stores")

# ###--- after stores, processing other neighborhood data
# writePickleFile(hexNetwork,'../Demand Estimation/hexNetwork_v02c+jobs+stationTimes+stores+other.pkl')
# convertNetworkToKeplerFiles(hexNetwork, filename="hex_network_v02c+jobs+stationTimes+stores+other")



# # print(weightedValue(10, value=8, curvature=1, scalingLimit=15))
# # thisHex = '-186_-222'
# # timesToNeighbors = nx.single_source_dijkstra_path_length(hexOnlyNetwork, thisHex, cutoff=15, weight='weight')
# # print(timesToNeighbors)












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
# ##---------------------------------------------------

# print("-- Adding geometry column to data")
# roomData['geometry'] = [Point(xy) for xy in zip(roomData.lon, roomData.lat)]
# roomData = gp.GeoDataFrame(roomData, geometry='geometry')
# roomData.crs = standardCRS

# print("-- Filtering to Tokyo Main")
# analysisArea = readPickleFile('../Data/Polygons/tokyoMainPolygon2.pkl')
# roomData = roomData[roomData['geometry'].intersects(analysisArea)]

# print("-- Refining Data Columns")
# roomData['built_year'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[0:4]))
# roomData['built_month'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[4:6]))
# roomData.drop(columns=['built_yyyymm', 'lat', 'lon'], inplace=True)


# roomData['age_in_months'] = roomData.apply(lambda row: makeInt((12 - row['built_month']) + (12 * (2023 - row['built_year']))), axis=1)
# roomData['age_in_years'] = roomData['built_year'].apply(lambda val: 2023 - row['built_year'] )

# def getDepositAmount(rowData):
#     if isNumber(rowData['deposit_amount']):
#         return int(rowData['deposit_amount'])
#     elif isNumber(rowData['deposit_months']):
#         return int(rowData['deposit_months'] * rent)
#     else:
#         return 0

# def getKeyMoneyAmount(rowData):
#     if isNumber(rowData['key_money_amount']):
#         return int(rowData['key_money_amount'])
#     elif isNumber(rowData['key_money_months']):
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




# ##===========================================================================================
# ##================================= CREATE HEX+TRAIN NETWORK ================================
# ##===========================================================================================

# hexNodes = get_entire_data_table(dbConnection, table="hex+train_network_v2c_nodes")
# hexEdges = get_entire_data_table(dbConnection, table="hex+train_network_v2c_edges")
# hexNetwork = networkFromDataframes(hexNodes, hexEdges)

# writePickleFile(hexNetwork,'../Data/GridData/hexNetwork_v2c.pkl')
# ##-------------------------------------------------------------











# ##======================================== END OF FILE ===========================================