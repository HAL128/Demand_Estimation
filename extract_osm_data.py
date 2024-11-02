# -*- coding: utf-8 -*-
from helpers.helper_functions import *
pd.set_option('display.max_columns', None)




####=================== GENERATE THE GRID =====================

###=== Convert the point for Tokyo station to equalDistCRS
#TokyoStationLon = 139.7671
#TokyoStationLat = 35.6812
#tokyoPoint_DistCRS = convertPoint(TokyoStationLon, TokyoStationLat, standardCRS, distCalcCRS)
#print("Tokyo Station coords in distCRS:", tokyoPoint_DistCRS)   ##-- (0,0) because the CRS also uses this as a reference point

###=== Convert the bounding coords for Japan into the same CRS
#regionWestBound = 129.400485
#regionEastBound = 146.175857
#regionNorthBound = 45.611193
#regionSouthBound = 30.875052
#SWcorner = convertPoint(regionWestBound, regionSouthBound, standardCRS, distCalcCRS)
#NEcorner = convertPoint(regionEastBound, regionNorthBound, standardCRS, distCalcCRS)
#print(SWcorner)
#print(NEcorner)
#westBound,southBound,eastBound,northBound = [-1154007, -535018, 713420, 1105402]
#boundingPolygon = Polygon([(westBound, southBound), (westBound, northBound), (eastBound, northBound), (eastBound, southBound)])

#westBound,southBound,eastBound,northBound = [139.663103,35.645812,139.766271,35.714904]  ##-- test area at the center of 23Wards
#boundingPolygon = Polygon([(westBound, southBound), (westBound, northBound), (eastBound, northBound), (eastBound, southBound)])
#boundingPolygon = convertGeometry(boundingPolygon, standardCRS, distCalcCRS)
##print(boundingPolygon)
#westBound,southBound,eastBound,northBound = boundingPolygon.bounds
#print(list(thisGeom.exterior.coords)[0][0])  ### it's the same order as the definition

###------------------
#JapanShapes = gp.read_file("../Data/GISandAddressData/JapanShapeGeopackage.gpkg")
#JapanShapes = JapanShapes.dissolve(by='NAME_0')['geometry']
#geoJapanShapes = gp.GeoDataFrame(JapanShapes)
#geoJapanShapes.crs = standardCRS
#geoJapanShapes = geoJapanShapes.to_crs(distCalcCRS)
#japanMultiPolygon = geoJapanShapes['geometry'][0]
#writePickleFile(japanMultiPolygon, "../Data/GISandAddressData/japanMultiPolygon.pkl")

#boundingPolygon = readPickleFile("../Data/GISandAddressData/japanMultiPolygon.pkl")
#boundingPolygon = convertGeometry(readPickleFile('../Data/tokyoAreaPolygon.pkl'), standardCRS, distCalcCRS)
#
####=== Create a geoDataFrame with columns for netGridNum, x-index, y-index, combo-index (str), xMin, yMin, xMax, yMax, and geometry
####=== ...and initialize it with a box that is 1.5km wide and tall, with Tokyo Station at the center.
#from shapely.geometry import Polygon


###============================== Can start from here
#
#boundingPolygon = convertGeometry(readPickleFile('../Data/tokyoAreaPolygon.pkl'), standardCRS, distCalcCRS)
#westBound,southBound,eastBound,northBound = boundingPolygon.bounds
#
#def resetToTokyo():
#    xIndex = 0
#    yIndex = 0
#    comboIndex = str(xIndex)+'_'+str(yIndex)
#    xMin = -750
#    yMin = -750
#    xMax = 750
#    yMax = 750
#    thisGeom = Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)])
#    return (xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#
#def goEast(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom):
#    xIndex = xIndex + 1
#    yIndex = yIndex
#    comboIndex = str(xIndex)+'_'+str(yIndex)
#    xMin = xMin + 1500
#    yMin = yMin
#    xMax = xMax + 1500
#    yMax = yMax
#    thisGeom = Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)])
#    return (xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#
#def goWest(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom):
#    xIndex = xIndex - 1
#    yIndex = yIndex
#    comboIndex = str(xIndex)+'_'+str(yIndex)
#    xMin = xMin - 1500
#    yMin = yMin
#    xMax = xMax - 1500
#    yMax = yMax
#    thisGeom = Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)])
#    return (xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#
#def goNorth(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom):
#    xIndex = xIndex
#    yIndex = yIndex + 1
#    comboIndex = str(xIndex)+'_'+str(yIndex)
#    xMin = xMin
#    yMin = yMin + 1500
#    xMax = xMax
#    yMax = yMax + 1500
#    thisGeom = Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)])
#    return (xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#
#def goSouth(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom):
#    xIndex = xIndex
#    yIndex = yIndex - 1
#    comboIndex = str(xIndex)+'_'+str(yIndex)
#    xMin = xMin
#    yMin = yMin - 1500
#    xMax = xMax
#    yMax = yMax - 1500
#    thisGeom = Polygon([(xMin, yMin), (xMin, yMax), (xMax, yMax), (xMax, yMin)])
#    return (xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#
#
#
####--- initialize with the grid for Tokyo station
#xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = resetToTokyo()
#
#japanGrid = [{'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom }]
#
####--- first do the row with Tokyo, going east
#while (xMax < eastBound):
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goEast(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#    if thisGeom.intersects(boundingPolygon):
#        japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
####--- reinitialize with the grid for Tokyo station
#xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = resetToTokyo()
#
####--- now do the row with Tokyo, going west
#while (xMin > westBound):
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goWest(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#    if thisGeom.intersects(boundingPolygon):
#        japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#print("==== Finished Tokyo Row, Going South ====")
#
####--- reinitialize with the grid for Tokyo station
#REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom = resetToTokyo()
#
####-- now go north one, and create the center grid
#while (yMin < northBound):
#    ##-- set the ref data, used for going east and west
#    REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom = goNorth(REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom)
#
#    ##-- set the data equal to this row's ref data, and add this center point to the data
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom
#    if thisGeom.intersects(boundingPolygon):
#        japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#    ###--- then go east from this new center point
#    while (xMin < eastBound):
#        xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goEast(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#        if thisGeom.intersects(boundingPolygon):
#            japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#    ##-- reset the data equal to this row's ref data
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom
#    ###--- then go west from this new center point
#    while (xMax > westBound):
#        xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goWest(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#        if thisGeom.intersects(boundingPolygon):
#            japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#
#print("==== Finished Going North, Now Going South ====")
#
####--- reinitialize with the grid for Tokyo station
#REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom = resetToTokyo()
#
####-- go south one, and create the center grid
#while (yMax > southBound):
#
#    ##-- set the ref data, used for going east and west
#    REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom = goSouth(REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom)
#
#    ##-- set the data equal to this row's ref data, and add this center point to the data
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom
#    if thisGeom.intersects(boundingPolygon):
#        japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#    ###--- then go east from this new center point
#    while (xMin <= eastBound):
#        xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goEast(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#        if thisGeom.intersects(boundingPolygon):
#            japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#    ##-- reset the data equal to this row's ref data
#    xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = REFxIndex, REFyIndex, REFcomboIndex, REFxMin, REFyMin, REFxMax, REFyMax, REFthisGeom
#    ###--- then go west from this new center point
#    while (xMax >= westBound):
#        xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom = goWest(xIndex, yIndex, comboIndex, xMin, yMin, xMax, yMax, thisGeom)
#        if thisGeom.intersects(boundingPolygon):
#            japanGrid.append({'xIndex': xIndex, 'yIndex': yIndex, 'comboIndex':comboIndex, 'xMin':xMin, 'yMin':yMin, 'xMax':xMax, 'yMax':yMax, 'geometry':thisGeom })
#
#
#
#print("==== Writing Lookup File ====")
#japanGrid = gp.GeoDataFrame(pd.DataFrame(japanGrid), crs=distCalcCRS, geometry='geometry')
#writePickleFile(japanGrid,'../Data/OSMData/networkGridLookup-TokyoArea-distCRS.pkl')
#writeGeoCSV(japanGrid, '../Data/OSMData/networkGridLookup-TokyoArea-distCRS.csv')
#
#japanGrid = japanGrid.to_crs(standardCRS)
####=== Also convert the xMin, xMax, yMin, and yMax values to lat/lon for speedier lookup.
####--- Instead of making them geometries and converting them, just pull them from the converted geometry column
#japanGrid['xMin'] = japanGrid.apply(lambda row: list(row['geometry'].exterior.coords)[0][0], axis=1)
#japanGrid['xMax'] = japanGrid.apply(lambda row: list(row['geometry'].exterior.coords)[2][0], axis=1)
#japanGrid['yMin'] = japanGrid.apply(lambda row: list(row['geometry'].exterior.coords)[0][1], axis=1)
#japanGrid['yMax'] = japanGrid.apply(lambda row: list(row['geometry'].exterior.coords)[2][1], axis=1)
#
#writePickleFile(japanGrid,'../Data/OSMData/networkGridLookup-TokyoArea-standardCRS.pkl')
#writeGeoCSV(japanGrid, '../Data/OSMData/networkGridLookup-TokyoArea-standardCRS.csv')
#
#print(japanGrid.head(10))
#print("Number of grids created:", len(japanGrid))

#print(list(japanGrid.loc[1]['geometry'].exterior.coords)[0][0])

#print(japanGrid[japanGrid['comboIndex']=="0|1"])
#print(japanGrid.loc[1]['geometry'])




###======================== CREATE 23WARDS POLYGON =======================
#print("==== Getting Admin Area Polygon Data ====")
#chomeData = getDataForVariables(['geometry', 'in23Wards', 'adminLevel'], dataType="chomeData")
#chomeData = chomeData[chomeData.adminLevel == 1][['geometry', 'in23Wards', 'adminLevel']]
#chomeData = chomeData[chomeData['adminLevel'] == 1]
#chomeData = chomeData[chomeData['in23Wards'] == True]
#chomeData = chomeData.dissolve(by='adminLevel')
#print(chomeData.head(10))
###=== Now get the dissolved polygon and save it as a pickle
#wardsPolygon = chomeData['geometry'].values[0]
#print(type(wardsPolygon))  ##-- this confirms that it is just the Shapely multipolygon geometry object.
#writePickleFile(wardsPolygon,'../Data/wardsPolygon.pkl')





####======================== ISOLATE GRIDS WITHIN 23WARDS =======================
####=== Rduce the gridData to the areas intersecting the 23Wards
#wardsPolygon = readPickleFile('../Data/wardsPolygon.pkl')
#gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoArea-standardCRS.pkl')
#print("Number of grids in Tokyo Area:", len(gridData))   ## == 7699
#gridData2 = gridData[gridData.apply(lambda row: row['geometry'].intersects(wardsPolygon), axis=1)]
#print("Number of grids in 23 Wards:", len(gridData2))    ## == 409
#print(gridData2.head())
#writePickleFile(gridData2,'../Data/OSMData/networkGridLookup-23Wards-standardCRS.pkl')
#writeGeoCSV(gridData2, '../Data/OSMData/networkGridLookup-23Wards-standardCRS.csv')


#smallGridData = readGeoPickle('../Data/OSMData/networkGridLookup-23Wards-standardCRS.pkl')
#numGrids = len(smallGridData)
#print("Number of grids in lookup table", numGrids)
#print("xmin:", min(smallGridData['xIndex'].values))
#print("ymin:", min(smallGridData['yIndex'].values))
#print("xmax:", max(smallGridData['xIndex'].values))
#print("ymax:", max(smallGridData['yIndex'].values))
#print(smallGridData.head())


####======================== CREATE LARGE GRID =======================
#smallGridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoArea-standardCRS.pkl')
#print("xmin:", min(smallGridData['xIndex'].values))
#print("ymin:", min(smallGridData['yIndex'].values))
#print("xmax:", max(smallGridData['xIndex'].values))
#print("ymax:", max(smallGridData['yIndex'].values))

#xLarge = 0
#yLarge = 0


#####======================== CLEAN STORE DATA =======================
#storeData = readCSV("../Data/townPageData_with_storeCategory.csv", fillNaN='', theEncoding='utf-8')
##print("Number of stores:",len(storeData)) ##==> 93145
##print(list(storeData))
#storeData['nodeID'] = storeData.apply(lambda row: "store_"+str(row['index']), axis=1)
#storeData = storeData[['nodeID', 'lat', 'lon', 'bestbashoCategory', 'company_name', 'category', 'sub_category', 'category_code', 'sub_category_id', 'prefecture', 'city', 'town', 'chome', 'banchi', 'floor', 'company_name_kana', 'corporation', 'corporation_id', 'addressMatched']]
#storeData['bestbashoCategory'] = storeData.apply(lambda row: "eatery" if row['bestbashoCategory'] == "restrant" else row['bestbashoCategory'], axis=1)
#storeData.rename(columns={'company_name':'companyName', 'sub_category':'subCategory', 'category_code':'categoryCode', 'sub_category_id':'subCategoryCode', 'company_name_kana':'companyNameKana', 'corporation_id':'corporationCode', }, inplace=True)
#storeData['modality'] = 'store'
##print(list(set(storeData['bestbashoCategory'])))
##print(storeData[storeData['bestbashoCategory'] == ''])  ##== 19 rows are missing a categroy,
##print(storeData[storeData['subCategory'] == '遊園地・テーマパーク'])  ##== and there is no category for the same subcatID, so add a category for them
#storeData['bestbashoCategory'] = storeData.apply(lambda row: "amusement" if row['bestbashoCategory']=='' else row['bestbashoCategory'], axis=1)
#storeData = gp.GeoDataFrame(storeData)
#storeData['geometry'] = storeData.apply(lambda row: Point(row['lon'],row['lat']), axis=1)  ### Kepler can use lat/lon, but needed for web apps
####=== Group stores together with the same geometry and convert all data to lists of values to reduce the number of nodes created.
#storeBuildings = storeData.groupby(by=['lat','lon']).agg(lambda x: list(x)).reset_index()
##print("Number of unique locations:",len(storeBuildings))  ##==> 60853   ###== saves 30%
##print(storeBuildings.head(5))
#storeBuildings['nodeID'] = storeBuildings.apply(lambda row: row['nodeID'][0], axis=1)
#storeBuildings['geometry'] = storeBuildings.apply(lambda row: row['geometry'][0], axis=1)
#storeBuildings['modality'] = storeBuildings.apply(lambda row: row['modality'][0], axis=1)
#storeBuildings['prefecture'] = storeBuildings.apply(lambda row: row['prefecture'][0], axis=1)
#storeBuildings['city'] = storeBuildings.apply(lambda row: row['city'][0], axis=1)
#storeBuildings['town'] = storeBuildings.apply(lambda row: row['town'][0], axis=1)
#storeBuildings['chome'] = storeBuildings.apply(lambda row: row['chome'][0], axis=1)
#storeBuildings['banchi'] = storeBuildings.apply(lambda row: row['banchi'][0], axis=1)
#storeBuildings['numStores'] = storeBuildings.apply(lambda row: len(row['bestbashoCategory']), axis=1)
###=== create three other geometry columns with buffers increasingly large buffers ==> converted roughly into degrees
#storeBuildings['smBufferGeom'] = storeBuildings.apply(lambda row: row['geometry'].buffer(10 / 111030), axis=1)
#storeBuildings['mdBufferGeom'] = storeBuildings.apply(lambda row: row['geometry'].buffer(20 / 111030), axis=1)
#storeBuildings['lgBufferGeom'] = storeBuildings.apply(lambda row: row['geometry'].buffer(100 / 111030), axis=1)
#print(storeBuildings.head())
#
##print(len(storeData[storeData.apply(lambda row: not isinstance(row['lat'],float), axis=1)]))  ##--all lon,lat values are floats
#writeGeoCSV(storeData, '../Data/storeData-23Wards.csv')
#writePickleFile(storeData, '../Data/storeData-23Wards.pkl')
#writeGeoCSV(storeBuildings, '../Data/storeBuildings-23Wards.csv')
#writePickleFile(storeBuildings, '../Data/storeBuildings-23Wards.pkl')



# storeIntregrationTimes = []
#print(storeIntregrationTimes)
#writePickleFile(storeIntregrationTimes,'../Data/WalkabilityAnalysis/storeIntregrationTimes.pkl')


####=====================================================================
####======================== CREATE NETWORK TILES =======================
####=====================================================================

###=== Method to handle the details of adding a bidirectional walking edge between two nodes.
def createWalkingEdge(thisNetwork, node1, node2, standardToDistProj, modality="road"):
    x1 = thisNetwork.nodes[node1]['lon']
    y1 = thisNetwork.nodes[node1]['lat']
    x2 = thisNetwork.nodes[node2]['lon']
    y2 = thisNetwork.nodes[node2]['lat']
    length = thisNetwork.nodes[node1]['geomDist'].distance(thisNetwork.nodes[node2]['geomDist']) # Euclidean distance
    walkTime = metersToMinutes(length)
    thisGeom = Point(x1,y1) if ((x1==x2) & (y1==y2)) else LineString([(x1,y1), (x2,y2)])
    thisGeomDist = convertGeomCRS(thisGeom, standardToDistProj)

    thisNetwork.add_edge(node1, node2, modality=modality, walking=True, driving=False, x1=x1, y1=y1, x2=x2, y2=y2, geometry=thisGeom, geomDist=thisGeomDist, length=length, timeWeight=walkTime, speed=walkingSpeed)

    if nx.is_directed(thisNetwork):
        ###--- add the edge in the other direction if the network is directed
        thisGeomFlipped = Point(x2,y2) if ((x1==x2) & (y1==y2)) else LineString([(x2,y2), (x1,y1)])
        thisGeomDistFlipped = convertGeomCRS(thisGeomFlipped, standardToDistProj)
        thisNetwork.add_edge(node2, node1, modality=modality, walking=True, driving=False, x1=x2, y1=y2, x2=x1, y2=y1, geometry=thisGeomFlipped, geomDist=thisGeomDistFlipped, length=length, timeWeight=walkTime, speed=walkingSpeed)



###--------------------------------------------
def getClosestPointEuclidean(originGeomDist, nodeDF):
    dists = np.array([originGeomDist.distance(point) for point in nodeDF['geomDist']])
    return dists.argmin(), dists.min()


###------------------------------------------------------------
def findAndConnectLooseExits(thisNetwork, standardToDistProj):
#    beginTime = time.time()
    ###=== Get the components of the network sorted by decreasing size as a dictionary keyed by index#
    if nx.is_directed(thisNetwork):
        components = {index:{'nodes':list(comp)} for index,comp in enumerate(list(sorted(nx.weakly_connected_components(thisNetwork), key=len, reverse=True)))}
    else:
        components = {index:{'nodes':list(comp)} for index,comp in enumerate(list(sorted(nx.connected_components(thisNetwork), key=len, reverse=True)))}
    #print("Number of components before connecting:", len(components))
    #print("Component sizes:", [len(attr.get('nodes',0)) for index,attr in components.items()])

    ###=== Create a dataframe with the lon/lats of the road nodes of the main component for fast minDist calculations
    mainComponentNodeLocations = pd.DataFrame.from_dict(dict(thisNetwork.subgraph(components[0]['nodes']).nodes(data=True)), orient='index')
    mainComponentNodeLocations = mainComponentNodeLocations[mainComponentNodeLocations['modality'] == 'road']

    ###=== Identify the non-main conponent indices that contain at least one exit node.
    componentsWithExits = [index for index,attr in components.items() if len([node for node in attr['nodes'] if thisNetwork.nodes[node]['modality'] == "exit"]) > 0]
    #print("Number of disconnected components with exits:", len(componentsWithExits[1:]))

    ###=== connect the node in this comp with the closest large component road node via the shortest edge.
    for thisCompNum in componentsWithExits[1:]:
        ###-- Get the list of road node refs in this component
        thisComp = [node for node in components[thisCompNum]['nodes'] if thisNetwork.nodes[node]['modality'] == "road"]
        minDist = 100000  ##-- initialize to very large value
        edgeToMake = []
        for thisNode in thisComp:
            # closestNodeIndex, closestNodeDist = getClosestPointHaversine(thisNetwork.nodes[thisNode]['lat'], thisNetwork.nodes[thisNode]['lon'], mainComponentNodeLocations)
            closestNodeIndex, closestNodeDist = getClosestPointEuclidean(thisNetwork.nodes[thisNode]['geomDist'], mainComponentNodeLocations)
            closestNodeIndex = mainComponentNodeLocations.iloc[[closestNodeIndex]].index[0]
            if closestNodeDist < minDist:
                minDist = closestNodeDist
                edgeToMake = [thisNode, closestNodeIndex]

        node1, node2 = edgeToMake
        ## now we have the info for the shortest edge needed to connect this exit component to the main component, so add the edge.
        createWalkingEdge(thisNetwork, node1, node2, standardToDistProj, modality="road")
#    reportRunTime(beginTime, "Finding and connecting disconnected exits")



###--------------------------------------------
###=== Split an edge at a point by breaking it into two edges connected at the specified point with attributes updated appropriately
def splitEdge(thisNetwork, startNode, endNode, middleNode, standardToDistProj, roadEdgesGDF, roadNodesGDF):

    origEdgeAttr = thisNetwork[startNode][endNode]
#    print("this road edge attributes",origEdgeAttr)
    x1 = thisNetwork.nodes[startNode]['lon']
    y1 = thisNetwork.nodes[startNode]['lat']
    x2 = thisNetwork.nodes[endNode]['lon']
    y2 = thisNetwork.nodes[endNode]['lat']
    x3 = thisNetwork.nodes[middleNode]['lon']
    y3 = thisNetwork.nodes[middleNode]['lat']

    length1 = thisNetwork.nodes[startNode]['geomDist'].distance(thisNetwork.nodes[middleNode]['geomDist'])  # Euclidean distance
    walkTime1 = metersToMinutes(length1)
    firstGeom = Point(x1,y1) if ((x1==x3) & (y1==y3)) else LineString([(x1,y1), (x3,y3)])
    firstGeomDist = convertGeomCRS(firstGeom, standardToDistProj)

    length2 = thisNetwork.nodes[middleNode]['geomDist'].distance(thisNetwork.nodes[endNode]['geomDist'])  # Euclidean distance
    walkTime2 = metersToMinutes(length2)
    lastGeom = Point(x2,y2) if ((x2==x3) & (y2==y3)) else LineString([(x3,y3), (x2,y2)])
    lastGeomDist = convertGeomCRS(lastGeom, standardToDistProj)

    thisNetwork.add_edge(startNode, middleNode, x1=x1, y1=y1, x2=x3, y2=y3, geometry=firstGeom, geomDist=firstGeomDist, length=length1, timeWeight=walkTime1)
    thisNetwork.add_edge(middleNode, endNode, x1=x3, y1=y3, x2=x2, y2=y2, geometry=lastGeom, geomDist=lastGeomDist, length=length2, timeWeight=walkTime2)
    ###=== Now transfer the original attributes to both split edges except the ones just set.
    redoneAttr = ['geometry', 'geomDist', 'length','timeWeight','x1','y1','x2','y2']
    oldAttrToKeep = list(set(origEdgeAttr.keys()) - set(redoneAttr))   ### This is much faster for finding list differences than list comprehension
#    attrToKeep = [k for k,v in origEdgeAttr.items() if k not in attrToErase ]
    for thisAttr in oldAttrToKeep:
        thisNetwork[startNode][middleNode][thisAttr] = origEdgeAttr[thisAttr]
        thisNetwork[middleNode][endNode][thisAttr] = origEdgeAttr[thisAttr]

    thisNetwork.remove_edge(startNode, endNode) ##== remove the old edge
    ###=== removing the edge here causes an error because it is still available in the geoDFs for the nodes and edges used for proximity lookup
    ###=== so, those have to be updated here.
    roadEdgesGDF = roadEdgesGDF[((roadEdgesGDF['source'] != startNode) & (roadEdgesGDF['target'] != endNode)) & ((roadEdgesGDF['source'] != endNode) & (roadEdgesGDF['target'] != startNode))]
#    roadEdgesGDF = roadEdgesGDF[((roadEdgesGDF['source'] != endNode) & (roadEdgesGDF['target'] != startNode))]  ##-- do both because we don't know which it is
    newEdgeAsDict = thisNetwork[startNode][middleNode]
    newEdgeAsDict['source'] = startNode
    newEdgeAsDict['target'] = middleNode
    roadEdgesGDF = roadEdgesGDF.append(newEdgeAsDict, ignore_index=True)
    newEdgeAsDict = thisNetwork[middleNode][endNode]
    newEdgeAsDict['source'] = middleNode
    newEdgeAsDict['target'] = endNode
    roadEdgesGDF = roadEdgesGDF.append(newEdgeAsDict, ignore_index=True)
#    print(roadEdgesGDF.tail(5))
#    print(type(roadEdgesGDF))  ## confirmed it's still geopandas
    roadEdgesGDF = gp.GeoDataFrame(roadEdgesGDF, geometry='geomDist')  ##== This is needed to reset the geomDist column as the geomtry after append

    roadNodesGDF.loc[middleNode] = ['road', x3, y3, thisNetwork.nodes[middleNode]['geometry'], thisNetwork.nodes[middleNode]['geomDist']]
#    print(type(roadNodesGDF.tail(5))

    return (roadEdgesGDF, roadNodesGDF)  ## return the updated reference DFs of node and edge locations

###------------------------------------------------------------
###=== Method to connect a node to the closest point on the closest road which may be an endpoint)
def findClosestPointOnRoadAndConnect(thisNetwork, nodeID, nodeGeomDist, roadEdgesGDF, roadNodesGDF, standardToDistProj, distToStandardProj, thisModality):
#    potentialNearestEdges = roadEdgesGDF.loc[roadEdgesGDF.intersects(nodeGeomDist)]  ##== geomDist was specified as the geometry column when building these GDFs
    potentialNearestEdges = roadEdgesGDF[roadEdgesGDF['geomDist'].intersects(nodeGeomDist)]

    if len(potentialNearestEdges) == 0:
        potentialNearestEdges = roadEdgesGDF[roadEdgesGDF['geomDist'].intersects(nodeGeomDist.buffer(10))]

    if len(potentialNearestEdges) == 0:
        potentialNearestEdges = roadEdgesGDF[roadEdgesGDF['geomDist'].intersects(nodeGeomDist.buffer(20))]

    if len(potentialNearestEdges) == 0:
        potentialNearestEdges = roadEdgesGDF[roadEdgesGDF['geomDist'].intersects(nodeGeomDist.buffer(100))]

    if len(potentialNearestEdges) > 0:
        newPointDist, theNode_endpoints = findNearestEdgeEuclidean(potentialNearestEdges, nodeGeomDist)
        theNodeCoords, startNodeCoords, endNodeCoords, startNodeID, endNodeID = theNode_endpoints

        if theNodeCoords == startNodeCoords:  ## the closest point on the line is the sourceNode
            createWalkingEdge(thisNetwork, nodeID, startNodeID, standardToDistProj, modality=thisModality)
        elif theNodeCoords == endNodeCoords:  ## the closest point on the line is the targetNode
            createWalkingEdge(thisNetwork, nodeID, endNodeID, standardToDistProj, modality=thisModality)
        else:  ## the closest point is along the the edge somewhere, so make a new node.
            newNodeID = str(startNodeID)+"v"+str(endNodeID)
            theNodeLon, theNodeLat = distToStandardProj.transform(theNodeCoords[0], theNodeCoords[1])
            theNodeGeometry = Point(theNodeLon, theNodeLat)
            thisNetwork.add_node(newNodeID, lon=theNodeLon, lat=theNodeLat, modality="road", geometry=theNodeGeometry, geomDist=convertGeomCRS(theNodeGeometry, standardToDistProj))
            createWalkingEdge(thisNetwork, nodeID, newNodeID, standardToDistProj, modality=thisModality)
            ###=== now split the road edge from startNodeID to endNodeID into two road nodes connected at the new node.
            roadEdgesGDF,roadNodesGDF = splitEdge(thisNetwork, startNodeID, endNodeID, newNodeID, standardToDistProj, roadEdgesGDF, roadNodesGDF)
    else:  ###--- if still none are found using the large buffer, then just use the closest road node
        closestNodeNum, closestNodeDist = getClosestPointEuclidean(nodeGeomDist, roadNodesGDF)
        closestNodeIndex = roadNodesGDF.iloc[[closestNodeNum]].index[0]
        createWalkingEdge(thisNetwork, nodeID, closestNodeIndex, standardToDistProj, modality=thisModality)

    return (roadEdgesGDF, roadNodesGDF)  ##== these have to be updated based on the new nodes and edges added/removed

####=====================================================================
####==== HELPER FUNCTIOINS FOR CLOSEST POINT ON LINE USING GEOM DIST ====
####=====================================================================
###=== from https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def nearestPointOnLine(startNodePoint, endNodePoint, otherPoint):
    # Returns the nearest point on a given line and its distance
    x1, y1 = startNodePoint
    x2, y2 = endNodePoint
    x3, y3 = otherPoint

    if startNodePoint == endNodePoint:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)

    ### this may not work here because coords are in degrees, so the delta values for x and y are not commensurable
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = ( (dy * (y3 - y1)) + (dx * (x3 - x1)) ) / det

    # Corner cases
    if a >= 1:
        return x2, y2, euclideanDistance(x2, y2, x3, y3)
    elif a <= 0:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)

    newpx = x1 + a * dx
    newpy = y1 + a * dy
    return newpx, newpy, euclideanDistance(newpx, newpy, x3, y3)


def findNearestEdgeEuclidean(potentialNearestEdges, otherPoint):
    otherPoint = list(otherPoint.centroid.coords)[0]
    minDist = float('inf')
    theNode_endpoints = None
    for startNodeID, endNodeID, geomDist in potentialNearestEdges[['source', 'target', 'geomDist']].values:
        startNodeCoords, endNodeCoords = list(geomDist.coords)
        x3, y3, dist = nearestPointOnLine(startNodeCoords, endNodeCoords, otherPoint)
        if dist < minDist:
            minDist = dist
            theNode_endpoints = (dist, [(x3, y3), startNodeCoords, endNodeCoords, startNodeID, endNodeID])

    return theNode_endpoints


###========================= BEGIN TILE CREATION ==============================
def createTile(useDatabase, useMultiprocessing, gridNum, totalGridNum, thisGrid, theseStores, rowNum, beginTime, trainNetwork, hexData, theStoreConnectionTimes):
    startTime = time.time()
#        time.sleep(30)  ###=== in order to avoid getting a "too many requests" error, try waiting

    varsToKeep = ['x1', 'y1', 'x2', 'y2', 'geometry', 'highway', 'maxspeed', 'oneway', 'lanes', 'width', 'area', 'bridge', 'tunnel', 'crossing']
    #stationVarsToKeep = ['name', 'operator']
    walkingBadRoadTypes = ["cycleway", "motor", "proposed", "construction", "abandoned", "platform", "raceway", "motorway", "motorway_link"]
    drivingBadRoadTypes = ['cycleway', 'footway', 'path', 'pedestrian', 'steps', 'track', 'corridor', 'elevator', 'escalator', 'proposed', 'construction', 'bridleway', 'abandoned', 'platform', 'raceway']

    ###=== Add approx speed limits and widths for road segments that don't have them
    ###--- TODO: Fill in for remaining OSM road types
    speedLimitByRoadType = {'motorway':80, 'motorway_link':60, 'trunk':60, 'trunk_link':50, 'primary':50, 'primary_link':50, 'secondary':40, 'secondary_link':40, 'tertiary':30, 'tertiary_link':30, 'road':30}
    driveSpeedByRoadType = {'motorway':60, 'motorway_link':40, 'trunk':30, 'trunk_link':30, 'primary':30, 'primary_link':30, 'secondary':30, 'secondary_link':30, 'tertiary':30, 'tertiary_link':30, 'road':25}
    roadWidthByRoadType = {'motorway':21, 'motorway_link':10.5, 'trunk':14, 'trunk_link':7, 'primary':9, 'primary_link':4.5, 'secondary':6, 'secondary_link':3, 'tertiary':5.5,  'tertiary_link':2.75, 'road':6}


    ###=== To use projection on Point and Polygon, always_xy=True required
    standardToDistProj = createStandardToDistProj()
    distToStandardProj = createDistToStandardProj()

    thisComboIndex = thisGrid['comboIndex']

    ###=== Pad the area by a little bit, then trim down to the tile size after integration and component filtering
    ###=== 1 degree lat ~ 111km, 1 degree lon ~ 85km at 40deg latitude...so close enough
    bufferDegSizeX = 200 / 111030  ## a 300m buffer to include the areas around to avoid eliminating useful components.
    bufferDegSizeY = 200 / 85390  ## a 300m buffer to include the areas around to avoid eliminating useful components.

###----------------------- GET OSM DATA  ---------------------
#    startTime = time.time()
    thisAreaData = {}

    if useDatabase == True:
        print("=== this shouldn't happen ===")
        # loader = osm_loader.OsmLoader(
        #     database='osm_kanto',
        #     user='osm_user',
        #     password='osm_pass'
        # )
        # try:
        #     thisAreaData = loader.query(
        #         network_obj='way',
        #         tags=['highway'],
        #         bbox=(thisGrid.xMin - bufferDegSizeX, thisGrid.yMin - bufferDegSizeY,
        #                 thisGrid.xMax + bufferDegSizeX, thisGrid.yMax + bufferDegSizeY)
        #         )
        # except:
        #     print("=== Couldn't use local database, trying overpass API ===")
        #     useDatabase = False

    if useDatabase == False:   ###=== If it is not using the Database, try Overpass
        api = overpy.Overpass()
        theWayQuery = """
            [out:json][timeout:900];
            (way["highway"]({yMin},{xMin},{yMax},{xMax});
            );
            out body;
            (._;>;);
            out skel qt;
            """.format(xMin=str(thisGrid.xMin - bufferDegSizeX), yMin=str(thisGrid.yMin - bufferDegSizeY), xMax=str(thisGrid.xMax + bufferDegSizeX), yMax=str(thisGrid.yMax + bufferDegSizeY))

        overpassLookupFailed = True
        while (overpassLookupFailed == True):
            try:
                thisAreaData = api.query(theWayQuery)
                overpassLookupFailed = False
            except:
                print(" -- Road data for tile",thisComboIndex,"could not be pulled from overpass. Trying Again.")
                time.sleep(37)
                thisAreaData = api.query(theWayQuery)

#        reportRunTime(startTime, "pull data for one window:")


    ###----------------------- COLLECT ROAD EDGES ---------------------
    ###== Using pandas as an intermediary adds a bunch of None value attr to the network and is much slower
#        startTime = time.time()
    edgeDict = {}
    nodeDict = {}
#        edgeCount = 0

    if (isinstance(thisAreaData,dict) == False):
        if (len(thisAreaData.ways) > 0):
            for way in thisAreaData.ways:
                thisWayTags = {k:v for k,v in way.tags.items() if k in varsToKeep}
                theseNodes = way.get_nodes(resolve_missing=True)
                if len(theseNodes) > 1:
                    for index,node in enumerate(theseNodes[:-1]):
                        thisEdgeTags = thisWayTags.copy()
                        x1 = float(node.lon)
                        y1 = float(node.lat)
                        x2 = float((theseNodes[index+1]).lon)
                        y2 = float((theseNodes[index+1]).lat)
                        x1_dist, y1_dist = standardToDistProj.transform(x1, y1)
                        x2_dist, y2_dist = standardToDistProj.transform(x2, y2)
                        length = euclideanDistance(x1_dist, y1_dist, x2_dist, y2_dist)
                        thisEdgeTags["geometry"] = Point(x1,y1) if ((x1==x2) & (y1==y2)) else LineString([(x1,y1), (x2,y2)])
                        thisEdgeTags["geomDist"] = convertGeomCRS(thisEdgeTags["geometry"], standardToDistProj)
                        thisEdgeTags["x1"] = x1
                        thisEdgeTags["y1"] = y1
                        thisEdgeTags["x2"] = x2
                        thisEdgeTags["y2"] = y2
                        thisEdgeTags["length"] = length
                        thisEdgeTags["speed"] = walkingSpeed ##== this has to be changed, because there needs to be one per transportation (walk, bike, car, etc.).
                        thisEdgeTags["walking"] = True if thisEdgeTags.get("highway", None) not in walkingBadRoadTypes else False
                        if thisEdgeTags["walking"] == True:
                            thisEdgeTags["walkingSpeed"] = walkingSpeed
                            thisEdgeTags["walkingTime"] = metersToMinutes(length, theSpeed=thisEdgeTags["speed"])
                        thisEdgeTags["driving"] = True if thisEdgeTags.get("highway", None) not in drivingBadRoadTypes else False
                        if thisEdgeTags["driving"] == True:
                            thisEdgeTags["drivingSpeed"] = 35 ##== TODO:Convert this to the lookup table for kinds of roads
                            thisEdgeTags["drivingTime"] = metersToMinutes(length, theSpeed=thisEdgeTags["drivingSpeed"])
                        thisEdgeTags["modality"] = "road"
                        thisEdgeTags["timeWeight"] = metersToMinutes(length, theSpeed=thisEdgeTags["speed"])
                        ###=== If the source node isn't already in the network, add it, otherwise just add the edge from the existing source node.
                        if edgeDict.get(node.id, None) == None:
                            edgeDict[node.id] = {(theseNodes[index+1]).id: thisEdgeTags}
                        else:
                            edgeDict[node.id][theseNodes[index+1].id] = thisEdgeTags

                        ###----------------------- ADD ROAD NODE ATTRIBUTES ---------------------
                        nodeDict[node.id] = {}
                        nodeDict[node.id]['modality'] = 'road'
                        nodeDict[node.id]['lon'] = x1
                        nodeDict[node.id]['lat'] = y1
                        nodeDict[node.id]['geometry'] = Point(x1,y1)
                        nodeDict[node.id]['geomDist'] = convertGeomCRS(Point(x1,y1), standardToDistProj)

                        nodeDict[theseNodes[index+1].id] = {}
                        nodeDict[theseNodes[index+1].id]['modality'] = 'road'
                        nodeDict[theseNodes[index+1].id]['lon'] = x2
                        nodeDict[theseNodes[index+1].id]['lat'] = y2
                        nodeDict[theseNodes[index+1].id]['geometry'] = Point(x2,y2)
                        nodeDict[theseNodes[index+1].id]['geomDist'] = convertGeomCRS(Point(x2,y2), standardToDistProj)

        #                        edgeCount += 1

    ####=== Convert the edgeDF into NetworkX undirected graph
    thisNetwork = nx.Graph(edgeDict)

#        reportRunTime(startTime, "complete building the network from edges")
    ###-----------------------------------------------------------
    # print("  -- Number of nodes from Overpass for tile", gridNum,":",len(list(thisNetwork.nodes())))

    ###----------------------- IF THE TILE IS EMPTY ---------------------
    ###=== Check for whether the network is empty; if so save it now and stop, else keep going
    if len(edgeDict) == 0:
        # writePickleFile(thisNetwork, '../Data/OSMData/WalkabilityNetworkTiles_v5b/networkGrid='+thisComboIndex+'.pkl')
        ###=== Because this network is empty anyway, it's enough to just convert it to a directed graph type and save that.
#         thisNetwork = nx.Graph(thisNetwork)
#        writePickleFile(thisNetwork, '../Data/OSMData/FullNetworkTiles_v5/networkGrid='+thisComboIndex+'.pkl')

        if useMultiprocessing == True:
            printProgress(beginTime.value, rowNum.value, totalGridNum)
            # rowNum.value += 1
            return
        else:
            beginTime = printProgress(beginTime, rowNum, totalGridNum)
            # rowNum += 1
            return rowNum, beginTime
    ###-----------------------------------------------------------

    ######====== if the road network is not empty, continue processing ======

#    ###----------------------- ADD ROAD NODE ATTRIBUTES ---------------------
    nx.set_node_attributes(thisNetwork, nodeDict)

#    ####=== Add lat/lon to the road nodes from the edge info
#    for u,v,d in thisNetwork.edges(data=True):
#        thisNetwork.nodes[u]['modality'] = 'road'
#        thisNetwork.nodes[u]['lon'] = d['x1']
#        thisNetwork.nodes[u]['lat'] = d['y1']
#        thisNetwork.nodes[u]['geometry'] = Point(d['x1'], d['y1'])
#        thisNetwork.nodes[u]['geomDist'] = convertGeomCRS(thisNetwork.nodes[u]['geometry'], standardToDistProj)
#        thisNetwork.nodes[v]['modality'] = 'road'
#        thisNetwork.nodes[v]['lon'] = d['x2']
#        thisNetwork.nodes[v]['lat'] = d['y2']
#        thisNetwork.nodes[v]['geometry'] = Point(d['x2'], d['y2'])
#        thisNetwork.nodes[v]['geomDist'] = convertGeomCRS(thisNetwork.nodes[v]['geometry'], standardToDistProj)
##        reportRunTime(startTime, "add lat/lon to nodes")
#    ###-----------------------------------------------------------


    ###----------------------- GET THE STATION EXIT DATA FROM OSM ---------------------
    ###=== Now get the station exits and connect them to the road network and stations.
    ###=== This doesn't need a buffer, because it connects to stations from the whole network and we only care about exits within the tile
    thisAreaData = {}
    if useDatabase == True:
        try:
            thisAreaData = loader.query(
                network_obj='node',
                tags={'railway': ['subway_entrance', 'train_station_entrance']},
                bbox=(thisGrid.xMin, thisGrid.yMin, thisGrid.xMax, thisGrid.yMax)
            )
        except:
            print("=== Couldn't use local database for exits of tile",thisComboIndex,"===")
    else:  ## use Overpass API
        theExitQuery = """
            [out:json][timeout:900];
            (node["railway"="subway_entrance"]({yMin},{xMin},{yMax},{xMax});
                node["railway"="train_station_entrance"]({yMin},{xMin},{yMax},{xMax});
            );
            out body;
            (._;>;);
            out skel qt;
            """.format(xMin=str(thisGrid.xMin), yMin=str(thisGrid.yMin), xMax=str(thisGrid.xMax), yMax=str(thisGrid.yMax))
        try:
            thisAreaData = api.query(theExitQuery)
        except:
            print(" -- Exit data for tile",thisComboIndex,"could not pulled from overpass.")

    ###----------------------- PROCESS ROAD NODES/EDGES FOR CONNECTIONS ---------------------
    ###=== Create a DataFrame containing only the road nodes and edges relevant for connecting to exits and stores
    nodesToKeep = flattenList([[u,v] for u,v,attr in thisNetwork.edges(data=True) if ((attr.get('modality')=='road') & (attr.get('bridge')!=True) & (attr.get('tunnel')!=True) & (attr.get('crossing')!=True) & (attr.get('walking')!=False))])
    limitedRoadNetwork = getSubgraph(thisNetwork, nodesToKeep)
    ###=== To use intersects function, convert DataFrame to GeoDataFrame and use 'geomDist' as geometry
    limitedRoadEdges = gp.GeoDataFrame(nx.to_pandas_edgelist(limitedRoadNetwork), geometry='geomDist')
    roadNodeLocations = pd.DataFrame.from_dict(dict(limitedRoadNetwork.nodes(data=True)), orient='index')
#    print(roadNodeLocations.head())
    roadNodeLocations = gp.GeoDataFrame(roadNodeLocations, geometry='geomDist')

    ###===> TODO: limitedRoadNetwork contains the following incorrect edges.
    ### (2133066629 , 1105125707)
    ### (6397102736, 6397102742)
    ### (6188927991, 297162632)
    ### (1070862943, 6241744894)
    ### (1070863469, 4436218907)
    ### (6789046962, 588411742)
    ### (5110476588, 499185632)
    ### (1130812252, 3977436348)
    ### (3977436333, 1105125663)
    ### (499185467, 499185657)
    ###
    ### edges = [( row['source'],  row['target']) for index, row in limitedRoadEdgesDF.iterrows()]
    ### for u, v in limitedRoadNetwork.edges():
    ###     if (u, v) not in edges and (v, u) not in edges:
    ###         print(u, ",", v)
    ###
    ### print(limitedRoadNetwork.edges[2133066629 , 1105125707])

    ###----------------------- CONNECT EXITS TO ROADS AND STATIONS ---------------------
    if (isinstance(thisAreaData,dict) == False):
        if (len(thisAreaData.nodes) > 0):
            ###=== Create a DataFrame containing only the station location data for finding closest stations
            stationNodeLocations = gp.GeoDataFrame(pd.DataFrame.from_dict(dict(trainNetwork.nodes(data=True)), orient='index'), geometry='geomDist')
            stationNodeLocations = stationNodeLocations[stationNodeLocations['modality'] =='station']  ###== isolate the station nodes from the graph
            ###=== geomDist attributes exist in the trainNetwork data
            stationNodesWithData = [(node,attr) for node,attr in trainNetwork.nodes(data=True) if attr.get('modality', None) == 'station']
            thisNetwork.add_nodes_from(stationNodesWithData)  ###=== Instead of composing the networks, just add the station nodes

            for node in thisAreaData.nodes:
                ###=== Check for whether this exit is already a node in the road network, then only change its modality
                thisExitGeometry = Point(float(node.lon), float(node.lat))
                thisExitGeomDist = convertGeomCRS(thisExitGeometry, standardToDistProj)
                if node.id in list(thisNetwork.nodes()):
                    thisNetwork.nodes[node.id]['modality'] = 'exit'
                else:  ###=== If it's not an existing raod network, then add it as an exit node and connect it to the road network
                    thisNetwork.add_node(node.id, lon=float(node.lon), lat=float(node.lat), modality="exit", geometry=thisExitGeometry, geomDist=thisExitGeomDist)
                    limitedRoadEdges,roadNodeLocations = findClosestPointOnRoadAndConnect(thisNetwork, node.id, thisExitGeomDist, limitedRoadEdges, roadNodeLocations, standardToDistProj, distToStandardProj, "road")

                ###=== Connect a station exit node to all stations within 200m. If no station found, connect the exit node to the closeest station
                potentialStationNodes = stationNodeLocations[stationNodeLocations['geomDist'].intersects(thisExitGeomDist.buffer(200))]
                if len(potentialStationNodes) > 0:
#                    for stationIndex, stationGeomDist in zip(potentialStationNodes.index, potentialStationNodes['geometry'].values):
                    for stationIndex, stationGeomDist in zip(potentialStationNodes.index, potentialStationNodes['geomDist'].values):
#                        stationNodeDist = thisExitGeomDist.distance(stationGeomDist)
                        createWalkingEdge(thisNetwork, node.id, stationIndex, standardToDistProj, modality="stationAccess")
                else:
                ###=== Now connect this exit node to its nearest station
                    closestNodeNum, closestNodeDist = getClosestPointEuclidean(thisExitGeomDist, stationNodeLocations)
                    closestNodeIndex = stationNodeLocations.iloc[[closestNodeNum]].index[0]
                    createWalkingEdge(thisNetwork, node.id, closestNodeIndex, standardToDistProj, modality="stationAccess")

    #            reportRunTime(startTime, "add exit nodes")

    ###=== Find and connect disconnected components containing exit nodes...removes disconnected components without exits
    findAndConnectLooseExits(thisNetwork, standardToDistProj)
    ###-----------------------------------------------------------

    ###----------------------- CONNECT STORES TO ROADS ---------------------
    # startTime = time.time()
    ###=== Get and add the stores within this tile and connect to the nearest point on nearest edge.
#    theseStores = theseStores.copy() ###--- Why is the copy necessary?  I think theseStores is local anyway, so it shouldn't be necessary
#    theseStores['geomDist'] = [convertGeomCRS(value, standardToDistProj) for value in theseStores['geometry'].values]  ### Now geomDist in storeData

    ###--- create an STRtree for the road edges in this tile.
    # edgeGeomTree = createEdgeGeomTree(limitedRoadNetwork)

    numStoreNodes = len(theseStores)
    storeIntegrationStartTime = time.time()
    if (numStoreNodes) > 0:
        theseStoresDict = theseStores.set_index('nodeID').to_dict('index')  ## create a dictionary indexed by nodeID
        theseStoresDict = [(k,attr) for k,attr in theseStoresDict.items()]  ## convert to a list of dictionaries
        thisNetwork.add_nodes_from(theseStoresDict)

        ###=== Connect each store node to the closest point on the closest edges which may be an endpoint
        for num, thisStore in enumerate(getNodesByAttr(thisNetwork, 'modality', thisVal='store')):
            ###===> Attach to the closest point on the closest edge, which may be an endpoint
            thisStoreGeomDist = thisNetwork.nodes[thisStore]['geomDist']
            limitedRoadEdges,roadNodeLocations = findClosestPointOnRoadAndConnect(thisNetwork, thisStore, thisStoreGeomDist, limitedRoadEdges, roadNodeLocations, standardToDistProj, distToStandardProj, "storeAccess")

    theStoreConnectionTimes.append(time.time() - storeIntegrationStartTime)
    reportRunTime(storeIntegrationStartTime, "adding store nodes to tile")

    ###=== Adding store integration time for this tile to the shared variable, storeIntegrationTime
#    if useMultiprocessing:
#        storeIntegrationTime.value += time.time() - storeIntegrationStartTime
#        print('  -- accumulated store node integration time', storeIntegrationTime.value)
    ###-----------------------------------------------------------

    ###=== Now keep only the largest component, be careful that weird areas are not cut...maybe add size threshold too
    # components = list(sorted(nx.weakly_connected_components(thisNetwork), key=len, reverse=True))
    # thisNetwork = nx.DiGraph(thisNetwork.subgraph(components[0]))

    ###----------------------- FILTER NETWORK ELEMENTS AND ATTRIBUTES ---------------------
#     components = list(sorted(nx.connected_components(thisNetwork), key=len, reverse=True))
#     largeComponents = [comp for comp in components if len(comp) >= 100]
#     nodesToKeep = flattenList(largeComponents)
#     thisNetwork = getSubgraph(thisNetwork, nodesToKeep)  ###--- this can't be combined with the one later, because there are things we want to keep here, but delete below because they are outside the tile.

#     ###=== Reduce the network to the area within the grid
#     nodesToKeep = flattenList([[u,v] for u,v,attr in thisNetwork.edges(data=True) if attr['geometry'].intersects(thisGrid['geometry'])])
#     thisNetwork = getSubgraph(thisNetwork, nodesToKeep)
#     ###-----------------------------------------------------------

#     ###=== Elminate attributes with no or None value (to reduce file size)
#     allEdgeAttributes = getAllEdgeAttributes(thisNetwork)
#     for n1, n2, attr in thisNetwork.edges(data=True):
#         for thisAttr in allEdgeAttributes:
#             if attr.get(thisAttr, 'foo') != 'foo':
#                 if ((attr[thisAttr] == None) | (attr[thisAttr] == np.nan) | (attr[thisAttr] == '') | (attr[thisAttr] == ' ')):
#                     attr.pop(thisAttr, None)

#     ###=== Also, after all the nodes have been added, add the hexNum to each node for later area data Lookup
#     allNodeAttributes = getAllNodeAttributes(thisNetwork)
#     for node,attr in thisNetwork.nodes(data=True):
#         if attr.get('geometry', None) == None:
#             print('geometry not found for node:', node)
#             attr['geometry'] = Point(attr['lon'],attr['lat'])
#             thisNetwork.nodes[node]['geomDist'] = convertGeomCRS(thisNetwork.nodes[node]['geometry'], standardToDistProj)
#             # thisNetwork.nodes[node]['hexNum'] = getHexNumForPoint(Point(attr['lon'],attr['lat']), theHexData=hexData, node=attr)
#         # else:
#             # thisNetwork.nodes[node]['hexNum'] = getHexNumForPoint(attr['geometry'], theHexData=hexData, node=attr)
#         for thisAttr in allNodeAttributes:
#             if attr.get(thisAttr, 'foo') != 'foo':
#                 if ((attr[thisAttr] == None) | (attr[thisAttr] == np.nan) | (attr[thisAttr] == '') | (attr[thisAttr] == ' ')):
#                     attr.pop(thisAttr, None)
#     ###-----------------------------------------------------------

#     ###----------------------- PROCESS WALKABILITY NETWORK ---------------------
#     ###=== Create a slightly simplified network for use in walkability scoring
#     ###=== Simplify the graph for just walking (no driving) by making it undirected
#     walkabilityNetwork = thisNetwork.copy()  ## make a copy of the full network, because we're going to remove some elements
#     ###--- remove road edges that cannot be walked on, and the isolated nodes that it creates.
#     unwalkableEdges = [[u,v] for u,v,attr in walkabilityNetwork.edges(data=True) if attr.get("walking",None) == False]
#     walkabilityNetwork.remove_edges_from(unwalkableEdges)
#     walkabilityNetwork.remove_nodes_from(list(nx.isolates(walkabilityNetwork)))  ##-- needed because deleting unwalkableEdges leaves nodes

# #    print(getAllEdgeAttributes(walkabilityNetwork))
# #    print(getAllNodeAttributes(walkabilityNetwork))
#     edgeAttrToKeep = ['geometry', 'geomDist', 'bridge', 'x1', 'x2', 'speed', 'timeWeight', 'tunnel', 'crossing', 'walkingTime', 'length', 'y2', 'y1', 'modality']
#     edgeAttrToRemove = [x for x in getAllEdgeAttributes(walkabilityNetwork) if x not in edgeAttrToKeep]
#     for n1, n2, attr in walkabilityNetwork.edges(data=True):
#         for thisAttr in edgeAttrToRemove:
#             attr.pop(thisAttr, None)

#     nodeAttrToKeep = ['modality', 'category', 'city', 'geometry', 'geomDist', 'subCategory', 'subCategoryCode', 'prefecture', 'bestbashoCategory', 'categoryCode', 'lon', 'lat', 'lineName', 'lineNameEN', 'stationID', 'stationName', 'stationNameEN', "lines", 'linesEN']
#     nodeAttrToRemove = [x for x in getAllNodeAttributes(walkabilityNetwork) if x not in nodeAttrToKeep]
#     for node,attr in walkabilityNetwork.nodes(data=True):
#         for thisAttr in nodeAttrToRemove:
#             attr.pop(thisAttr, None)
    ###-----------------------------------------------------------

    ###----------------------- PROCESS STATIONFINDER NETWORK ---------------------
    ###=== Create a minimal walking network for use in the stationFinder API
    ###=== TODO: make further simplifications to improve runtime of searches (e.g. reduce nodes to intersections and interpolate edges)
#    stationFinderNetwork = walkabilityNetwork.copy()
#    ###=== Remove the stores (and, with them, their access edges)
#    storeNodes = [n for n,attr in stationFinderNetwork.nodes(data=True) if attr.get("modality",None) == "store"]
#    stationFinderNetwork.remove_nodes_from(storeNodes)
##            print(getAllEdgeAttributes(stationFinderNetwork))
##            print(getAllNodeAttributes(stationFinderNetwork))
#    ###=== Remove edge attributes that are not needed for this application
#    edgeAttrToKeep = ['modality', 'geometry', 'geomDist','length', 'x2', 'x1', 'y2', 'timeWeight', 'y1']
#    edgeAttrToRemove = [x for x in getAllEdgeAttributes(stationFinderNetwork) if x not in edgeAttrToKeep]
#    for n1, n2, attr in stationFinderNetwork.edges(data=True):
#        for thisAttr in edgeAttrToRemove:
#            attr.pop(thisAttr, None)
#
#    nodeAttrToKeep = ['stationName', 'geometry', 'geomDist', 'stationID', 'lat', 'stationNameEN', 'lon', 'modality', 'lines', 'linesEN']
#    nodeAttrToRemove = [x for x in getAllNodeAttributes(stationFinderNetwork) if x not in nodeAttrToKeep]
#    for node,attr in stationFinderNetwork.nodes(data=True):
#        for thisAttr in nodeAttrToRemove:
#            attr.pop(thisAttr, None)
    ###-----------------------------------------------------------

    ###=== Write the simple networks, the location depends on whether the Database is being used
    # writePickleFile(walkabilityNetwork, '../Data/OSMData/WalkabilityNetworkTiles_v5b/networkGrid='+thisComboIndex+'.pkl')
#    writePickleFile(stationFinderNetwork, '../Data/OSMData/StationFinderNetworkTiles_v5b/networkGrid='+thisComboIndex+'.pkl')


    ###================= CONVERT TO DIRECTED, ADD TRAIN NETWORK, AND PROCESS ONEWAY ROADS ====================

#    thisNetwork = nx.DiGraph(thisNetwork)  ### This automatically creates edges with identical attr in the other direction, so fix geoms, etc.
#
#    ###=== Add elevations to nodes
#
#    ###-- ... but how do we know which one: do this before we fix the the x1,y1, etc.  if the x1 doesn't match the source lon, then it's new => del
#    ###=== Fix the edge x1, y1, x2, y2, geometry, and geomDist based on the source/target nodes, and process oneway roads
#    for n1, n2, attr in thisNetwork.edges(data=True):
#        ###=== Go through the edges, and if oneway=True, delete the edge in the wrong direction
#        if attr.get('oneway', None) == True:
#            if attr.get('x1', None) != thisNetwork.nodes[n1]['lon']: ##-- the edge is flipped
#                thisNetwork.remove_edge(n1, n2)
#        else:
#            x1 = thisNetwork.nodes[n1]['lon']
#            y1 = thisNetwork.nodes[n1]['lat']
#            x2 = thisNetwork.nodes[n2]['lon']
#            y2 = thisNetwork.nodes[n2]['lat']
#            thisNetwork[n1][n2]["geometry"] = Point(x1,y1) if ((x1==x2) & (y1==y2)) else LineString([(x1,y1), (x2,y2)])
#            thisNetwork[n1][n2]["geomDist"] = convertGeomCRS(thisNetwork[n1][n2]["geometry"], standardToDistProj)
#            ###=== Add driving speeds etc to driving roads
#
#            ###=== Add slopes to edges
#
#    ###=== Now add the train network
#    thisNetwork = nx.compose(thisNetwork, trainNetwork)
#
#    ###=== Now add the bus network
#    thisNetwork = nx.compose(thisNetwork, busNetwork)
#
#     writePickleFile(thisNetwork, '../Data/OSMData/FullNetworkTiles_v5/networkGrid='+thisComboIndex+'.pkl')
    ####-------------------------------------------------

    reportRunTime(startTime, "row index "+str(gridNum)+" time")


#    print("== Row:", rowNum, "-----------------------------------------------")
#    print("  -- EdgeCount:", edgeCount)
#    print("  -- EdgeDict Length:", len(edgeDict))
#    print("  -- Number of Nodes:", thisNetwork.number_of_nodes(), "  |  Number of Edges:", thisNetwork.number_of_edges() )
#
#    print(list(thisNetwork.nodes(data=True))[1])
#    print(list(thisNetwork.edges(data=True))[1])
#





###----------------------- EXPORT FOR VISUALIZATION ---------------------
    ###=== Convert to pandas and save as csv for plotting on Kepler (to check for overlaps)
    # nodeDF, edgeDF = convertGraphToGeopandas(walkabilityNetwork)
#    print(len(nodeDF))
#    print(len(edgeDF))
    # writeGeoCSV(nodeDF, '../Data/OSMData/networkGrid_v5b_nodes='+thisComboIndex+'.csv')
    # writeGeoCSV(edgeDF, '../Data/OSMData/networkGrid_v5b_edges='+thisComboIndex+'.csv')
    ####-------------------------------------------------


    ###----------------------- FUNCTION OUTPUTS ---------------------
    if useMultiprocessing == True:
        printProgress(beginTime.value, rowNum.value, totalGridNum)
        # rowNum.value += 1
        return
    else:
        beginTime = printProgress(beginTime, rowNum, totalGridNum)
        # rowNum += 1
        return rowNum, beginTime
####======================================================================


if __name__ == '__main__':
    ###------------------------------------------------------------
    ###=== This is a flag for whether to use mulitprocessing or not
    useMultiprocessing = False
#    useMultiprocessing = False
    ###=== This is a flag for whether to use the database or overpassAPI version
    useDatabase = False
#    useDatabase = False

    #gridData = readGeoPickle('../Data/OSMData/networkGridLookup-TokyoArea-standardCRS.pkl')
    gridData = readGeoPandasCSV('../Data/OSMData/networkGridLookup-23Wards-standardCRS.csv')
    print("  -- Total number of Grids:", len(gridData))  ## 409 for 23 wards | 7699 for Tokyo Area
    trainNetwork = readPickleFile('../Data/trainNetworks/trainNetwork6d-TokyoArea.pkl')
    # storeData = readGeoPickle('../Data/storeBuildings-23Wards.pkl')
    storeData = readGeoPandasCSV('../Data/storeBuildings-23Wards.csv')
    hexData = loadCoreData()  ### load the hex coreData that contains just the hexNum, geometry, and totalPopulation
    # storeConnectionTimes= []
    storeConnectionTimes= readPickleFile('../Data/WalkabilityAnalysis/storeConnectionTimes.pkl')
    print("  -- Finished preloading data")

    if useMultiprocessing == True:
        print("Number of CPUs:", cpu_count())

        ###=== These variables are shared between pools
        manager = Manager()
        m_rowNum = manager.Value('i', 1)
        m_beginTime = manager.Value('d', time.time())
#        m_storeIntegrationTime = manager.Value('d', 0)

        with Pool(processes=cpu_count()) as pool:
            ###=== Prepare argument for each process
            args = [(
                useDatabase,
                useMultiprocessing,
                gridNum,
                len(gridData),
                thisGrid,
                storeData[storeData['geometry'].intersects(thisGrid['geometry'])],
                m_rowNum,
                m_beginTime,
                trainNetwork,
                hexData,
                storeConnectionTimes
            ) for gridNum, thisGrid in gridData.iterrows()]

            ###=== Delete unnecessary large data
            del gridData, storeData, trainNetwork, hexData
            gc.collect()

            ###=== Start multiprocessing
            pool.starmap(createTile, args)
    else:
        beginTime = time.time()
        rowNum = 0
        for gridNum, thisGrid in gridData.iterrows():
            rowNum += 1
        #    if ((gridNum >= 0) & (gridNum < 5)):  ## select row indices to do
            # if (gridNum >= 0): ## continue where we left off
            # if gridNum == 5:    ## run selected rows of the gridData
            # if (rowNum < 5):     ## run the first X tiles
            if (rowNum >= 288):  ## just run all of them
                rowNum, beginTime = createTile(
                    useDatabase,
                    useMultiprocessing,
                    gridNum,
                    len(gridData),
                    thisGrid,
                    storeData[storeData['geometry'].intersects(thisGrid['geometry'])],
                    rowNum,
                    beginTime,
                    trainNetwork,
                    hexData,
                    storeConnectionTimes
                )

    print(storeConnectionTimes)
    writePickleFile(storeConnectionTimes, '../Data/WalkabilityAnalysis/storeConnectionTimes2.pkl')
###=============================================================================================







####==== Check for edge and node property consistency
#thisAttribute = 'timeWeight'
##theseEdges = [[u,v] for u,v,attr in thisNetwork.edges(data=True) if attr.get(thisAttribute,None) == None]
##theseEdges = [[u,v] for u,v,attr in thisNetwork.edges(data=True) if attr.get(thisAttribute,None) != None]
#print(len(theseEdges))
#print("has",thisAttribute,list(set([attr.get('modality',None) for u,v,attr in thisNetwork.edges(data=True) if attr.get(thisAttribute,None) != None])))
#print("not",thisAttribute,list(set([attr.get('modality',None) for u,v,attr in thisNetwork.edges(data=True) if attr.get(thisAttribute,None) == None])))
#
#storeNodes = getNodesByAttr(thisNetwork, 'modality', thisVal='store')
#print(len(storeNodes))
#print(storeNodes[:5])
#for thisStore in storeNodes[:5]:
#    print(thisNetwork.nodes[thisStore])

#
















#####======================================== END OF FILE ===========================================
