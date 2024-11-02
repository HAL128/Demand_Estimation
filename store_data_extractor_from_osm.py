# -*- coding: utf-8 -*-
from helpers.helper_module import os, sys, math, gc, defaultdict, nx, gp, np, pd, split, scale, osm_loader
from helpers.helper_functions import *
pd.set_option('display.max_columns', None)


# ###-------------------- Setup the parameters --------------------

# # theArea = "23Wards"
theArea = "TokyoArea"


# ###-------------------- Get/Make a Grid File --------------------
# ###=== The old square grid file for network tiles (useful for tests because a tile is small)
# # gridData = readGeoPandasCSV('../Data/OSMData/networkGridLookup-23Wards-standardCRS.csv')

###=== Create a geoDataframe consisting of the 23Wards/TokyoArea Polygon to use as the gridData
if theArea == "23Wards":
    thisPolygon = readPickleFile('../Data/Polygons/wardsPolygon.pkl')
else:
    thisPolygon = readPickleFile('../Data/Polygons/tokyoAreaPolygon.pkl')

gridData = gp.GeoDataFrame(pd.DataFrame({'tileName': [theArea], 'geometry': [thisPolygon]}), geometry="geometry")
gridData['geomDist'] = gridData['geometry'].apply(lambda thisGeom: convertGeomCRS(thisGeom, standardToDistProj))
gridData['xMin'],gridData['yMin'],gridData['xMax'],gridData['yMax'] = (gridData['geometry'].bounds).iloc[0]

# print(gridData.head())

thisGrid = gridData.iloc[0]  ##-- there is only one row anyway
thisComboIndex = thisGrid['tileName']


# ###-------------------- Get the Store Name/Category Converter --------------------

# ###=== Create a translation dictionary from the conversion dataframe
conversionDF = readCSV('../Data/storeData-OSM_Categories_v4.csv', fillNaN='')
# print(list(conversionDF))  ##--> ['oldStoreCount', 'storeCount', 'oldType', 'newType', 'subcategory', 'category', 'scoreCurvature', 'concern', 'child_related', 'elderly_related', 'car_related', 'pet_related', 'fashion', 'homegoods', 'food']
conversionDF = conversionDF[conversionDF['storeCount'] > 0]  ##-- remove unused types
conversionDict = conversionDF.set_index('oldType').to_dict('index')  ##-- set the index to the type listed in OSM for lookup
# print(conversionDF.head())
# print(conversionDict)

###--------------- GET ALL RELEVANT NODES --------------
nodeTypes = ['shop', 'amenity', 'leisure', 'tourism']
varsToKeep = ['shop', 'amenity', 'leisure', 'tourism', 'name', 'sport']

# storeTypes = []
nodeDict = {}

for thisNodeType in nodeTypes:
    # print("  -- Collecting", thisNodeType, "data")
    thisAreaData = {}
    loader = osm_loader.OsmLoader(database='osm_kanto', user='osm_user', password='osm_pass')
    try:
        thisAreaData = loader.query(network_obj='node', tags=[thisNodeType], bbox=(thisGrid.xMin, thisGrid.yMin, thisGrid.xMax, thisGrid.yMax))
    except:
        print("=== Couldn't use local database ===")
    ###--------------------------------------------
    ###--- decifer tags using info from https://wiki.openstreetmap.org/wiki/Map_features
    if (isinstance(thisAreaData,dict) == False):
        if (len(thisAreaData.nodes) > 0):
            for node in thisAreaData.nodes:
                # thisNodeTags= {k:v for k,v in node.tags.items()}
                # if thisNodeTags.get(thisNodeType, '') == 'supermarket':
                #     print(node.tags.items())
                thisNodeTags = {k:v for k,v in node.tags.items() if k in varsToKeep}
                oldStoreType = thisNodeTags.get(thisNodeType, '')
                if oldStoreType in list(conversionDF['oldType']):
                    newStoreType = conversionDict[oldStoreType]['newType']
                    if newStoreType != "None":
                        x1 = float(node.lon)
                        y1 = float(node.lat)
                        nodeDict[node.id] = {}
                        nodeDict[node.id]['lon'] = x1
                        nodeDict[node.id]['lat'] = y1
                        nodeDict[node.id]['modality'] = 'store'
                        nodeDict[node.id]['geometry'] = Point(x1,y1)
                        nodeDict[node.id]['geomDist'] = convertGeomCRS(Point(x1,y1), standardToDistProj)
                        nodeDict[node.id]['geomAngle'] = convertGeomCRS(Point(x1,y1), standardToMapProj)
                        if thisNodeTags.get('name', '') != '':
                            nodeDict[node.id]['name'] = thisNodeTags.get('name', '')
                        ##-- the 'name:en' tag from OSM is not preserved in our PBF file (among others)
                        # if thisNodeTags.get('name:en', '') != '':
                        #     nodeDict[node.id]['name_EN'] = thisNodeTags.get('name:en', '')
                        # if thisNodeTags.get('sport', '') != '':
                        #     nodeDict[node.id]['sport'] = thisNodeTags.get('sport', '')

                        ###--- disambiguate store=fuel (sells coal,gas) and amenity=fuel => gas_station
                        if ((thisNodeType=='store') & (oldStoreType=='fuel')):
                            nodeDict[node.id]['storeType'] = 'fuel_shop'
                            nodeDict[node.id]['subcategory'] = 'shopping'
                            nodeDict[node.id]['category'] = 'retail'
                        else:
                            nodeDict[node.id]['storeType'] = newStoreType
                            nodeDict[node.id]['subcategory'] = conversionDict[oldStoreType]['subcategory']
                            nodeDict[node.id]['category'] = conversionDict[oldStoreType]['category']
                            if conversionDict[oldStoreType]['concern'] == True:
                                nodeDict[node.id]['concern'] = True
                            if conversionDict[oldStoreType]['child_related'] == True:
                                nodeDict[node.id]['child_related'] = True
                            if conversionDict[oldStoreType]['elderly_related'] == True:
                                nodeDict[node.id]['elderly_related'] = True
                            if conversionDict[oldStoreType]['car_related'] == True:
                                nodeDict[node.id]['car_related'] = True
                            if conversionDict[oldStoreType]['pet_related'] == True:
                                nodeDict[node.id]['pet_related'] = True
                            if conversionDict[oldStoreType]['fashion'] == True:
                                nodeDict[node.id]['fashion'] = True
                            if conversionDict[oldStoreType]['homegoods'] == True:
                                nodeDict[node.id]['homegoods'] = True
                            if conversionDict[oldStoreType]['food'] == True:
                                nodeDict[node.id]['food'] = True

                    # storeTypes.append(thisNodeTags.get(thisNodeType, ''))
        else:
            print(" -- Nothing Found")
    ###--------------------------------------------

print("Number of stores:",len(nodeDict))  ##==> 72399 in 23Wards,  164,373 in TokyoArea  (for the limited category data)
# print(nodeDict)

storeDF = gp.GeoDataFrame(pd.DataFrame.from_dict(nodeDict, orient='index'), geometry='geometry').fillna('')
print(storeDF.head())

# storeDF.at[3630300485,'subcategory'] = 'cafe'  ###=== this is fixed in the conversion dictionary
# storeDF.at[3630300485,'category'] = 'eatery'
# storeDF.at[3630300485,'food'] = True

# storeDF.at[5042281367,'storeType'] = 'restaurant'
# storeDF.at[5042281367,'subcategory'] = 'restaurant'
# storeDF.at[5042281367,'category'] = 'eatery'
# storeDF.at[5042281367,'food'] = True

# storeDF.at[5611525814,'storeType'] = 'jewelry'
# storeDF.at[5611525814,'subcategory'] = 'shopping'
# storeDF.at[5611525814,'category'] = 'retail'
# storeDF.at[5611525814,'homegoods'] = True

writeGeoCSV(storeDF, '../Data/StoreData/storeData-'+theArea+'_v6_OSM.csv')



# print(list(storeDF))
# ['lon', 'lat', 'geometry', 'geomDist', 'name', 'storeType', 'subcategory', 'category', 'food', 'homegoods', 'car_related', 'fashion', 'elderly_related', 'pet_related', 'sport', 'concern', 'child_related']

# print(list(set(list(storeDF['subcategory']))))
# allSubCats = ['bicycle_parking', 'parking', 'spa', 'healthcare', 'fitness', 'supermarket', 'post_secondary_school', 'hospital', 'office', 'gambling', 'shopping', 'convenience_store', 'specialty_food', 'vending_machine', 'restaurant', 'specialty_school', 'fire_station', 'post_office', 'hotel', 'community_center', 'bank', 'cinema', 'primary-school', 'gas_station', 'cafe', 'smoking_area', 'police', 'leisure', 'nursing_home', 'school', 'motorcycle_parking', 'gallery', 'beauty', 'religion', 'service', 'park', 'clinic', 'bar', 'laundry']

# print(list(set(list(storeDF['category']))))
# allCats = ['retail', 'eatery', 'motorcycle_parking', 'vending_machine', 'police', 'lodging', 'social', 'religion', 'parking', 'healthcare', 'fire_station', 'cultural', 'education', 'entertainment', 'bicycle_parking']














# ###--------------------------------------------
# ###=== Get the info from the first pull to generate the conversion Dict:
# storeCounts = getItemCounts(storeTypes)
# # print(storeCounts)
# allStoreTypes = sorted(list(set(storeTypes)))
# print(len(allStoreTypes))  ##=> 584


# conversionDF = pd.DataFrame({'storeCount': [v for k,v in storeCounts.items()], 'oldType': [k for k,v in storeCounts.items()]})
# print(conversionDF.head())
# writeCSV(conversionDF, '../Data/StoreData/storeData-OSM_Categories_v2.csv')

# print(allStoreTypes)
# ## allLeisureTypes = ['0', '1', '2', '3', '5', 'ChopSticks', 'Electronics_maintenance_shop', 'Konyaku Donya', 'Long_and_Short_Stays_for_Foreigners', 'Masking_tape', 'abandoned:drinking_water', 'accessories', 'administration', 'adult_gaming_centre', 'alcohol', 'amusement_arcade', 'animal_boarding', 'anime', 'antiques', 'antiques;art', 'apartment', 'appliance', 'aquarium', 'art', 'arts_centre', 'artwork', 'ashtray', 'atm', 'attraction', 'baby_care', 'baby_goods', 'bag', 'bakery', 'balloon', 'bank', 'bar', 'bar;cafe', 'bathhouse', 'bbq', 'beauty', 'bed', 'bell', 'bench', 'beverages', 'bicycle', 'bicycle_parking', 'bicycle_rental', 'bicycle_repair_station', 'biergarten', 'bitcoin', 'board_game', 'board_games', 'boat_rental', 'bookmaker', 'books', 'boutique', 'bowling_alley', 'brothel', 'bureau_de_change', 'bus_station', 'butcher', 'c', 'cafe', 'cake_shop', 'camera', 'camp_site', 'car', 'car_parts', 'car_rental', 'car_repair', 'car_sharing', 'car_wash', 'cargo', 'carpet', 'casino', 'chair', 'chalet', 'charging_station', 'charity', 'cheese', 'chemist', 'chicket', 'childcare', 'china', 'chocolate', 'chopsticks', 'cinema', 'clinic', 'clock', 'clothes', 'coffee', 'collector', 'college', 'community_centre', 'community_centre;social_facility', 'community_centre;theatre', 'computer', 'confectionery', 'conference_centre', 'convenience', 'copyshop', 'cosmetics', 'country_store', 'courthouse', 'coworking_space', 'craft', 'crematorium', 'curtain', 'dairy', 'dance', 'darts', 'deli', 'delivery', 'delivery_office', 'dentist', 'department_store', 'discount', 'doctors', 'dog_park', 'doityourself', 'dojo', 'drinking_water', 'drinking_water;toilets', 'driving_school', 'dry_cleaning', 'dry_food', 'e-cigarette', 'electric', 'electrical', 'electronic_parts', 'electronics', 'electronics_parts_shop', 'embassy', 'erotic', 'estate_agent', 'events_venue', 'exhibition_centre', 'fabric', 'farm', 'fashion', 'fashion_accessories', 'fast_food', 'ferry_terminal', 'fire_station', 'fishing', 'fishmonger', 'fitness_centre', 'fitness_station', 'florist', 'food', 'food_court', 'fountain', 'frame', 'frozen_food', 'fuel', 'funeral_directors', 'furniture', 'gallery', 'gambling', 'game_centre', 'games', 'garden', 'garden_centre', 'general', 'gift', 'glass', 'go', 'golf', 'golf_course', 'grave_yard', 'greengrocer', 'grit_bin', 'grocery', 'guest_house', 'hackerspace', 'hairdresser', 'hairdresser;photo', 'hairdresser_supply', 'hardware', 'hat', 'health_food', 'hearing_aids', 'hifi', 'history', 'hobby', 'hospital', 'hostel', 'hot_spring', 'hotel', 'household', 'household_linen', 'houseware', 'hunting', 'hunting_stand', 'ice_cream', 'ice_rink', 'information', 'interior_decoration', 'internet_cafe', 'iron bar', 'iron_bar', 'jewelry', 'juice_bar', 'karaoke', 'karaoke_box', 'kimono', 'kindergarten', 'kiosk', 'kitchen', 'laboratory_equipment', 'language_school', 'laundry', 'leather', 'library', 'lighting', 'liquor store', 'loading_dock', 'locker', 'locksmith', 'lottery', 'love_hotel', 'luggage_locker', 'magic_and_illusion_supplies', 'mahjong', 'mall', 'marina', 'marketplace', 'massage', 'massage;thai;reflexology', 'medical_supply', 'militally', 'miniature_golf', 'mobile_phone', 'model', 'money_lender', 'moneylender', 'monitor', 'motel', 'motorcycle', 'motorcycle_parking', 'motorcycle_repair', 'mount', 'museum', 'music', 'music_school', 'music_venue', 'musical_instrument', 'newsagent', 'nightclub', 'nursing_home', 'nursing_room', 'nutrition_supplements', 'office_supplies', 'office_supply', 'optician', 'organic', 'outdoor', 'outdoor_seating', 'pachinco', 'paint', 'park', 'parking', 'parking_entrance', 'parking_space', 'party', 'pastry', 'pawnbroker', 'perfumery', 'pet', 'pet_grooming', 'pharmacy', 'photo', 'photo_booth', 'photo_studio', 'picnic_site', 'picnic_table', 'pitch', 'place_of_worship', 'planetarium', 'playground', 'police', 'post_box', 'post_office', 'prep_school', 'private_school', 'psychic', 'pub', 'public_bath', 'public_bookcase', 'public_building', 'radiotechnics', 'raw_rice', 'recycling', 'religion', 'repair', 'restaurant', 'rice', 'salon', 'sauna', 'school', 'scuba_diving', 'seafood', 'second_hand', 'security', 'sento', 'separate', 'sewing', 'sewing_machines', 'shelter', 'shoe_repair', 'shoes', 'shogi', 'shopping_centre', 'shower', 'signet', 'signs', 'slipway', 'smoking_area', 'social_centre', 'social_facility', 'social_facility;community_centre', 'social_facility;community_centre;library', 'spices', 'sports', 'sports_centre', 'stadium', 'stationery', 'storage_rental', 'street_lamp', 'stripclub', 'studio', 'supermarket', 'support_center', 'swimming_pool', 'table', 'tailor', 'tattoo', 'taxi', 'tea', 'tea_leaf', 'telephone', 'telephone_box', 'theatre', 'theme_park', 'ticket', 'tobacco', 'toilets', 'townhall', 'toys', 'track', 'trade', 'trading_cards', 'training', 'travel_agency', 'tyres', 'university', 'vacant', 'variety_store', 'vehicle_inspection', 'vending_machine', 'vending_machine;smoking_area', 'veterinary', 'video', 'video_arcade', 'video_games', 'video_rental', 'viewpoint', 'waste_basket', 'waste_disposal', 'waste_transfer_station', 'watch', 'watches', 'water_park', 'water_point', 'wholesale', 'wilderness_hut', 'wine', 'wrapping', 'yes', 'zoo', 'メンズエステ', 'ラーメン屋', 'リフレッシュサロン', '居酒屋', '風俗,ソープランド']




