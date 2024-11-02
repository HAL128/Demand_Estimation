# -*- coding: utf-8 -*-
from helpers.helper_module import traceback, date, relativedelta, stdev, sm, torch, nn, optim, warnings
from helpers.helper_functions import *
from helpers.database_helpers import *
warnings.filterwarnings('ignore', '.*The array interface is deprecated.*', )
warnings.filterwarnings('ignore', '.*parts of a multi-part geometry.*', )
warnings.filterwarnings('ignore', '.*invalid value encountered in intersects*', )
warnings.filterwarnings('ignore', '.*will attempt to set the values*', )
pd.set_option('future.no_silent_downcasting', True)



dbConnection = DatabaseConnInfo(username='data_warehouse_owner', password='3x4mp13us3r')
###============================================================================

###=== get the centroid of a polygon or other geometry using the angle-preserving CRS and recast back to standard CRS
def getCentroid(geom):
   return convertGeomCRS((convertGeomCRS(geom, standardToMapProj)).centroid, mapToStandardProj)

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

def fromTensorToList(vector):
    return list(vector.detach().numpy())


# print(type(y_pred.to_numpy()))

# # ###==============================================================================================
# # ###================================== CLEAN RENT DATA AND MERGE WITH HEXES ==================================
# # ###==============================================================================================

# print("== Loading room data.")
# # roomData = pd.read_csv('../Data/ScoringData/rents.csv', encoding='utf-8', nrows=10).fillna(np.nan)
# # print(roomData.dtypes.to_dict())
# # print(list(roomData))
# ##-- ['id', 'story_id', 'normalized_name', 'address_lowest_level', 'normalized_address', 'prefecture', 'city_ward', 'town', 'street', 'block', 'number', 'subnumber', 'lat', 'lon', 'data_type', 'property_type', 'property_event', 'building_name', 'line_name_1', 'station_name_1', 'station_walk_minutes_1', 'station_walk_meters_1', 'rent', 'key_money_amount', 'key_money_months', 'deposit_amount', 'deposit_months', 'surface_area', 'management_fee', 'service_fee', 'floor_plan_type', 'floor_plan_number_of_rooms', 'room_floor', 'built_yyyymm', 'basement_floors', 'building_structure', 'building_floors', 'published_date', 'updated_at', 'pm_id', 'total_rent', 'adj_rent', 'rppa', 'built_year', 'layout', 'geometry', 'hex_id', 'built_month', 'built_date', 'age_in_months']

# roomData = pd.read_csv('../Data/ScoringData/rents.csv', encoding='utf-8').fillna(np.nan)
# # print(len(roomData)) ##==> 806,863

# print("== Processing room data.")
# roomData.replace('',np.nan, inplace=True)

# print("  -- rows without room numbers:", len([x for x in list(roomData['room_floor']) if not isNumber(x)]))  ##==> 4505
# roomData = roomData[roomData['room_floor'].apply(lambda x: isNumber(x))]

# print("  -- rows without building_floors:", len([x for x in list(roomData['building_floors']) if not isNumber(x)]))  ##==> 3766
# roomData = roomData[roomData['building_floors'].apply(lambda x: isNumber(x))]

# print("  -- rows with floor_plan_number_of_rooms > 10:", len([x for x in list(roomData['floor_plan_number_of_rooms']) if x > 10]))  ##==> 29
# roomData = roomData[roomData['floor_plan_number_of_rooms'].apply(lambda x: x <= 10)]

# def month_delta(start_date, end_date):
#     delta = relativedelta(end_date, start_date)
#     return makeInt(12 * delta.years + delta.months)

# # print(uniqueItems(roomData['built_month']))
# # print(roomData[roomData['built_month'] == 0].head())
# # print(len([x for x in list(roomData['room_floor']) if pd.notnull(x)]) / len(roomData))

# roomData['built_year'] = roomData['built_yyyymm'].apply(lambda val: int(str(val)[0:4]))
# roomData['built_month'] = roomData['built_yyyymm'].apply(lambda val: 6 if int(str(val)[4:6]) == 0 else int(str(val)[4:6]))
# roomData['built_date'] = roomData.apply(lambda row: date(row['built_year'],row['built_month'],1), axis=1)
# roomData['age_in_months'] = roomData['built_date'].apply(lambda val: month_delta(val, date(2024,2,1)))
# # roomData['age_in_months'] = roomData.apply(lambda row: makeInt((2 - row['built_month']) + (2 * (2024 - row['built_year']))), axis=1)
# # roomData['age_in_years'] = roomData['built_year'].apply(lambda val: 2024 - val)

# roomData.rename(columns={'rppa':'adj_rent_per_sqm'}, inplace=True)

# roomData['log_adj_rent_per_sqm'] = safeLog(list(roomData['adj_rent_per_sqm']))

# roomData = roomData[['id', 'story_id', 'updated_at', 'hex_id', 'geometry', 'log_adj_rent_per_sqm', 'adj_rent_per_sqm', 'adj_rent', 'surface_area', 'lat', 'lon', 'built_year', 'age_in_months', 'building_floors', 'room_floor', 'layout', 'floor_plan_type', 'floor_plan_number_of_rooms']]
# # print(roomData.head())

# print("== Saving room data.")
# writeCSV(roomData, '../Data/ScoringData/roomData.csv')
# ###------------------

# print("== Normalizing roomData data.")
# roomData = readGeoPandasCSV('../Data/ScoringData/roomData.csv').fillna(np.nan)

# def rescaleData(dataList, threshold = 0.001):
#     sortedList = sorted([x for x in dataList if pd.notnull(x)])
#     thresholdPosition = makeInt(len(sortedList) * threshold)
#     bottomLimit = sortedList[thresholdPosition]
#     upperLimit = sortedList[-thresholdPosition]
#     # return normalizeVariable(dataList, bottomLimit, upperLimit, True)
#     return [np.clip((x - bottomLimit)/(upperLimit - bottomLimit), 0, 1) if pd.notnull(x) else np.nan for x in dataList]

# varstoNorm = ['surface_area', 'lat', 'lon', 'built_year', 'age_in_months', 'building_floors', 'room_floor', 'floor_plan_number_of_rooms']

# for index,var in enumerate(varstoNorm):
#     print("  -- normalizing data for", var, "(", index+1, "of", len(varstoNorm), ")")
#     roomData[var] = roomData[var].astype(float).fillna(np.nan)
#     thisData = list(roomData[var])
#     varName = "unit_" + var + "_normed"
#     threshold = 0.001
#     roomData[varName] = rescaleData(thisData, threshold)


# print("== Processing Hex data.")
# hexData = get_entire_data_table(dbConnection, table='hexData_v02c8_normed')
# # writeCSV(hexData, '../Data/ScoringData/hexData_v02c8_normed.csv')
# # print(list(hexData))
# hexData.drop(columns=['geometry', 'connected', 'lat', 'lon', 'modality', 'nearby_hex_times_dict', 'nearby_stations_dict'], inplace=True)
# hexData.rename(columns={'id':'hex_id'}, inplace=True)
# hexData.replace('',np.nan, inplace=True)
# hexData['closest_station_time'] = hexData['closest_station_time'].astype(float)

# joinedData = pd.merge(roomData, hexData, on="hex_id", how="left")

# print(list(joinedData))

# joinedData = joinedData[['id', 'story_id', 'hex_id', 'updated_at', 'geometry', 'log_adj_rent_per_sqm', 'adj_rent_per_sqm', 'adj_rent', 'surface_area', 'lat', 'lon', 'built_year', 'age_in_months', 'building_floors', 'room_floor', 'layout', 'floor_plan_type', 'floor_plan_number_of_rooms', 'reachable_stations_dict', 'closest_station_id', 'closest_station_name', 'closest_station_nameEN', 'closest_station_time', 'pop_A_total', 'unit_surface_area_normed', 'unit_lat_normed', 'unit_lon_normed', 'unit_built_year_normed', 'unit_age_in_months_normed', 'unit_building_floors_normed', 'unit_room_floor_normed', 'unit_floor_plan_number_of_rooms_normed', 'acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed', 'acc_station_access_score_normed', 'acc_station_area_score_normed', 'econ_num_companies_normed', 'econ_num_jobs_normed', 'econ_demand_sCurve-90_10_normed', 'bld_num_buildings_normed', 'bld_mean_building_surface_area_normed', 'bld_percent_building_surface_area_normed', 'stores_weighted_relevant_normed', 'stores_total_relevant_normed', 'stores_total_normed', 'iya_pachinko_normed', 'iya_lovehotel_normed', 'iya_yakuza_normed', 'embassy_normed', 'embassy_log_normed', 'zone_cat=commercial_normed', 'zone_cat=industrial_normed', 'zone_cat=residential_1_normed', 'zone_cat=residential_2_normed', 'zone_type=commercial_normed', 'zone_type=industrial_normed', 'zone_type=industrial_specific_normed', 'zone_type=neighborhood_commercial_normed', 'zone_type=residential_class1_normed', 'zone_type=residential_class1_highrise_normed', 'zone_type=residential_class1_lowrise_normed', 'zone_type=residential_class2_normed', 'zone_type=residential_class2_highrise_normed', 'zone_type=residential_class2_lowrise_normed', 'zone_type=semi-industrial_normed', 'zone_type=semi-residential_normed', 'land_use=agriculture_normed', 'land_use=beach_normed', 'land_use=facility_normed', 'land_use=factory_normed', 'land_use=forest_normed', 'land_use=golf_normed', 'land_use=high-rise building_normed', 'land_use=low-rise dense_normed', 'land_use=low-rise sparse_normed', 'land_use=park_normed', 'land_use=rail_normed', 'land_use=rice field_normed', 'land_use=road_normed', 'land_use=sea_normed', 'land_use=vacant_normed', 'land_use=wasteland_normed', 'land_use=water_normed', 'veg=development_normed', 'veg=factory_normed', 'veg=field_normed', 'veg=forest_grassland_normed', 'veg=green_residential_normed', 'veg=lawn_normed', 'veg=other_normed', 'veg=park_cemetery_normed', 'veg=urban_normed', 'veg=vacant_normed', 'haz_flooding_time_normed', 'haz_flooding_depth_normed', 'haz_storm_surge_depth_normed', 'haz_debris_flow_normed', 'haz_landslide_normed', 'haz_slope_collapse_normed', 'pop_A_total_normed', 'pop_A_15yrOrLess_normed', 'pop_A_65yr+_normed', 'pop_A_30-44yr_normed', 'pop_A_25-64yr_normed', 'pop_total_employed_normed', 'pop_employees_normed', 'pop_self_employed_normed', 'pop_family_employee_normed', 'pop_total_households_normed', 'pop_home_owners_normed', 'pop_home_renters_normed', 'pop_total_homes_normed', 'pop_house_normed', 'pop_row_house_normed', 'pop_apt_building_normed', 'pop_apt_building_11+fl_normed', 'pop_other_home_type_normed', 'pop_mean_household_size_normed', 'pop_percent_female_normed', 'pop_percent_15yrOrLess_normed', 'pop_percent_65yr+_normed', 'pop_percent_30_44yr_normed', 'pop_percent_25_64yr_normed', 'pop_percent_owners_normed', 'pop_percent_house_normed', 'pop_percent_apt_building_normed', 'pop_percent_apt_building_11+fl_normed']]

# print("== Saving joined data.")
# writeCSV(joinedData, '../Data/ScoringData/joinedData.csv')
# ###=================================================






# # # ###==============================================================================================
# # # ###====================================== ANALYZE THE DATA ======================================
# # # ###==============================================================================================

# joinedData = readGeoPandasCSV('../Data/ScoringData/joinedData.csv').fillna(np.nan)
# print(len(joinedData))  ##==> 798,563

# joinedData.replace('',np.nan, inplace=True)
# joinedData['closest_station_time'] = joinedData['closest_station_time'].astype(float)

# ###=== plot some features of the room data
# varsToPlot = ['unit_surface_area_normed', 'unit_lat_normed', 'unit_lon_normed', 'unit_built_year_normed', 'unit_age_in_months_normed', 'unit_building_floors_normed', 'unit_room_floor_normed', 'unit_floor_plan_number_of_rooms_normed', 'acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed', 'acc_station_access_score_normed', 'acc_station_area_score_normed', 'econ_num_companies_normed', 'econ_num_jobs_normed', 'econ_demand_sCurve-90_10_normed', 'bld_num_buildings_normed', 'bld_mean_building_surface_area_normed', 'bld_percent_building_surface_area_normed', 'stores_weighted_relevant_normed', 'stores_total_relevant_normed', 'stores_total_normed', 'iya_pachinko_normed', 'iya_lovehotel_normed', 'iya_yakuza_normed', 'embassy_normed', 'embassy_log_normed', 'zone_type=commercial_normed', 'zone_type=industrial_normed', 'zone_type=industrial_specific_normed', 'zone_type=neighborhood_commercial_normed', 'zone_type=residential_class1_normed', 'zone_type=residential_class1_highrise_normed', 'zone_type=residential_class1_lowrise_normed', 'zone_type=residential_class2_normed', 'zone_type=residential_class2_highrise_normed', 'zone_type=residential_class2_lowrise_normed', 'zone_type=semi-industrial_normed', 'zone_type=semi-residential_normed', 'land_use=agriculture_normed', 'land_use=beach_normed', 'land_use=facility_normed', 'land_use=factory_normed', 'land_use=forest_normed', 'land_use=golf_normed', 'land_use=high-rise building_normed', 'land_use=low-rise dense_normed', 'land_use=low-rise sparse_normed', 'land_use=park_normed', 'land_use=rail_normed', 'land_use=rice field_normed', 'land_use=road_normed', 'land_use=sea_normed', 'land_use=vacant_normed', 'land_use=wasteland_normed', 'land_use=water_normed', 'veg=development_normed', 'veg=factory_normed', 'veg=field_normed', 'veg=forest_grassland_normed', 'veg=green_residential_normed', 'veg=lawn_normed', 'veg=other_normed', 'veg=park_cemetery_normed', 'veg=urban_normed', 'veg=vacant_normed', 'haz_flooding_time_normed', 'haz_flooding_depth_normed', 'haz_storm_surge_depth_normed', 'haz_debris_flow_normed', 'haz_landslide_normed', 'haz_slope_collapse_normed', 'pop_A_total_normed', 'pop_A_15yrOrLess_normed', 'pop_A_65yr+_normed', 'pop_A_30-44yr_normed', 'pop_A_25-64yr_normed', 'pop_total_employed_normed', 'pop_employees_normed', 'pop_self_employed_normed', 'pop_family_employee_normed', 'pop_total_households_normed', 'pop_home_owners_normed', 'pop_home_renters_normed', 'pop_total_homes_normed', 'pop_house_normed', 'pop_row_house_normed', 'pop_apt_building_normed', 'pop_apt_building_11+fl_normed', 'pop_other_home_type_normed', 'pop_mean_household_size_normed', 'pop_percent_female_normed', 'pop_percent_15yrOrLess_normed', 'pop_percent_65yr+_normed', 'pop_percent_30_44yr_normed', 'pop_percent_25_64yr_normed', 'pop_percent_owners_normed', 'pop_percent_house_normed', 'pop_percent_apt_building_normed', 'pop_percent_apt_building_11+fl_normed']

# # print(max(list(roomData['closest_station_time'])))

# thisArea = bufferGeometry(Point(139.73634781089083, 35.6722465355721), 10000)
# filteredData = joinedData.copy()
# filteredData = filteredData[((filteredData['built_year'] >= 2010) & (filteredData['built_year'] < 2015))]
# filteredData = filteredData[((filteredData['closest_station_time'] >= 4) & (filteredData['closest_station_time'] < 10))]
# filteredData = filteredData[filteredData['geometry'].intersects(thisArea)]
# print("== number of properies in filtered data:", len(filteredData))

# yVar = 'log_adj_rent_per_sqm'
# yData = list(filteredData[yVar])

# ##== look for a pattern in rent/sqm across sizes
# for var in varsToPlot:
#     xVar = var
#     xData = list(filteredData[xVar])
#     try:
#         makeDensityPlot(xData, yData, xVar, yVar, titleText='', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Data Scatterplots/scatter_'+xVar+'_'+yVar+'.png')
#     except:
#         print("  -- Could not make plot for", var)

#     # makeDensityPlot(xData, yData, xVar, yVar, titleText='', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Data Scatterplots/scatter_'+xVar+'_'+yVar+'.png')




###================================== Variable Set Definitions ==============================

allFeatures = ['unit_surface_area_normed', 'unit_lat_normed', 'unit_lon_normed', 'unit_built_year_normed', 'unit_age_in_months_normed', 'unit_building_floors_normed', 'unit_room_floor_normed', 'unit_floor_plan_number_of_rooms_normed', 'acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed', 'acc_station_access_score_normed', 'acc_station_area_score_normed', 'econ_num_companies_normed', 'econ_num_jobs_normed', 'econ_demand_sCurve-90_10_normed', 'bld_num_buildings_normed', 'bld_mean_building_surface_area_normed', 'bld_percent_building_surface_area_normed', 'stores_weighted_relevant_normed', 'stores_total_relevant_normed', 'stores_total_normed', 'iya_pachinko_normed', 'iya_lovehotel_normed', 'iya_yakuza_normed', 'embassy_normed', 'embassy_log_normed', 'zone_cat=commercial_normed', 'zone_cat=industrial_normed', 'zone_cat=residential_1_normed', 'zone_cat=residential_2_normed', 'zone_type=commercial_normed', 'zone_type=industrial_normed', 'zone_type=industrial_specific_normed', 'zone_type=neighborhood_commercial_normed', 'zone_type=residential_class1_normed', 'zone_type=residential_class1_highrise_normed', 'zone_type=residential_class1_lowrise_normed', 'zone_type=residential_class2_normed', 'zone_type=residential_class2_highrise_normed', 'zone_type=residential_class2_lowrise_normed', 'zone_type=semi-industrial_normed', 'zone_type=semi-residential_normed', 'land_use=agriculture_normed', 'land_use=beach_normed', 'land_use=facility_normed', 'land_use=factory_normed', 'land_use=forest_normed', 'land_use=golf_normed', 'land_use=high-rise building_normed', 'land_use=low-rise dense_normed', 'land_use=low-rise sparse_normed', 'land_use=park_normed', 'land_use=rail_normed', 'land_use=rice field_normed', 'land_use=road_normed', 'land_use=sea_normed', 'land_use=vacant_normed', 'land_use=wasteland_normed', 'land_use=water_normed', 'veg=development_normed', 'veg=factory_normed', 'veg=field_normed', 'veg=forest_grassland_normed', 'veg=green_residential_normed', 'veg=lawn_normed', 'veg=other_normed', 'veg=park_cemetery_normed', 'veg=urban_normed', 'veg=vacant_normed', 'haz_flooding_time_normed', 'haz_flooding_depth_normed', 'haz_storm_surge_depth_normed', 'haz_debris_flow_normed', 'haz_landslide_normed', 'haz_slope_collapse_normed', 'pop_A_total_normed', 'pop_A_15yrOrLess_normed', 'pop_A_65yr+_normed', 'pop_A_30-44yr_normed', 'pop_A_25-64yr_normed', 'pop_total_employed_normed', 'pop_employees_normed', 'pop_self_employed_normed', 'pop_family_employee_normed', 'pop_total_households_normed', 'pop_home_owners_normed', 'pop_home_renters_normed', 'pop_total_homes_normed', 'pop_house_normed', 'pop_row_house_normed', 'pop_apt_building_normed', 'pop_apt_building_11+fl_normed', 'pop_other_home_type_normed', 'pop_mean_household_size_normed', 'pop_percent_female_normed', 'pop_percent_15yrOrLess_normed', 'pop_percent_65yr+_normed', 'pop_percent_30_44yr_normed', 'pop_percent_25_64yr_normed', 'pop_percent_owners_normed', 'pop_percent_house_normed', 'pop_percent_apt_building_normed', 'pop_percent_apt_building_11+fl_normed']


wideVars = ['econ_demand_sCurve-90_10_normed', 'acc_station_access_score_normed']

unitVars = ['unit_surface_area_normed', 'unit_built_year_normed', 'unit_age_in_months_normed', 'unit_building_floors_normed', 'unit_room_floor_normed', 'unit_floor_plan_number_of_rooms_normed']

accVars = ['acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed', 'acc_station_area_score_normed']

econVars = ['econ_num_companies_normed', 'econ_num_jobs_normed']

bldVars = ['bld_num_buildings_normed', 'bld_mean_building_surface_area_normed', 'bld_percent_building_surface_area_normed']

storeVars = ['stores_weighted_relevant_normed', 'stores_total_relevant_normed', 'stores_total_normed']

iyaVars = ['iya_pachinko_normed', 'iya_lovehotel_normed', 'iya_yakuza_normed']

emabassyVars = ['embassy_normed', 'embassy_log_normed']

zoneCatVars =  ['zone_cat=commercial_normed', 'zone_cat=industrial_normed', 'zone_cat=residential_1_normed', 'zone_cat=residential_2_normed']

zoneTypeVars = ['zone_type=commercial_normed', 'zone_type=industrial_normed', 'zone_type=industrial_specific_normed', 'zone_type=neighborhood_commercial_normed', 'zone_type=residential_class1_normed', 'zone_type=residential_class1_highrise_normed', 'zone_type=residential_class1_lowrise_normed', 'zone_type=residential_class2_normed', 'zone_type=residential_class2_highrise_normed', 'zone_type=residential_class2_lowrise_normed', 'zone_type=semi-industrial_normed', 'zone_type=semi-residential_normed']

landUseVars = ['land_use=agriculture_normed', 'land_use=beach_normed', 'land_use=facility_normed', 'land_use=factory_normed', 'land_use=forest_normed', 'land_use=golf_normed', 'land_use=high-rise building_normed', 'land_use=low-rise dense_normed', 'land_use=low-rise sparse_normed', 'land_use=park_normed', 'land_use=rail_normed', 'land_use=rice field_normed', 'land_use=road_normed', 'land_use=sea_normed', 'land_use=vacant_normed', 'land_use=wasteland_normed', 'land_use=water_normed']

vegVars = ['veg=development_normed', 'veg=factory_normed', 'veg=field_normed', 'veg=forest_grassland_normed', 'veg=green_residential_normed', 'veg=lawn_normed', 'veg=other_normed', 'veg=park_cemetery_normed', 'veg=urban_normed', 'veg=vacant_normed']

hazVars = ['haz_flooding_time_normed', 'haz_flooding_depth_normed', 'haz_storm_surge_depth_normed', 'haz_debris_flow_normed', 'haz_landslide_normed', 'haz_slope_collapse_normed']

popVars = ['pop_A_total_normed', 'pop_A_15yrOrLess_normed', 'pop_A_65yr+_normed', 'pop_A_30-44yr_normed', 'pop_A_25-64yr_normed', 'pop_total_employed_normed', 'pop_employees_normed', 'pop_self_employed_normed', 'pop_family_employee_normed', 'pop_total_households_normed', 'pop_home_owners_normed', 'pop_home_renters_normed', 'pop_total_homes_normed', 'pop_house_normed', 'pop_row_house_normed', 'pop_apt_building_normed', 'pop_apt_building_11+fl_normed', 'pop_other_home_type_normed', 'pop_mean_household_size_normed']

popPercentVars = ['pop_percent_female_normed', 'pop_percent_15yrOrLess_normed', 'pop_percent_65yr+_normed', 'pop_percent_30_44yr_normed', 'pop_percent_25_64yr_normed', 'pop_percent_owners_normed', 'pop_percent_house_normed', 'pop_percent_apt_building_normed', 'pop_percent_apt_building_11+fl_normed']

allVars = unitVars + wideVars + accVars + econVars + bldVars + storeVars + iyaVars + emabassyVars + zoneTypeVars + landUseVars + vegVars + hazVars + popPercentVars

# mitaFeatures = [
#         "unit_surface_area_normed",
#         "unit_room_floor_normed",
#         "unit_floor_plan_number_of_rooms_normed",
#         "unit_building_floors_normed",
#         "unit_built_year_normed",
#         "unit_age_in_months_normed",
#         "acc_closest_station_time_normed",
#         "acc_num_nearby_hexes_normed",
#         "acc_num_nearby_stations_normed",
#         "acc_station_access_score_normed",
#         "acc_station_area_score_normed",
#         "econ_num_companies_normed",
#         "econ_num_jobs_normed",
#         "econ_demand_sCurve-90_10_normed",
#         "bld_num_buildings_normed",
#         "bld_mean_building_surface_area_normed",
#         "bld_percent_building_surface_area_normed",
#         "stores_weighted_relevant_normed",
#         "stores_total_relevant_normed",
#         "stores_total_normed",
#         "iya_pachinko_normed",
#         "iya_lovehotel_normed",
#         "iya_yakuza_normed",
#         "embassy_normed",
#         "embassy_log_normed",
#         "zone_type=commercial_normed",
#         "zone_type=industrial_normed",
#         "zone_type=industrial_specific_normed",
#         "zone_type=neighborhood_commercial_normed",
#         "zone_type=residential_class1_normed",
#         "zone_type=residential_class1_highrise_normed",
#         "zone_type=residential_class1_lowrise_normed",
#         "zone_type=residential_class2_normed",
#         "zone_type=residential_class2_highrise_normed",
#         "zone_type=residential_class2_lowrise_normed",
#         "zone_type=semi-industrial_normed",
#         "zone_type=semi-residential_normed",
#         "land_use=agriculture_normed",
#         "land_use=beach_normed",
#         "land_use=facility_normed",
#         "land_use=factory_normed",
#         "land_use=forest_normed",
#         "land_use=golf_normed",
#         "land_use=high-rise building_normed",
#         "land_use=low-rise dense_normed",
#         "land_use=low-rise sparse_normed",
#         "land_use=park_normed",
#         "land_use=rail_normed",
#         "land_use=rice field_normed",
#         "land_use=road_normed",
#         "land_use=sea_normed",
#         "land_use=vacant_normed",
#         "land_use=wasteland_normed",
#         "land_use=water_normed",
#         "veg=development_normed",
#         "veg=factory_normed",
#         "veg=field_normed",
#         "veg=forest_grassland_normed",
#         "veg=green_residential_normed",
#         "veg=lawn_normed",
#         "veg=other_normed",
#         "veg=park_cemetery_normed",
#         "veg=urban_normed",
#         "veg=vacant_normed",
#         "haz_flooding_time_normed",
#         "haz_flooding_depth_normed",
#         "haz_storm_surge_depth_normed",
#         "haz_debris_flow_normed",
#         "haz_landslide_normed",
#         "haz_slope_collapse_normed",
#         "pop_percent_female_normed",
#         "pop_percent_15yrOrLess_normed",
#         "pop_percent_65yr+_normed",
#         "pop_percent_30_44yr_normed",
#         "pop_percent_25_64yr_normed",
#         "pop_percent_owners_normed",
#         "pop_percent_house_normed",
#         "pop_percent_apt_building_normed",
#         "pop_percent_apt_building_11+fl_normed"
#     ]

# extraFeatures = [val for val in mitaFeatures if val not in allVars]
# print(extraFeatures)

# missingFeatures = [val for val in allVars if val not in mitaFeatures]
# print(missingFeatures)


# print(len(mitaFeatures),"vs",len(allVars))
# print(set(mitaFeatures) == set(allVars))

# ###================================== FIX DATA FOR OLS AND NN ==============================

# print("== Preparing Room Data.")
# joinedData = readGeoPandasCSV('../Data/ScoringData/joinedData.csv').fillna(np.nan)

# # print(list(joinedData))
# joinedData['log_adj_rent'] = safeLog(list(joinedData['adj_rent']))
# joinedData.replace('',np.nan, inplace=True)


# ##== filter to data with values for all the dependent vars (OLS can't handle nan or inf)
# for thisVar in allFeatures:
#     joinedData[thisVar] = np.asarray(joinedData[thisVar].astype(float))
#     joinedData = joinedData[~joinedData[thisVar].isna()]

# print("== Number of room in joinedData after filtering missing data:", len(joinedData))  ##==> 798,563 --> 793,375

# writeCSV(joinedData, '../Data/ScoringData/refinedData.csv')

# # ###==============================================================================================

# joinedData = readGeoPandasCSV('../Data/ScoringData/olsResultsData.csv').fillna(np.nan)

# print(list(joinedData))

# joinedData.drop(columns=['error_onlyRoomlog_adj_rent_per_sqm', 'error_onlyWidelog_adj_rent_per_sqm', 'error_onlyNeighlog_adj_rent_per_sqm', 'error_allVarslog_adj_rent_per_sqm', 'error_onlyRoomadj_rent_per_sqm', 'error_onlyWideadj_rent_per_sqm', 'error_onlyNeighadj_rent_per_sqm', 'error_allVarsadj_rent_per_sqm', 'error_onlyRoomlog_adj_rent', 'error_onlyWidelog_adj_rent', 'error_onlyNeighlog_adj_rent', 'error_allVarslog_adj_rent', 'error_onlyRoomadj_rent', 'error_onlyWideadj_rent', 'error_onlyNeighadj_rent', 'error_allVarsadj_rent'], inplace=True)

# writeCSV(joinedData, '../Data/ScoringData/refinedData.csv')
# # ###==============================================================================================



# ###==============================================================================================
# ###================================== HEDONIC OLS PRICING ANALYSIS ==============================
# ###==============================================================================================

# def revertPrice(thePrices, priceType, surfaceAreas):
#     thePrices = thePrices.to_numpy()
#     if priceType == 'log_adj_rent_per_sqm':
#         revertedPrices = np.multiply(np.exp(thePrices), surfaceAreas)
#     elif priceType == 'adj_rent_per_sqm':
#         revertedPrices = np.multiply(thePrices, surfaceAreas)
#     elif priceType == 'log_adj_rent':
#         revertedPrices = np.exp(thePrices)
#     else:
#         revertedPrices = thePrices
#     return revertedPrices


# print("== Preparing Room Data.")
# joinedData = readGeoPandasCSV('../Data/ScoringData/refinedData.csv').fillna(np.nan)


# # onlyRoom = unitVars
# # onlyWide = unitVars + wideVars
# # onlyNeigh = unitVars + accVars + econVars + bldVars + storeVars + iyaVars + emabassyVars + zoneTypeVars + landUseVars + vegVars + hazVars + popPercentVars
# # selectVars1 = unitVars + wideVars + ['acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed'] + bldVars + ['stores_weighted_relevant_normed']
# # selectVars2 = unitVars + wideVars + ['acc_closest_station_time_normed', 'acc_num_nearby_hexes_normed', 'acc_num_nearby_stations_normed'] + bldVars + ['stores_weighted_relevant_normed'] + iyaVars + ['land_use=high-rise building_normed', 'land_use=low-rise dense_normed', 'land_use=low-rise sparse_normed', 'land_use=park_normed']
# # allVars = unitVars + wideVars + accVars + econVars + bldVars + storeVars + iyaVars + emabassyVars + zoneTypeVars + landUseVars + vegVars + hazVars + popPercentVars  ##-- 79
# # allVars = unitVars + wideVars  ##-- 8

# ###--- try adding a nonlinear term for the unit surface area
# joinedData["unit_surface_area_normed_squared"] = joinedData["unit_surface_area_normed"].apply(lambda val: val * val)
# allVars = allVars + ["unit_surface_area_normed_squared"]

# print("  -- Number of explanatory variables:", len(allVars))  ##-- 79
# #
# # standardizeVars = True
# standardizeVars = False

# if standardizeVars == True:
#     print("== Standardizing the variables.")
#     for var in allVars:
#         joinedData[var] = standardizeVariable(joinedData[var])


# print("== Splitting train and test sets.")

# ##=== Use the last month as test data
# holdout = "lastMonth"
# dates = pd.to_datetime(joinedData["updated_at"])
# one_month_ago = dates.max() - pd.Timedelta(days=30)
# is_test = dates > one_month_ago
# trainData = joinedData[~is_test]
# testData = joinedData[is_test]

# # ###=== Use different areas as test data
# # cityData = readGeoPandasCSV('../Data/GISandAddressData/adminAreaPoly_cities-AllJapan_simp.csv')
# # # print(cityData.head())
# # # areaName = "新宿"
# # # holdout = "Shinjuku"
# # areaName = "杉並"
# # holdout = "Suginami"
# # cityPolygon = list(cityData[((cityData['prefName'] == "東京都") & (cityData['cityName'].str.contains(areaName) ))]['geometry'])[0]
# # # print(cityPolygon)
# # is_test = joinedData['geometry'].intersects(cityPolygon)


# ###-----------------
# trainData = joinedData[~is_test]
# testData = joinedData[is_test]
# print("  -- Number of train data samples:", len(trainData))  ##-- 745,875   suginami => 765347   shinjuku => 753091
# print("  -- Number of test data samples:", len(testData))    ##--  47,500   suginami =>  28028   shinjuku => 40284

# y_true = trainData['adj_rent'].to_numpy()
# y_test_true = testData['adj_rent'].to_numpy()
# train_surfaces = trainData['surface_area'].to_numpy()
# test_surfaces = testData['surface_area'].to_numpy()

# print(trainData.head())

# print(np.mean(joinedData['adj_rent']))
# print()

# ###--------------------------------

# # ###=== Define experiments
# # varSets = [onlyRoom, onlyWide, onlyNeigh, allVars]
# # varSetLabels = ["onlyRoom", "onlyWide", "onlyNeigh", "allVars"]
# # varSetTitles = ["Only Property Variables", "Property and Wide Area Variables", "Property and Neighborhood Variables", "All Variables"]

# varSets = [allVars]
# varSetLabels = ["allVars"]
# varSetTitles = ["All Variables"]

# # model1 = sm.OLS(joinedData[priceVars[0]], sm.add_constant(joinedData[explVars1])).fit()
# # model1.summary()

# ###--------------------------------
# priceVars = ['log_adj_rent_per_sqm', 'adj_rent_per_sqm', 'log_adj_rent', 'adj_rent']
# priceLabels = ['log adjusted rent/m$^2$', 'adjusted rent/m$^2$', 'log adjusted rent', 'adjusted rent']
# priceTitles = ['Log Adjusted Rent/m$^2$', 'Adjusted Rent/m$^2$', 'Log Adjusted Rent', 'Adjusted Rent']

# print("== Performing OLS analysis.")
# for priceNum,thisPrice in enumerate(priceVars[:1]):
#     print(" ")
#     # print("  -------------------------- Analysis against", thisPrice, "--------------------------")
#     print("  -------------------------------------------------------------------------------")
#     for setNum,thisVarSet in enumerate(varSets):
#         thisLabel = varSetLabels[setNum] + "_" + thisPrice + "_" + holdout + "_sqrd"
#         # thisTitle = varSetTitles[setNum] + " Using " + priceTitles[priceNum]
#         print("  -------------------------- Analysis for", thisLabel, "--------------------------")


        # X = trainData[allVars]
        # y = trainData[thisPrice]
        # X_test = testData[allVars]
        # y_test = testData[thisPrice]

        # thisModel = sm.OLS(y, sm.add_constant(X)).fit()
        # print(thisModel.summary())

        # y_pred = thisModel.predict(sm.add_constant(X))
        # test_y_pred = thisModel.predict(sm.add_constant(X_test))

        # adj_rent_pred = revertPrice(y_pred, thisPrice, train_surfaces)
        # adj_rent_pred_test = revertPrice(test_y_pred, thisPrice, test_surfaces)

        # print("    -- train", thisLabel, " -- MAPE:", MAPE(y_true, adj_rent_pred), "% | MAE:", MAE(y_true, adj_rent_pred, 0) , "| R2:", getR2(y_true, adj_rent_pred) )
        # print("    -- test", thisLabel, " -- MAPE:", MAPE(y_test_true, adj_rent_pred_test), "% | MAE:", MAE(y_test_true, adj_rent_pred_test, 0) , "| R2:", getR2(y_test_true, adj_rent_pred_test) )

#         with open('../Demand Estimation/OLS Results/OLS_summary_'+thisLabel+'.txt', 'w') as fh:
#             fh.write(thisModel.summary().as_text())
#         with open('../Demand Estimation/OLS Results/OLS_summary_'+thisLabel+'.tex', 'w') as fh:
#             fh.write(thisModel.summary().as_latex())
#         # print("    --", thisLabel, "and", thisPrice, "MSE =", makeInt(thisModel.mse_model))
#         # print("    --", thisLabel, "and", thisPrice, "R2 =", rnd(thisModel.rsquared))
#         # trainData["error_"+varSetLabels[setNum]+thisPrice] = thesePreds - trainData[thisPrice]

#         # makeDensityPlot(adj_rent_pred_test, y_test_true, "predicted rent", "actual rent", titleText="Accuracy of OLS Model", numBins=200, maxCount=None, colorScheme=None, figSize=7, scaledAxes=True, bestFit=[1,2,3], diagonalLine=True, plotRange=None, filename='../Demand Estimation/OLS Results/scatter_predictions_'+thisLabel+'.png')

#         makeDensityPlot(adj_rent_pred_test, y_test_true, "test data predicted rent", "test data actual rent", titleText='Accuracy of OLS Model', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=True, bestFit=[1,2,3], diagonalLine=True, plotRange=None, addMargin=False, filename='../Demand Estimation/OLS Results/scatter_predictions_'+thisLabel+'.png')

#         theResids = thisModel.resid
#         makeDensityPlot(adj_rent_pred, theResids, "test data predicted rent", "test data prediction residual", titleText="Residuals for OLS Model", numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, filename='../Demand Estimation/OLS Results/scatter_residuals_'+thisLabel+'.png')


# # ###--- temporal holdout
# #     # -- train allVars_log_adj_rent_per_sqm and log_adj_rent_per_sqm MAPE: 11.587 % | MAE: 13234.0 | R2: 0.84
# #     # -- test allVars_log_adj_rent_per_sqm and log_adj_rent_per_sqm MAPE: 11.698 % | MAE: 15085.0 | R2: 0.804

# # ###--- shinjuku holdout
# #     # -- train allVars_log_adj_rent_per_sqm_Shinjuku and log_adj_rent_per_sqm MAPE: 11.672 % | MAE: 13192.0 | R2: 0.834
# #     # -- test allVars_log_adj_rent_per_sqm_Shinjuku and log_adj_rent_per_sqm MAPE: 10.655 % | MAE: 16963.0 | R2: 0.84

# # ###--- suginami holdout
# #     # -- train allVars_log_adj_rent_per_sqm_Suginami and log_adj_rent_per_sqm MAPE: 11.674 % | MAE: 13432.0 | R2: 0.839
# #     # -- test allVars_log_adj_rent_per_sqm_Suginami and log_adj_rent_per_sqm MAPE: 10.518 % | MAE: 11125.0 | R2: 0.81

# ###--- adding squared room size
#     # -- train allVars_log_adj_rent_per_sqm_lastMonth_sqrd and log_adj_rent_per_sqm MAPE: 11.031 % | MAE: 12581.0 | R2: 0.872
#     # -- test allVars_log_adj_rent_per_sqm_lastMonth_sqrd and log_adj_rent_per_sqm MAPE: 11.21 % | MAE: 14411.0 | R2: 0.843




# ###--------------------------------------------------------------------------------










# # ###==============================================================================================
# # ###================================== FIRST NEURAL NETWORK MODEL ================================
# # ###==============================================================================================

# def revertPrice(thePrices, priceType, surfaceAreas):
#     thePrices = thePrices.detach().numpy()
#     if priceType == 'log_adj_rent_per_sqm':
#         revertedPrices = np.multiply(np.exp(thePrices), surfaceAreas)
#     elif priceType == 'adj_rent_per_sqm':
#         revertedPrices = np.multiply(thePrices, surfaceAreas)
#     elif priceType == 'log_adj_rent':
#         revertedPrices = np.exp(thePrices)
#     else:
#         revertedPrices = thePrices
#     return revertedPrices
#     # return torch.tensor(revertedPrices, dtype=torch.float32).reshape(-1, 1)

# # class RentalPriceModel(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.dropout = nn.Dropout(p=0.2)
# #         self.hidden1 = nn.Linear(len(allVars), 160)
# #         self.act1 = nn.LeakyReLU()
# #         self.hidden2 = nn.Linear(160, 80)
# #         self.act2 = nn.LeakyReLU()
# #         self.hidden3 = nn.Linear(80, 20)
# #         self.act3 = nn.ReLU()
# #         self.output = nn.Linear(20, 1)

# #     def forward(self, x):
# #         x = self.act1(self.hidden1(x))
# #         x = self.dropout(x)
# #         x = self.act2(self.hidden2(x))
# #         x = self.dropout(x)
# #         x = self.act3(self.hidden3(x))
# #         x = self.output(x)
# #         return x


# class RentalPriceModel(nn.Module):

#     def __init__(self):
#         # Call parent contructor
#         super().__init__()
#         # self.relu = nn.ReLU()
#         # self.leakyRelu = nn.LeakyReLU()
#         # self.swish = nn.SiLU()

#         ##--- models v06
#         self.dropout = nn.Dropout(p=0.2)
#         self.linear1 = nn.Linear(len(allVars), 120)
#         self.act1 = nn.SiLU()
#         self.linear2 = nn.Linear(120, 60)
#         self.act2 = nn.SiLU()
#         self.linear3 = nn.Linear(60, 30)
#         self.act3 = nn.SiLU()
#         self.linear4 = nn.Linear(30, 15)
#         self.act4 = nn.SiLU()
#         self.output = nn.Linear(15, 1)


#     def forward(self, netModel):
#         netModel = self.act1(self.linear1(netModel))
#         netModel = self.dropout(netModel)
#         netModel = self.act2(self.linear2(netModel))
#         netModel = self.act3(self.linear3(netModel))
#         netModel = self.act4(self.linear4(netModel))
#         netModel = self.output(netModel)
#         return netModel


#     #     ###--- model v04
#     #     self.dropout = nn.Dropout(p=0.2)
#     #     self.linear1 = nn.Linear(len(allVars),120)
#     #     self.act1 = nn.SiLU()
#     #     self.linear2 = nn.Linear(120, 120)
#     #     self.act2 = nn.SiLU()
#     #     self.linear3 = nn.Linear(120, 60)
#     #     self.act3 = nn.SiLU()
#     #     self.output = nn.Linear(60, 1)

#     # def forward(self, netModel):
#     #     netModel = self.act1(self.linear1(netModel))
#     #     netModel = self.dropout(netModel)
#     #     netModel = self.act2(self.linear2(netModel))
#     #     netModel = self.act3(self.linear3(netModel))
#     #     netModel = self.output(netModel)
#     #     return netModel

#         # ###--- model v03
#         # self.dropout = nn.Dropout(p=0.2)
#         # self.linear1 = nn.Linear(len(allVars), 60)
#         # self.linear2 = nn.Linear(60, 20)
#         # self.output = nn.Linear(20, 1)

#         ###--- models v01 and v02
#         # self.dropout = nn.Dropout(p=0.2)
#         # self.linear1 = nn.Linear(len(allVars), 160)
#         # self.linear2 = nn.Linear(160, 80)
#         # self.linear3 = nn.Linear(80, 20)
#         # self.output = nn.Linear(20, 1)


#     # def forward(self, netModel):
#     #     netModel = self.linear1(netModel)
#     #     # netModel = self.leakyRelu(netModel)
#     #     netModel = self.swish(netModel)
#     #     netModel = self.dropout(netModel)
#     #     netModel = self.linear2(netModel)
#     #     # netModel = self.relu(netModel)
#     #     netModel = self.swish(netModel)
#     #     # netModel = self.leakyRelu(netModel)
#     #     netModel = self.dropout(netModel)
#     #     # netModel = self.linear3(netModel)
#     #     # netModel = self.relu(netModel)
#     #     netModel = self.output(netModel)
#         # return netModel



# print("== Preparing Room Data.")
# joinedData = readGeoPandasCSV('../Data/ScoringData/refinedData.csv').fillna(np.nan)

# # joinedData['updated_at'] = pd.to_datetime(list(joinedData['updated_at']))
# # # print(type(dateData[0]))
# # joinedData = joinedData.sort_values(by='dttime')

# allVars = unitVars + wideVars + accVars + econVars + bldVars + storeVars + iyaVars + emabassyVars + zoneTypeVars + landUseVars + vegVars + hazVars + popPercentVars  ##-- 79
# # allVars = unitVars + wideVars  ##-- 8
# print("  -- Number of explanatory variables:", len(allVars))

# # standardizeVars = True
# standardizeVars = False

# if standardizeVars == True:
#     print("== Standardizing the variables.")
#     for var in allVars:
#         joinedData[var] = standardizeVariable(joinedData[var])

# print("== Splitting train and test sets.")

# ###=== Use the last month as test data
# # holdout = "lastMonth"
# # dates = pd.to_datetime(joinedData["updated_at"])
# # one_month_ago = dates.max() - pd.Timedelta(days=30)
# # is_test = dates > one_month_ago
# # trainData = joinedData[~is_test]
# # testData = joinedData[is_test]

# ###=== Use different areas as test data
# cityData = readGeoPandasCSV('../Data/GISandAddressData/adminAreaPoly_cities-AllJapan_simp.csv')
# # print(cityData.head())
# # areaName = "新宿"
# # holdout = "Shinjuku"
# areaName = "杉並"
# holdout = "Suginami"
# cityPolygon = list(cityData[((cityData['prefName'] == "東京都") & (cityData['cityName'].str.contains(areaName) ))]['geometry'])[0]
# # print(cityPolygon)
# is_test = joinedData['geometry'].intersects(cityPolygon)

# ###-----------------
# trainData = joinedData[~is_test]
# testData = joinedData[is_test]
# print("  -- Number of train data samples:", len(trainData))  ##-- 745,875   suginami => 765,347   shinjuku => 753,091
# print("  -- Number of test data samples:", len(testData))    ##--  47,500   suginami =>  28,028   shinjuku =>  40,284

# # print(trainData.head())

# priceVars = ['log_adj_rent_per_sqm', 'adj_rent_per_sqm', 'log_adj_rent', 'adj_rent']
# priceLabels = ['log adjusted rent/m$^2$', 'adjusted rent/m$^2$', 'log adjusted rent', 'adjusted rent']
# priceTitles = ['Log Adjusted Rent/m$^2$', 'Adjusted Rent/m$^2$', 'Log Adjusted Rent', 'Adjusted Rent']

# # learningRates = [0.01, 0.001, 0.0001, 0.00005, 0.00001]
# learningRates = [0.0001]

# # optimizers = ["adam", "SGD"]
# optimizers = ["adam"]

# # lossFunctions = ["MSE", "L1", "SmoothL1", "Huber"]
# lossFunctions = ["MSE"]

# y_true = torch.tensor(trainData['adj_rent'].to_numpy(), dtype=torch.float32).reshape(-1, 1)
# y_test_true = torch.tensor(testData['adj_rent'].to_numpy(), dtype=torch.float32).reshape(-1, 1)
# train_surfaces = trainData['surface_area'].to_numpy().reshape(-1, 1)
# test_surfaces = testData['surface_area'].to_numpy().reshape(-1, 1)

# versionNum = 6
# versionNum = str(versionNum) if versionNum > 9 else "0" + str(versionNum)

# print("== Training the NN model.")
# n_epochs = 6000
# batch_size = 5000
# ###-----------------------
# for priceNum,thisPrice in enumerate(priceVars[:1]):

#     print("--------------------------------------------------------")
#     # print("  -- Running NN for", thisPrice)
#     X = torch.tensor(trainData[allVars].to_numpy(), dtype=torch.float32)
#     y = torch.tensor(trainData[thisPrice].to_numpy(), dtype=torch.float32).reshape(-1, 1)

#     ###-----------------------
#     for learningRate in learningRates:

#         ###-----------------------
#         for theOptimizer in optimizers:

#             ###-----------------------
#             for theLoss in lossFunctions:

#                 ###---- run the model
#                 runTime = time.time()

#                 model = RentalPriceModel()

#                 if theLoss == "MSE":
#                     loss_fn = nn.MSELoss()
#                 elif theLoss == "L1":
#                     loss_fn = nn.L1Loss()
#                 elif theLoss == "SmoothL1":
#                     loss_fn = nn.SmoothL1Loss()
#                 elif theLoss == "Huber":
#                     loss_fn = nn.HuberLoss()
#                 else:
#                     loss_fn = nn.MSELoss()


#                 if theOptimizer == "adam":
#                     optimizer = optim.Adam(model.parameters(), lr=learningRate)
#                 elif theOptimizer == "SGD":
#                     optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.8)
#                 else:
#                     optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.8)



#                 ###-----------------------
#                 modelName = 'NNmodel_allVars_v'+versionNum+'_'+holdout+"_"+thisPrice+'_opt='+theOptimizer+'_lr='+str(learningRate)+"_"+theLoss
#                 if standardizeVars == True:
#                     modelName + "_stdized"

#                 print("  -- Running NN for", thisPrice, " | Optimizer =", theOptimizer, " | learning rate =", learningRate, " | loss =", theLoss)

#                 ###--- Run with batches
#                 for epoch in range(n_epochs):
#                     for i in range(0, len(X), batch_size):
#                         Xbatch = X[i:i+batch_size]
#                         ybatch = y[i:i+batch_size]
#                         y_pred = model(Xbatch)
#                         loss = loss_fn(y_pred, ybatch)
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()

#                 ###--- Run without batches
#                 # for epoch in range(n_epochs):
#                 #     y_pred = model(X)
#                 #     loss = loss_fn(y_pred, y)
#                 #     optimizer.zero_grad()
#                 #     loss.backward()
#                 #     optimizer.step()

#                     ###--- Evaluate progress across epochs
#                     if (epoch % 100 == 0):
#                         y_pred = model(X)
#                         adj_rent_pred = revertPrice(y_pred, thisPrice, train_surfaces)

#                         print(f'    -- Finished epoch {epoch:0>4} | loss', rnd(loss.detach().numpy(),10), '| MAPE:', MAPE(y_true, adj_rent_pred), "| MAE:", MAE(y_true, adj_rent_pred, 0) , "| R2:", getR2(y_true, adj_rent_pred) )

#                 print("  -- Saving NN model")
#                 torch.save(model.state_dict(), '../Demand Estimation/NeuralNets/'+modelName+'.pt')

#                 ###----------- results --------------
#                 print("== Testing the NN model.")
#                 X_test = torch.tensor(testData[allVars].to_numpy(), dtype=torch.float32)
#                 y_test = torch.tensor(testData[thisPrice].to_numpy(), dtype=torch.float32).reshape(-1, 1)

#                 # compute accuracy (no_grad is optional)
#                 with torch.no_grad():
#                     y_pred = model(X)
#                     test_y_pred = model(X_test)

#                 adj_rent_pred = revertPrice(y_pred, thisPrice, train_surfaces)
#                 adj_rent_pred_test = revertPrice(test_y_pred, thisPrice, test_surfaces)

#                 print("  -- Results for", 'v'+versionNum,'|', thisPrice, "| Optimizer =", theOptimizer, "| learning rate =", learningRate, " | loss =", theLoss)
#                 print(f"    -- Training data results | MAPE:", MAPE(y_true, adj_rent_pred), "% | MAE:", MAE(y_true, adj_rent_pred, 0) , "| R2:", getR2(y_true, adj_rent_pred) )
#                 print(f"    -- Test data accuracy | MAPE:", MAPE(y_test_true, adj_rent_pred_test), "% | MAE:", MAE(y_test_true, adj_rent_pred_test, 0), "| R2:", getR2(y_test_true, adj_rent_pred_test) )

#                 ###------------ plot ------------
#                 xData = list(np.array(adj_rent_pred_test).reshape(-1, 1).flatten())
#                 yData = list(np.array(y_test_true).reshape(-1, 1).flatten())

#                 makeDensityPlot(xData, yData, "test data predicted rent", "test data actual rent", titleText='Accuracy of NN Model', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=True, bestFit=[1,2,3], diagonalLine=True, plotRange=None, addMargin=False, filename='../Demand Estimation/NeuralNets/'+modelName+'_predAccuracy.png')

#                 reportRunTime(runTime)




# # == Testing the NN model.

# # -- Running NN for v01 | log_adj_rent_per_sqm  | Optimizer = SGD  | learning rate = 0.0001
# #   -- Training data results | MAPE: 11.771 % | MAE: 13043.0 | R2: 0.869
# #   -- Test data accuracy | MAPE: 11.75 % | MAE: 14519.0 | R2: 0.846

# # -- Running NN for v01 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 9.044 % | MAE: 9841.0 | R2: 0.924
# #   -- Test data accuracy | MAPE: 8.686 % | MAE: 10312.0 | R2: 0.921

# # -- Running NN for v02 | adj_rent_per_sqm  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 11.616 % | MAE: 12460.0 | R2: 0.895
# #   -- Test data accuracy | MAPE: 11.31 % | MAE: 13378.0 | R2: 0.884

# # -- Running NN for v02 | log_adj_rent  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 9.046 % | MAE: 10079.0 | R2: 0.915
# #   -- Test data accuracy | MAPE: 8.825 % | MAE: 10676.0 | R2: 0.913

# # -- Running NN for v02 | adj_rent  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 12.634 % | MAE: 13435.0 | R2: 0.887
# #   -- Test data accuracy | MAPE: 12.498 % | MAE: 14488.0 | R2: 0.88

# # -- Running NN for v03 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 5e-05
# #   -- Training data results | MAPE: 9.038 % | MAE: 9870.0 | R2: 0.924
# #   -- Test data accuracy | MAPE: 8.834 % | MAE: 10613.0 | R2: 0.916

# # -- Running NN for v03 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 1e-05
# #   -- Training data results | MAPE: 8.663 % | MAE: 9512.0 | R2: 0.926
# #   -- Test data accuracy | MAPE: 8.749 % | MAE: 10640.0 | R2: 0.915

# # -- Conintuing NN v04 | for log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 1e-05
# #   -- Training data results | MAPE: 8.61 % | MAE: 9449.0 | R2: 0.928
# #   -- Test data accuracy | MAPE: 8.749 % | MAE: 10637.0 | R2: 0.914

# # -- Results for v04 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 10.525 % | MAE: 11569.0 | R2: 0.897
# #   -- Test data accuracy | MAPE: 10.301 % | MAE: 12423.0 | R2: 0.893
# #   -- runtime: 1.0 hour(s) 36.0 minutes

# # -- Results for v04 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 9.412 % | MAE: 10223.0 | R2: 0.921
# #   -- Test data accuracy | MAPE: 8.881 % | MAE: 10518.0 | R2: 0.92
# #   -- runtime: 3.0 hour(s) 59.0 minutes

# # -- Results for v05 | log_adj_rent_per_sqm  | Optimizer = adam  | learning rate = 0.0001
# #   -- Training data results | MAPE: 9.569 % | MAE: 10375.0 | R2: 0.919
# #   -- Test data accuracy | MAPE: 9.077 % | MAE: 10748.0 | R2: 0.917
# #   -- runtime: 2.0 hour(s) 52.0 minutes

# # -- Results for v06 | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = MSE
# #   -- Training data results | MAPE: 8.464 % | MAE: 9315.0 | R2: 0.93  ##-- running output MAPE: 9.416, but then this number?
# #   -- Test data accuracy | MAPE: 8.619 % | MAE: 10512.0 | R2: 0.916
# #   -- runtime: 7.0 hour(s) 1.0 minutes

# # -- Results for v06 | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = L1
# #   -- Training data results | MAPE: 9.152 % | MAE: 10016.0 | R2: 0.922
# #   -- Test data accuracy | MAPE: 8.664 % | MAE: 10310.0 | R2: 0.92
# #   -- runtime: 5.0 hour(s) 38.0 minutes

# #   -- Results for v06 | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = SmoothL1
# #     -- Training data results | MAPE: 9.456 % | MAE: 10263.0 | R2: 0.922
# #     -- Test data accuracy | MAPE: 8.81 % | MAE: 10361.0 | R2: 0.923
# #     -- runtime: 5.0 hour(s) 25.0 minutes

# #   -- Results for v06 | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = Huber
# #     -- Training data results | MAPE: 9.464 % | MAE: 10298.0 | R2: 0.921
# #     -- Test data accuracy | MAPE: 8.787 % | MAE: 10343.0 | R2: 0.923
# #     -- runtime: 5.0 hour(s) 28.0 minutes

  # -- Results for v06 Shinjuku | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = MSE
  #   -- Training data results | MAPE: 9.24 % | MAE: 9907.0 | R2: 0.923
  #   -- Test data accuracy | MAPE: 9.975 % | MAE: 14725.0 | R2: 0.903
  #   -- runtime: 6.0 hour(s) 46.0 minutes

  # -- Results for v06 Suginami | log_adj_rent_per_sqm | Optimizer = adam | learning rate = 0.0001  | loss = MSE
  #   -- Training data results | MAPE: 9.429 % | MAE: 10295.0 | R2: 0.923
  #   -- Test data accuracy | MAPE: 8.612 % | MAE: 8997.0 | R2: 0.869
  #   -- runtime: 7.0 hour(s) 15.0 minutes



# ###==========================================================================














# ###======================== LOAD NN ==========================
# print("  -- Loading NN model")
# thePath = '../Demand Estimation/NeuralNets/NNmodel_allVars_v06_log_adj_rent_per_sqm_opt=adam_lr=0.0001_MSE.pt.pt'
# model = RentalPriceModel(nn.Module)
# model.load_state_dict(torch.load(thePath))
# model.eval()

# makeDensityPlot(xData, yData, "test data predicted rent", "test data actual rent", titleText='Accuracy of NN Model', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=True, bestFit=[1,2,3], diagonalLine=True, plotRange=None, addMargin=False, filename='../Demand Estimation/NeuralNets/'+modelName+'_predAccuracy.png')
# ###==========================================================================















# ###======================== LATEX TABLE ==========================
# ##=== Convert the feature importance table information fromMita to LaTeX
# cityData = readCSV('../Demand Estimation/LGBM Results/changes_of_permuted_prediction_lgbm_comparison.csv')
# cityData = cityData.astype('str')
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace("_"," "))
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace(" normed",""))
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace("econ demand sCurve-90","econ est demand"))
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace("unit floor plan number of rooms","unit number of rooms"))
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace("bld percent building surface area","bld building surface area"))
# cityData['feature'] = cityData['feature'].apply(lambda val: val.replace("zone type=residential class1 lowrise","zone type=res cls1 lowrise"))
# print(cityData.to_latex(index=False))

# 1 & unit surface area & 20.459 &  24,938  \\
# 2 & econ est demand 10 & 11.013 &  11,966  \\
# 3 & unit built year & 8.111 &  8,718  \\
# 4 & unit age in months & 7.830 &  8,577  \\
# 5 & closest station id & 6.004 &  6,669  \\
# 6 & unit building floors & 5.535 &  6,869  \\
# 7 & unit number of rooms & 5.445 &  5,394  \\
# 8 & unit room floor & 3.195 &  3,715  \\
# 9 & econ num companies & 2.990 &  3,463  \\
# 10 & econ num jobs & 2.656 &  3,421  \\
# 11 & unit lon & 2.214 &  2,153  \\
# 12 & embassy & 2.027 &  3,229  \\
# 13 & unit lat & 1.876 &  2,049  \\
# 14 & embassy log & 1.498 &  2,567  \\
# 15 & veg=field & 1.325 &  1,569  \\
# 16 & acc closest station time & 1.125 &  1,182  \\
# 17 & bld building surface area & 1.125 &  1,184  \\
# 18 & land use=low-rise sparse & 1.118 &  1,198  \\
# 19 & pop percent 30 44yr & 1.064 &  1,121  \\
# 20 & pop percent house & 1.030 &  1,048  \\


# rank & feature & MAPE & MAE \\
# 1 & unit surface area & 20.512 &  25,265  \\
# 2 & econ est demand 10 & 12.568 &  13,257  \\
# 3 & unit age in months & 7.998 &  8,732  \\
# 4 & unit built year & 7.966 &  8,533  \\
# 5 & unit building floors & 5.623 &  6,959  \\
# 6 & unit number of rooms & 5.129 &  5,134  \\
# 7 & econ num jobs & 3.759 &  4,512  \\
# 8 & unit room floor & 3.234 &  3,726  \\
# 9 & bld building surface area & 2.395 &  2,404  \\
# 10 & embassy log & 2.360 &  3,948  \\
# 11 & acc closest station time & 1.980 &  2,013  \\
# 12 & veg=field & 1.932 &  2,426  \\
# 13 & pop percent female & 1.790 &  1,998  \\
# 14 & econ num companies & 1.719 &  1,988  \\
# 15 & pop percent house & 1.487 &  1,525  \\
# 16 & land use=low-rise sparse & 1.460 &  1,608  \\
# 17 & embassy & 1.441 &  2,373  \\
# 18 & zone type=res cls1 lowrise & 1.438 &  1,637  \\
# 19 & acc station area score & 1.353 &  1,453  \\
# 20 & zone type=commercial & 1.350 &  1,468  \\

###==========================================================================




# # ###======================== LGBM PLOT ==========================
# ###=== Make a prediction accuracy plot for the LGBM results
# thisData = readCSV('../Demand Estimation/LGBM Results/kitchen_sink-wo-latlon-station_id_test_predict_.csv')
# print(thisData.head())
# y_pred = list(thisData['y_pred'])
# adj_rent = list(thisData['adj_rent'])
# thisLabel = "LGBM_spatial"

# makeDensityPlot(y_pred, adj_rent, "test data predicted rent", "test data actual rent", titleText='Accuracy of LGBM Spatial Model', numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=True, bestFit=[1,2,3], diagonalLine=True, plotRange=None, addMargin=False, filename='../Demand Estimation/LGBM Results/scatter_predictions_'+thisLabel+'.png')
# ###==========================================================================






# ###======================== EXPLORE SURFACE AREA ==========================
# roomData = readGeoPandasCSV('../Data/ScoringData/roomData.csv').fillna(np.nan)
# # print(list(roomData))  ['id', 'story_id', 'updated_at', 'hex_id', 'geometry', 'log_adj_rent_per_sqm', 'adj_rent_per_sqm', 'adj_rent', 'surface_area', 'lat', 'lon', 'built_year', 'age_in_months', 'building_floors', 'room_floor', 'layout', 'floor_plan_type', 'floor_plan_number_of_rooms']

# # print(len(roomData))  ##-- 798563
# roomData = roomData[(roomData['adj_rent_per_sqm'] <= 20000)].copy()  ##-- filter out some outliers


# centralArea = bufferGeometry(Point(139.73634781089083, 35.6722465355721), 8000)  ##-- around Akasaka
# isCenter = roomData['geometry'].intersects(centralArea)
# centralData = roomData[isCenter].copy()
# suburbData = roomData[~ isCenter].copy()

# # print(len(roomData[(roomData['surface_area'] > 80)]))  ##-- 17471
# # print(len(roomData[(roomData['surface_area'] <= 80)]))  ##-- 780989

# xVar = 'surface_area'
# yVar = 'adj_rent_per_sqm'




# ##------------------------------
# customAll = roomData[((roomData['surface_area'] > 24) & (roomData['surface_area'] <= 27))].copy()
# customCenter = customAll[isCenter].copy()
# customSuburb = customAll[~ isCenter].copy()

# xData = list(customAll[xVar])
# yData = list(customAll[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + '24-27_All'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(customCenter[xVar])
# yData = list(customCenter[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + '24-27_Center'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(customSuburb[xVar])
# yData = list(customSuburb[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + '24-27_Suburb'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')




# ##------------------------------
# filteredData = roomData[((centralData['age_in_months'] >= 120) & (roomData['age_in_months'] < 156))].copy()
# filteredData = filteredData[(filteredData['surface_area'] <= 40)].copy()
# filteredCenter = filteredData[isCenter].copy()
# filteredSuburb = filteredData[~ isCenter].copy()  ##-- EMPTY!!!

# print(len(filteredCenter))

# xData = list(filteredCenter[xVar])
# yData = list(filteredCenter[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'filtered_under40_Center'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(filteredSuburb[xVar])
# yData = list(filteredSuburb[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'filtered_under40_Suburb'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')









###------------------------------
# under40all = roomData[(roomData['surface_area'] <= 40)].copy()
# under40Center = centralData[(centralData['surface_area'] <= 40)].copy()
# under40Suburb = suburbData[(suburbData['surface_area'] <= 40)].copy()

# # xData = list(under50all[xVar])
# # yData = list(under50all[yVar])
# # thisLabel = xVar + '_vs_' + yVar + '_' + 'over50_All'
# # makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(under40Center[xVar])
# yData = list(under40Center[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'under40_Center'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(under40Suburb[xVar])
# yData = list(under40Suburb[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'under40_Suburb'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')







###----------------------------
# over80all = roomData[(roomData['surface_area'] > 80)].copy()
# over80Center = centralData[(centralData['surface_area'] > 80)].copy()
# over80Suburb = suburbData[(suburbData['surface_area'] > 80)].copy()

# xData = list(over80all[xVar])
# yData = list(over80all[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'over80_All'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(over80Center[xVar])
# yData = list(over80Center[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'over80_Center'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(over80Suburb[xVar])
# yData = list(over80Suburb[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'over80_Suburb'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')


# under80all = roomData[(roomData['surface_area'] <= 80)].copy()
# under80Center = centralData[(centralData['surface_area'] <= 80)].copy()
# under80Suburb = suburbData[(suburbData['surface_area'] <= 80)].copy()

# xData = list(under80all[xVar])
# yData = list(under80all[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'under80_All'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(under80Center[xVar])
# yData = list(under80Center[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'under80_Center'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# xData = list(under80Suburb[xVar])
# yData = list(under80Suburb[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'under80_Suburb'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')







# xData = list(roomData[xVar])
# yData = list(roomData[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'allData'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# smallerRooms = roomData[roomData['surface_area'] < 30].copy()
# xData = list(smallerRooms[xVar])
# yData = list(smallerRooms[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'rooms_u30'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# smallRooms = roomData[roomData['surface_area'] < 20].copy()
# xData = list(smallRooms[xVar])
# yData = list(smallRooms[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'rooms_u20'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# medRooms = roomData[((roomData['surface_area'] >= 20) & (roomData['surface_area'] < 25))].copy()
# xData = list(medRooms[xVar])
# yData = list(medRooms[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'rooms_20-25'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# lgRooms = roomData[((roomData['surface_area'] >= 25) & (roomData['surface_area'] < 30))].copy()
# xData = list(lgRooms[xVar])
# yData = list(lgRooms[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'rooms_25-30'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# ###---------------------------
# xData = list(centralData[xVar])
# yData = list(centralData[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'centralArea'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# smallerCentral = centralData[centralData['surface_area'] < 30].copy()
# xData = list(smallerCentral[xVar])
# yData = list(smallerCentral[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'central_u30'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# smallCentral = centralData[centralData['surface_area'] < 20].copy()
# xData = list(smallCentral[xVar])
# yData = list(smallCentral[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'central_u20'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# medCentral = centralData[((centralData['surface_area'] >= 20) & (centralData['surface_area'] < 25))].copy()
# xData = list(medCentral[xVar])
# yData = list(medCentral[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'central_20-25'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')

# lgCentral = centralData[((centralData['surface_area'] >= 25) & (centralData['surface_area'] < 30))].copy()
# xData = list(lgCentral[xVar])
# yData = list(lgCentral[yVar])
# thisLabel = xVar + '_vs_' + yVar + '_' + 'central_25-30'
# makeDensityPlot(xData, yData, xVar, yVar, titleText=thisLabel, numBins=200, maxCount=None, colorScheme=None, figSize=5, scaledAxes=False, bestFit=[1,2,3], diagonalLine=False, plotRange=None, addMargin=False, filename='../Demand Estimation/Post Analysis/scatter_'+thisLabel+'.png')














###======================================== END OF FILE ===========================================
