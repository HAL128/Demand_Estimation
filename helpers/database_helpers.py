# -*- coding: utf-8 -*-



# for writing
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.types import String
from geoalchemy2 import Geometry

# for reading
import psycopg2 as pg
import pandas as pd
import geopandas as gpd
from psycopg2.extras import RealDictCursor
from shapely.geometry import Polygon
from shapely import wkt
from shapely.wkb import loads as wkbloads


# commandline database access
# psql --host=postgres-postgis.chybxyoboolv.ap-northeast-1.rds.amazonaws.com --port=5432 --username=postgres --password --dbname=data_warehouse


###===> dbConnection = DatabaseConnInfo(username='data_warehouse_owner', password='3x4mp13us3r')
###===> osmConn = DatabaseConnInfo(username='postgres', password='t4ng1b131mp4cT', database='osm_data')


###============================================================================


###=== Create object to store batabase connection variables for easier reference
class DatabaseConnInfo:
    def __init__(self, username, password, database='data_warehouse', host="postgres-postgis.chybxyoboolv.ap-northeast-1.rds.amazonaws.com", port='5432'):
        self.host = host
        self.port = port
        self.dbname = database
        self.user = username
        self.pswd = password


###=== To use this, you create the connection object in your python code, like this:
# dbConnection = DatabaseConnInfo(database='example_DB_name', username='example_DB_user', password='3x4mp13us3r')
###--- then you can get data with things like:
# theData = get_data_for_geom(dbConnection, table='my_data', geom=thisArea)








###============================================================================
###============================== HELPER FUNCTIONS ============================
###============================================================================

###--- when returning tables from the DB, convert geometries with other names to geometry type
def loadOtherGeomsCols(thisData, geomCols=[]):

    if len(geomCols) > 0:
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















###============================================================================
###=========================== WRITE DATA TO TABLE ============================
###============================================================================

###=== We use sqlalchemy for writing
def write_geodataframe_to_database(dbConn, table:str, geodataframe:gpd.GeoDataFrame):

    engine = create_engine(
        URL.create(
            "postgresql",
            database=dbConn.dbname,
            username=dbConn.user,
            password=dbConn.pswd,
            port=dbConn.port,
            host=dbConn.host
        )
    )
    ###--- If the table exists, but is missing columns from the current dataframe, instead of failing, just add the column
    if table in list_tables(dbConn):
        existingCols = get_column_names(dbConn, table=table)
        thisDFcols = list(geodataframe)
        missingCols = [x for x in thisDFcols if x not in existingCols]
        if len(missingCols) > 0:
            dataTypes = geodataframe.dtypes.astype(str).to_dict()  ##-- a dictionary for variables to the pandas type
            colNameTypeDict = {col:dataTypes[col] for col in missingCols}
            dataTypeTranslator = {'str':'VARCHAR', 'int':'BIGINT', 'Int64':'BIGINT', 'Int32':'BIGINT', 'int32':'BIGINT', 'int64':'BIGINT', 'float':'FLOAT8', 'float64':'FLOAT8', 'bool':'BOOLEAN', 'object':'VARCHAR', 'geometry':'GEOMETRY'}
            addColumns = ', '.join(['ADD COLUMN '+'"'+col+'"'+' '+dataTypeTranslator[thisType] for col,thisType in colNameTypeDict.items()])
            query1 = f"""
                ALTER TABLE "{table}"
                  {addColumns};
            """
            conn = pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd)
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query1)

    try:  ##-- first try with the current geometry type
        writing_result = geodataframe.to_postgis(
            name=table,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=400000,  ##-- up to that, write at once, othersize chunk it
            dtype={'geometry': Geometry('GEOMETRY', srid=4326), 'source':String(), 'target':String(), 'id':String() },  ##-- This doesn't work to cast the column type
        )
    except:   ##-- if it fails, then update the table to have a generic geometry type, and try again
        query2 = f"""
            ALTER TABLE "{table}"
               ALTER COLUMN geometry TYPE geometry(Geometry,4326);
        """
        conn = pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd)
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query2)

        writing_result = geodataframe.to_postgis(
            name=table,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=400000,  ##-- up to that, write at once, othersize chunk it
            dtype={'geometry': Geometry('GEOMETRY', srid=4326), 'source':String(), 'target':String(), 'id':String() },
        )
        ##-- if it fails for another reason, add more kludges as the needs arise.

    # return writing_result
###--------------------------------------------------------------------------


###=== add new data column to an existing table where the new data centroid is in the original geometry.
def add_data_to_table(dbConn, table:str, geodataframe:gpd.GeoDataFrame, colNameTypeDict:dict):

    dataTypeTranslator = {'str':'VARCHAR', 'int':'BIGINT', 'Int64':'BIGINT', 'Int32':'BIGINT', 'int32':'BIGINT', 'int64':'BIGINT', 'float':'FLOAT8', 'float64':'FLOAT8', 'bool':'BOOLEAN', 'object':'VARCHAR'}

    write_geodataframe_to_database(dbConn, table='temp_table', geodataframe=geodataframe)

    columns = [col for col,thisType in colNameTypeDict.items()]
    setStatement = ', '.join(['"'+col+'"'+' = t2.'+'"'+col+'"' for col in columns if col != 'level'])

    ###=== you can't add columns that already exist in the database, so check them first
    existingCols = get_column_names(dbConn, table=table)
    duplicateColumns = [k for k,v in colNameTypeDict.items() if k in existingCols]
    if len(duplicateColumns) > 0:
        print("!! The columns", duplicateColumns,"already exist in table", table, "and cannot be added.")
    colNameTypeDict = {k:v for k,v in colNameTypeDict.items() if k not in existingCols}
    addColumns = ', '.join(['ADD COLUMN '+'"'+col+'"'+' '+dataTypeTranslator[thisType] for col,thisType in colNameTypeDict.items()])
    columnsToAdd = len([k for k,v in colNameTypeDict.items()])

    query1 = f"""
        ALTER TABLE "{table}"
          {addColumns};
    """

    query2 = f"""
        UPDATE "{table}" t1
          SET {setStatement}
          FROM temp_table t2
            WHERE ST_Within(ST_PointOnSurface(t2.geometry), t1.geometry)
    """
    ###=== If there is a level column in both in the input and existing table, it automatically joins using the same level
    if (('level' in columns) & ('level' in existingCols)):
        query2 += """ AND t2.level = t1.level"""
    elif (('level' not in columns) & ('level' in existingCols)):
        print("!! Database table", table ,"is level-based, but there is no level column in the input data")
        return None
    elif (('level' in columns) & ('level' not in existingCols)):
        print("!! input data is level-based, but there is no 'level' column in table", table)
        return None

    query3 = f"""
        DROP TABLE temp_table
    """
    conn = pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd)

    if columnsToAdd > 0:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query1)

    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query2)

    ##-- remove the intermediatary table
    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query3)

    conn.close()
    # print("  -- Data", columns ,"added to database table", table)


###--- test the column listing function
# columns = ['poop', 'crap', 'unko']
# columnNames = ', '.join(['temp_table.'+'"'+col+'"' for col in columns])
# print(columnNames) # ==> temp_table."poop", temp_table."crap", temp_table."unko"  but is that even rightt?

# ###--- test the set statement function
# colNameTypeDict = {'poop':'str', 'crap':'str', 'unko':'int'}
# columnNames = ', '.join(['ADD COLUMN '+'"'+col+'"'+" "+dataTypeTranslator[thisType] for col,thisType in colNameTypeDict.items()])
# print(columnNames) # ==> temp_table."poop", temp_table."crap", temp_table."unko"  but is that even rightt?


###============================================================================
###======================== VARIOUS READING FUNCTIONS =========================
###============================================================================

###=== we use psycopg2 for reading because it's faster and we're not making use of geoalchemy2/sqlalchemy's ORM
###=== we might switch in the future though

# ###--- database reading and writing fucks up several data type, so fix them.
# def fixDataTypes(df):



#     return df
# ###--------------------------------------------------------------------------


##-- any geometry can be used, not just polygons
def get_entire_data_table(dbConn, table: str):

    query = f"""
        SELECT *
        FROM "{table}"
    """

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------



##-- any geometry can be used, not just polygons
def get_part_data_table(dbConn, table:str, startRow:int, numRows:int, orderBy:str):

    query = f"""
        SELECT *
        FROM "{table}"
        ORDER BY {orderBy}
        LIMIT {numRows} OFFSET {startRow};
    """

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------




###=== we use psycopg2 for reading because it's faster and we're not making use of geoalchemy2/sqlalchemy's ORM
###=== we might switch in the future though

##-- any geometry can be used, not just polygons
def get_data_for_geom(dbConn, table: str, geom: Polygon):

    # print('here!')

    intersects_string = (f"""WHERE ST_Intersects(ST_GeomFromText('{geom.wkt}',4326), geometry)""")
    query = f"""
        SELECT *
        FROM "{table}"
        {intersects_string}
    """
    # print('also here!')
    try:
        with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
            # print('connected!')
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                
                cur.execute(query)
                
                # print('executed!')
                data = list(cur.fetchall())

                gdf = gpd.GeoDataFrame(
                    crs="epsg:4326",
                    data=data,
                    geometry=[wkbloads(row["geometry"]) for row in data],
                )
    except Exception as e:
        gdf = loadOtherGeomsCols(gdf)
        print('exception', e)
    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------




def get_random_sample(dbConn, table:str, numberOfRows:int):

    totalNumRows = get_number_rows(dbConn, table=table)
    bufferedNumRows = int(numberOfRows*1.3)
    percentOfRows = 100 * (bufferedNumRows / totalNumRows)
    # print(percentOfRows)

    query = f"""
        SELECT *
        FROM "{table}"
        TABLESAMPLE SYSTEM ({percentOfRows}) REPEATABLE (123)
        LIMIT {numberOfRows}
    """

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------



###=== As above, this can be used to get all locations that match a certain value, or get data by location name.
###=== Feed in a dictionary of variable:value pairs and it returns only data matching that pattern
def get_data_for_values(dbConn, table:str, columnValDict:dict):

    matchList = []
    valIsList = False
    for col,val in columnValDict.items():
        if isinstance(val,list):
            matchList.append('"'+col+'"'+" IN "+"(" + ", ".join(["'"+v+"'" if isinstance(v,str) else str(val) for v in val]) + ")")
        elif isinstance(val,str):
            matchList.append('"'+col+'"'+" = "+"'"+val+"'")
        else:
            matchList.append('"'+col+'"'+" = "+str(val))

    selectedValues = ' and '.join(matchList)
    intersects_string = (f"""WHERE {selectedValues}""")
    query = f"""
        SELECT *
        FROM "{table}"
        {intersects_string}
    """

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------


###=== As above, this can be used to get all locations that match a certain value, or get data by location name.
###=== Feed in a dictionary of variable:value pairs and it returns only data matching that pattern
def get_data_for_geom_for_values(dbConn, table:str, geom:Polygon, columnValDict: dict):

    matchList = []
    valIsList = False
    for col,val in columnValDict.items():
        if isinstance(val,list):
            matchList.append('"'+col+'"'+" IN "+"(" + ", ".join(["'"+v+"'" if isinstance(v,str) else str(val) for v in val]) + ")")
        elif isinstance(val,str):
            matchList.append('"'+col+'"'+" = "+"'"+val+"'")
        else:
            matchList.append('"'+col+'"'+" = "+str(val))

    selectedValues = ' and '.join(matchList)
    intersects_string = (f"""WHERE {selectedValues}
        AND ST_Intersects(ST_GeomFromText('{geom.wkt}',4326), geometry)"""
    )
    query = f"""
        SELECT *
        FROM "{table}"
        {intersects_string}
    """

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf

# ###=== Test converting a dict of vars and vals to SQL text
# someDict = {'var1':'poop', 'var2':'crap', 'var3':1}
# # print(' and '.join([col+" = "+val for col,val in someDict.items()]))
# print(' and '.join(['"'+col+'"'+" = "+"'"+val+"'" if isinstance(val,str) else '"'+col+'"'+" = "+str(val) for col,val in someDict.items()]))
###--------------------------------------------------------------------------


###=== As above, but return only select columns
def get_columns_for_geom(dbConn, table:str, geom:Polygon, columns:list):

    if 'geometry' not in columns:
        columns = columns + ['geometry']
    intersects_string = (f"""WHERE ST_Intersects(ST_GeomFromText('{geom.wkt}',4326), geometry)""" )
    columnNames = ','.join(['"'+col+'"' for col in columns])
    query = f"""
        SELECT {columnNames}
        FROM "{table}"
        {intersects_string}
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    gdf = loadOtherGeomsCols(gdf)
    return gdf
###--------------------------------------------------------------------------


###=== As above, this can be used to get all locations that match a certain value, or get data by location name.
###=== Feed in a dictionary of variable:value pairs and it returns only data matching that pattern
def get_unique_values_for_column(dbConn, table:str, column:str):

    query = f"""
        SELECT
          DISTINCT ON ("{column}") "{column}"
          FROM "{table}";
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())
            headers = [row[column] for row in data]

    return headers
###--------------------------------------------------------------------------




# use this to see the geospatial area in which the table has data, without having to pull tonnes of data from the database
def identify_coverage(dbConn: DatabaseConnInfo, table: str) -> Polygon:
    """Return convex hull of 'geometry' column of the table

    Args:
        dbConn (DatabaseConnInfo): database connector object
        table (str): table name
    Returns:
        Polygon: Convex hull of table
    """
    query = f"""
    SELECT ST_ConvexHull(ST_Collect(geometry)) AS geometry FROM {table};
    """

    with pg.connect(
        host=dbConn.host,
        port=dbConn.port,
        database=dbConn.dbname,
        user=dbConn.user,
        password=dbConn.pswd,
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())
    conn.close()
    return wkbloads(data=data[0]["geometry"], hex=True)





###======================= BY LEVEL ==================================

def get_data_for_geom_by_level(dbConn, table:str, level:int, geom:Polygon):

    columnValDict = {column:value}
    gdf = get_data_for_geom_for_values(dbConn, table=table, geom=geom, columnValDict=columnValDict)
    return gdf
###--------------------------------------------------------------------------


###=== As above, but return only select columns
def get_columns_for_geom_by_level(dbConn, table:str, geom:Polygon, columns:list, level:int=4):

    intersects_string = (
        f"""WHERE level={level}
            AND ST_Intersects(ST_GeomFromText('{geom.wkt}',4326), geometry)"""
    )
    columnNames = ','.join(['"'+col+'"' for col in columns])
    query = f"""
        SELECT {columnNames}
        FROM "{table}"
        {intersects_string}
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())

            gdf = gpd.GeoDataFrame(
                crs="epsg:4326",
                data=data,
                geometry=[wkbloads(row["geometry"]) for row in data],
            )

    return gdf
###--------------------------------------------------------------------------


###=== As above, this can be used to get all locations that match a certain value, or get data by location name.
###=== Feed in a dictionary of variable:value pairs and it returns only data matching that pattern
def get_data_by_level_for_values(dbConn, table:str, columnValDict:dict, level:int=4):

    columnValDict['level'] = level
    gdf = get_data_for_values(dbConn, table=table, columnValDict=columnValDict)

    return gdf

# ###=== Test converting a dict of vars and vals to SQL text
# someDict = {'var1':'poop', 'var2':'crap', 'var3':1}
# # print(' and '.join([col+" = "+val for col,val in someDict.items()]))
# print(' and '.join(['"'+col+'"'+" = "+"'"+val+"'" if isinstance(val,str) else '"'+col+'"'+" = "+str(val) for col,val in someDict.items()]))
###--------------------------------------------------------------------------


###=== This can be used to get all locations that match a certain value, or get data by location name.
###=== For example, setting 'column'='prefName' and 'value'='東京都' returns all the data for that prefecture.
def get_data_by_level_for_value(dbConn, table:str, column:str, value:str, level:int=4):

    columnValDict = {column:value}
    return get_data_by_level_for_values(dbConn, table, level, columnValDict)

###--------------------------------------------------------------------------












###============================================================================
###======================== DATA MANAGEMENT FUNCTIONS =========================
###============================================================================


###=== If you need to remove some data from a table, adapt and use this:
###--- removes data from table where 'level' column is equal to the specified value
def remove_data_for_values(dbConn, table:str, column:str, val:str):

    columnValDict = {column:val}  ##-- what if there is an empty cell, NaN, None, etc.
    selectedValues = ' and '.join(['"'+col+'"'+" = "+"'"+val+"'" if isinstance(val,str) else '"'+col+'"'+" = "+str(val) for col,val in columnValDict.items()])
    query = f"""
        DELETE FROM "{table}" WHERE {selectedValues};
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###--------------------------------------------------------------------------


###=== If you need to remove some data from a table, adapt and use this:
###--- removes data from table where 'level' column is equal to the specified value
def drop_columns_from_table(dbConn, table:str, columns:list):

    columnNames = ', '.join(['DROP COLUMN '+'"'+col+'"' for col in columns])
    query = f"""
        ALTER TABLE "{table}"
          {columnNames};
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###--------------------------------------------------------------------------


###=== Find duplicate rows based on "column" and overwrite the table with the duplicates removed
def remove_duplicate_rows(dbConn, table:str, column:str):

    ###--- the general version doesn't work until I get all columns from table, but this works for the hex table
    query = f"""
        DELETE FROM
            "{table}" a
                USING "{table}" b
        WHERE
            a.ctid < b.ctid
            AND a."{column}" = b."{column}";
    """  ##-- this works well with the implicit ctid, so it is fast and general

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)


###=== Find duplicate rows based on "column" and overwrite the table with the duplicates removed
def remove_duplicate_nodes(dbConn, table:str, column='id'):

    ###--- the general version doesn't work until I get all columns from table, but this works for the hex table
    query = f"""
        DELETE FROM
            "{table}" a
                USING "{table}" b
        WHERE
            a.ctid < b.ctid
            AND a."{column}" = b."{column}";
    """  ##-- this works well with the implicit ctid, so it is fast and general

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###=== Find duplicate rows based on "column" and overwrite the table with the duplicates removed
def remove_duplicate_edges(dbConn, table:str, column='id'):

    ###--- the general version doesn't work until I get all columns from table, but this works for the hex table
    query = f"""
        DELETE FROM
            "{table}" a
                USING "{table}" b
        WHERE
            a.ctid < b.ctid
            AND a.source = b.source
            AND a.target = b.target;
    """  ##-- this works well with the implicit ctid, so it is fast and general

    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###--------------------------------------------------------------------------


###=== change the name of a table that exists in the database
def duplicate_table(dbConn, table:str, new_name:str):

    query = f"""
        CREATE TABLE tempTable (LIKE "{table}");
        INSERT INTO tempTable
          ALTER TABLE tempTable
          RENAME TO "{new_name}";
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###--------------------------------------------------------------------------


###=== change the name of a table that exists in the database
def rename_table(dbConn, table:str, new_name:str):

    query = f"""
        ALTER TABLE "{table}"
          RENAME TO "{new_name}";
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)

###--------------------------------------------------------------------------


###=== if the table exists, completely detele it
def delete_table(dbConn, table:str):

    if table in list_tables(dbConn):
        query = f"""
            DROP TABLE "{table}"
        """
        with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)

###--------------------------------------------------------------------------


###=== lists all the tables in the given database
def list_tables(dbConn):

    query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = cur.fetchall()
    return list(pd.DataFrame(data)['table_name'])
###--------------------------------------------------------------------------


###=== returns column headers (with datatypes) of given table
def describe_table(dbConn, table:str):

    query = f"""
        SELECT
            table_name,
            column_name,
            data_type
        FROM
            information_schema.columns
        WHERE
            table_name = '{table}';
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = cur.fetchall()
    return pd.DataFrame(data)
###--------------------------------------------------------------------------



###=== returns column headers (with datatypes) of given table
def get_column_names(dbConn, table:str):

    query = f"""
        SELECT * FROM "{table}" LIMIT 0
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = cur.description
    return [desc[0] for desc in data]
###--------------------------------------------------------------------------




###=== returns the unique values in a column of a table
def get_unique_values(dbConn, table:str, column:str):

    query = f"""
        SELECT DISTINCT ON ({table}."{column}") {table}."{column}" FROM "{table}"
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())
            # df = pd.DataFrame(data=data)

    return [dict(row)[column] for row in data]
###--------------------------------------------------------------------------



###=== returns the exact number of rows in a table
def get_number_rows(dbConn, table:str):

    query = f"""
        SELECT count(*) FROM "{table}"
    """
    with pg.connect(host=dbConn.host, port=dbConn.port, database=dbConn.dbname, user=dbConn.user, password=dbConn.pswd) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = list(cur.fetchall())
            # df = pd.DataFrame(data=data)

    return [dict(row) for row in data][0]['count']
###--------------------------------------------------------------------------





###================================== END OF FILE =================================

