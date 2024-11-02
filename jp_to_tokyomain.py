from helpers.helper_module import json, pickle, Point, gp, Feature, FeatureCollection, dump


# Constants
STORE_GEOJSON_PATH = '../../Data/DataHirata/store.geojson'
POLYGON_PICKLE_PATH = '../../Data/Polygons/tokyoMainPolygon2.pkl'
OUTPUT_JSON_PATH = '../../Data/TokyoMain_Hirata/store.json'
OUTPUT_GEOJSON_PATH = 'out.geojson'

def replace_polygon_with_centroid(geometry):
    """Replace polygon geometries with their centroids."""
    return geometry.centroid if geometry.geom_type == 'Polygon' else geometry

def load_store_data():
    """Load and preprocess store data from GeoJSON file."""
    store_data = gp.read_file(STORE_GEOJSON_PATH)
    store_data.set_crs(epsg=4326, inplace=True)
    store_data['geometry'] = store_data['geometry'].apply(replace_polygon_with_centroid)
    return store_data

def load_polygon():
    """Load polygon data from pickle file."""
    with open(POLYGON_PICKLE_PATH, 'rb') as f:
        return pickle.load(f)

def process_store_data(store_data, polygon):
    """Process store data and write to JSON file."""
    with open(OUTPUT_JSON_PATH, 'w') as output_file:
        for _, row in store_data.iterrows():
            point = row['geometry']
            if polygon.contains(point):
                store_info = {
                    'type': row['type'],
                    'latitude': point.y,
                    'longitude': point.x
                }
                json.dump(store_info, output_file, ensure_ascii=False)
                output_file.write(',\n')

def create_geojson_feature_collection(points_data):
    """Create a GeoJSON FeatureCollection from points data."""
    features = []
    for point in points_data:
        feature = Feature(
            geometry=Point((point['longitude'], point['latitude'])),
            properties={
                'name': point['name'],
                'marker-color': point['color'],
                'marker-size': 'medium',
                'marker-symbol': ''
            }
        )
        features.append(feature)
    return FeatureCollection(features)

def write_geojson(feature_collection, output_path):
    """Write GeoJSON FeatureCollection to file."""
    with open(output_path, 'w') as f:
        dump(feature_collection, f, indent=2)

def main():
    # Process store data
    store_data = load_store_data()
    polygon = load_polygon()
    process_store_data(store_data, polygon)

    # Create and write GeoJSON
    points_data = [
        {
            'name': 'Ho Chi Minh City Opera House',
            'latitude': 10.777406835725145,
            'longitude': 106.70299858740313,
            'color': '#ff2600'
        },
        {
            'name': 'Independence Palace',
            'latitude': 10.778137451508647,
            'longitude': 106.69531332149265,
            'color': '#00ff26'
        }
    ]
    feature_collection = create_geojson_feature_collection(points_data)
    write_geojson(feature_collection, OUTPUT_GEOJSON_PATH)

if __name__ == "__main__":
    main()