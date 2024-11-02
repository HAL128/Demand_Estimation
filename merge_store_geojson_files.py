from helpers.helper_module import os, json, glob


folder_path = './japan_shops_grid'

all_features = []

geojson_files = glob.glob(os.path.join(folder_path, '*.geojson'))

for file_path in geojson_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        all_features.extend(data['features'])

merged_geojson = {
    "type": "FeatureCollection",
    "features": all_features
}

output_file = 'merged_japan_shops_grid.geojson'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(merged_geojson, outfile, ensure_ascii=False, indent=2)

print(f"Merged GeoJSON file created: {output_file}")
