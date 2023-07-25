import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

def label_encoding(df):

    # mapping_dict = {'D':0, 'W': 1, 'L': 2}
    # df['result'] = df['result'].map(mapping_dict)

    label_encoder = LabelEncoder()
    
    columns_to_encode = [ 'result', 'opponent', 'venue', 'formation', 'team', 'referee']

    for column in columns_to_encode:
        if column in df.columns:
            df[column] =  label_encoder.fit_transform(df[column])

        existing_data = {}

        try:
            with open('../map.json', 'r', encoding='utf-8') as json_file:
                if json_file.read().strip():
                    json_file.seek(0) 
                    existing_data = json.load(json_file)
        except FileNotFoundError:
            pass
        
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        mapping = {category: int(encoded_value.item()) for category, encoded_value in mapping.items()}
        existing_data.update(mapping)

        with open('../map.json', 'w',  encoding='utf-8') as json_file:
            json.dump(existing_data, json_file)

    return df