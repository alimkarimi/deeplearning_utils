import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client import models as qd_models
from qdrant_client.models import Distance, PointStruct, VectorParams
import os
import numpy as np
import pandas as pd

def init_vdb(folder= 'qdrant_dbs/sfo_tiles', collection_name = 'sfo_tile_vectors', vector_size = 768):
    if os.path.exists(folder):
        # just load the db:
        client = QdrantClient(path=folder)
        if client.collection_exists(collection_name=collection_name):
            print('collection already exists ... getting info')
            get_vdb_info(client_name=client, collection_name=collection_name)
            
            return client

        else:      
            client.create_collection( collection_name=collection_name, 
                                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),)
            return client

    if not os.path.exists(folder):
        print(f'path {folder} does not exist ...')
        os.makedirs(folder)
        # create db
        # load client

        client = QdrantClient(path=folder)  # Persists changes to disk, fast prototyping
        print('made it here')
        # create the collection
        client.create_collection( collection_name=collection_name, 
                                vectors_config=VectorParams(size=768, distance=Distance.COSINE),)
            
        return client
    
def init_db(name : str):
    df = pd.DataFrame()
    df['tile_id'] = None
    df['chip_row'] = None
    df['chip_col'] = None
    df['file_name'] = None
    df.to_pickle(name)

    return df

def insert_row(df, tile_id, chip_row, chip_col, file_name):
    # first, get the length of dataframe:
    df_rows = df.shape[0]
    df.loc[df_rows,'tile_id'] = tile_id 
    df.loc[df_rows,'chip_row'] = int(chip_row / 165)
    df.loc[df_rows, 'chip_col'] = int(chip_col / 165)
    df.loc[df_rows, 'file_name'] = file_name

    return df

def delete_vdb(client_name, collection_name : str):
     client_name.delete_collection(collection_name=collection_name)
     print('deleted collection!')


def get_vdb_info(client_name, collection_name : str, verbose=False):
    """
    Get information about how many vectors are stored in the DB
    """
    vector_count = client_name.get_collection(collection_name=collection_name).vectors_count
    if verbose:
        print(f'db has {vector_count} vectors as of now')
    return vector_count # can use vector count as the max id.

def insert_embedding(client_name, collection_name : str, id: int, embedding, metadata : dict):
    """
    Insert embedding into vector DB with metadata in payload
    """
    client_name.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=id,
                payload=metadata,
                vector=embedding,
            ),
        ],
    )

if __name__ == "__main__":
    # test insert:

    client = init_vdb(folder='qdrant_dbs/sfo_test_2')
    print('here!')
    num_pts = get_vdb_info(client_name=client, collection_name='sfo_tile_vectors')
    insert_embedding(client_name=client, collection_name='sfo_tile_vectors', id = num_pts+1, embedding=np.zeros(768).tolist(), 
                     metadata = {})
    print('inserted embedding')
    get_vdb_info(client_name=client, collection_name='sfo_tile_vectors')
    #delete_db(client_name = client, collection_name='sfo_tile_vectors')



    


