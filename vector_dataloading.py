import torch
from vdb_utils import init_vdb, delete_vdb, get_vdb_info, insert_embedding
import numpy as np

tile_classes = {(0,0) : ['water'], (0,1) : ['water'] , 
                (0,2) : ['neighborhood',  ],
                (0,3) : [ 'neighboorhood'],
                (0,4) : ['neighborhood'],
                (0,5) : [ 'neighborhood'] ,
                (0,6) : ['neighborhood'], (0,7) : ['neighborhood'], (0,8) : ['neighborhood'], (0,9) : ['industrial'], (0,10) : ['industrial'],
               (0,11) : ['industrial'], (0,12) : ['water'] , (0, 13) : ['water'] , (0,14) : ['water'],
                (1,0) : ['water'] , (1,1) : ['water'], (1,2) : ['neighborhood'], (1,3) : ['grass'], (1,4) : ['grass'], (1,5) : ['neighborhood' ], (1,6) : ['neighborhood'],
               (1,7) : ['neighborhood'], (1,8) : ['neighborhood'], (1,9) : ['neighborhood'] , (1, 10) : ['neighborhood'] , (1,11) : ['airport'], (1,12) : ['airport', ], (1, 13) : ['airport'],
               (1,14) : ['water'], (2,0) : ['water'], (2,1) : ['water'], (2,2) : ['grass'], (2,3) : ['grass'], (2,4) : ['grass'], (2,5) : ['grass'], (2,6) : ['neighborhood'],
               (2,7) : ['neighborhood'], (2,8) : ['neighborhood'], (2,9) : ['neighborhood'], (2,10) : ['neighborhood'], (2,11) : ['airport'], (2,12) : ['airport'], (2,13) : ['airport'],
               (2,14) : ['airport'], (3,0) : ['water'], (3,1) : ['water'], (3,2) : ['grass'], (3,3) : [ 'neighborhood'], (3,4) : ['grass'], (3,5) : ['grass'], (3,6) : ['grass'],
               (3,7) : ['neighborhood'], (3,8) : ['neighborhood'], (3,9) : ['neighborhood'], (3,10) : ['neighborhood'], (3,11) : ['airport'], (3,12) : ['airport'], (3,13) :
               ['water'], (3,14) : ['airport' ],
               (4,0) : ['water'], (4,1) : ['water'], (4,2) :['grass'], (4,3) : ['grass'], (4,4) : ['grass'], (4,5) : ['grass'], (4,6) : ['grass'], (4,7) : ['grass'], (4,8) : [ 'neighborhood'],
               (4,9) : ['neighborhood'], (4,10) : ['neighborhood'], (4,11) : ['neighborhood'], (4,12) : ['industrial'], (4,13) : ['industrial'], (4,14) : ['water']}

def min_max_tiles(client, collection_name, verbose=False):
    max_x = 0
    min_x = 1e6
    max_y = 0
    min_y = 1e6
    for i in range(200):
        out = client.retrieve( collection_name=collection_name,  ids=[i], with_vectors=True)
        if len(out) == 1: # means we have a result
            if out[0].payload['sensor'] == 'ps': # ps
                # get x, y coordinates for embedding:
                temp_x, temp_y = out[0].payload['x'], out[0].payload['y']
                if temp_x > max_x:
                    max_x = temp_x
                if temp_x < min_x:
                    min_x = temp_x
                if temp_y > max_y:
                    max_y = temp_y
                if temp_y < min_y:
                    min_y = temp_y
    if verbose:
        print('here are the min/maxes')
        print(f'max_x : {max_x} , min_x : {min_x}')
        print(f'max_y : {max_y} , min_y : {min_y}')
    return max_x, min_x, max_y, min_y

class VectorDataset(torch.utils.data.Dataset):
    """
    Class that holds the dataset in a vector DB. Will be wrapped in a dataloader to load data for contrastive learning.


    """
    def __init__(self, collection='sfo_tile_vectors'):
        super(VectorDataset).__init__()
        self.collection = collection
        print(self.collection)
        
        self.client = init_vdb(folder = 'qdrant_dbs/sfo_tiles_experiment', collection_name=self.collection)
        self.minmax = min_max_tiles(self.client, self.collection, verbose=True)
        self.max_x, self.min_x, self.max_y, self.min_y = self.minmax


        self.li_valid_ids = []
        for i in range(150):
            out = self.client.retrieve( collection_name=self.collection,  ids=[i], with_vectors=True)
            if len(out) == 1:
                self.li_valid_ids.append(out[0].id)
        #print(self.li_valid_ids)

    def __len__(self):
        return len(self.li_valid_ids)
    
    def __getitem__(self,index):
        # code to load an indexed vector:
        vdb_id = self.li_valid_ids[index]
        #get_vdb_info(client_name=self.client, collection_name='sfo_tile_vectors')
        vdb_payload = self.client.retrieve(collection_name=self.collection, ids = [vdb_id])[0].payload
        vdb_vector = self.client.retrieve(collection_name=self.collection, ids = [vdb_id], with_vectors=True)[0].vector
        #print(vdb_id, vdb_payload, vdb_vector)
        if vdb_payload['sensor'] == 's2':
            sensor = 's2'
            y_tile, x_tile = vdb_payload['chip_row'], vdb_payload['chip_col']
        if vdb_payload['sensor'] == 'ps':
            sensor = 'ps'
            x_tile, y_tile = vdb_payload['x'] - self.min_x, vdb_payload['y'] - self.min_y

        tile_metadata = tile_classes[(y_tile,x_tile)]

        # convert vector to torch vector:
        vdb_vector_torch = torch.tensor(vdb_vector)

        return vdb_vector_torch, x_tile, y_tile, sensor, tile_metadata

if __name__ == "__main__":
    vdb_dataset = VectorDataset(collection='sfo_tile_vectors_CLIP_new')
    dataloader = torch.utils.data.DataLoader(vdb_dataset, batch_size=10,
                                              num_workers = 1, drop_last=False, shuffle=True)
    
    for n, i in enumerate(dataloader):
        print(i[0].shape)
        arr = np.array(i[4])
        pairwise_bools = arr == arr.T
        print(pairwise_bools)


        



