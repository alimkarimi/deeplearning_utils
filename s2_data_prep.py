# using sentinel-hub platform
import shub_auth
from shub_auth import client_id, client_secret, token, oauth
import requests
import json
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import mercantile
import io
import pandas as pd
from vdb_utils import init_db, insert_row

send_request = True
# norcal bbox:
# Northern California bbox:
# Top left corner:
min_lon = y_top = -122.51422006402419
min_lat = x_top = 37.63953742890372

# bottom right corner
max_lon = y_bottom = -122.35540953649489
max_lat = x_bottom = 37.60527120819383

tile1 = mercantile.tile(min_lon, min_lat, zoom=15)
tile2 = mercantile.tile(max_lon, max_lat, zoom=15)
print(tile1, tile2)
print('num of tiles in x direction:', np.abs(tile1.x - tile2.x))
print('num of tiles in y direction:', np.abs(tile1.y - tile2.y))
scale_factor_x = int(np.abs(tile1.x - tile2.x)) + 1
scale_factor_y = int(np.abs(tile1.y - tile2.y)) + 1
print(scale_factor_x, scale_factor_y)
img_width = 165 * scale_factor_x
img_height = 165 * scale_factor_y
print(img_width, img_height)


access_token = token['access_token']

# 1 - set up parameters:
#bbox = [-88.72171, 17.11848, -87.342682, 17.481674] # lower left and upper right
bbox = [-122.51422006402419, 37.60527120819383, -122.35540953649489, 37.63953742890372] # lower left and upper right
start_date = "2020-06-01"
end_date = "2020-08-31"
collection_id = "sentinel-2-l2a"

# 2 - create an evalscript request body and payload:
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: {
      bands: 3,
      sampleType: "AUTO" // default value - scales the output values from [0,1] to [0,255].
    }
  }
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
}
"""

# create json_request (incorporates the eval script)
json_request = {
  "input": {
    "bounds": {
      "bbox": bbox
    },
    "data": [
      {
        "dataFilter": {
          "timeRange": {
            "from": "2020-06-01T00:00:00Z",
            "to": "2020-08-31T23:59:59Z"
          },
          "maxCloudCoverage": "1"
        },
        "type": "sentinel-2-l2a"
      }
    ]
  },
  "output": {
    "width": img_width ,
    "height": img_height ,
    "responses": [
      {
        "identifier": "default",
        "format": {
          "type": "image/jpeg"
        }
      }
    ]
  },
"evalscript": evalscript
}

if send_request:
    # send the POST request:
    # Set the request url and headers
    url_request = 'https://services.sentinel-hub.com/api/v1/process'
    headers_request = {
        "Authorization" : "Bearer %s" %token['access_token']
    }

    #Send the request
    response = oauth.request(
        "POST", url_request, headers=headers_request, json = json_request
    )


    # # Print the response
    # read the image as numpy array
    print(response)
    np_img = np.array(Image.open(io.BytesIO(response.content)))
    print(np_img.shape)
    print(np.max(np_img))

    plt.imshow(np_img)
    plt.savefig('tests2.jpg')

    # save numpy file so we do not need to make more requests:
    np.save('sfo_tile', np_img)

if send_request == False:
    # load sfo_tile
    np_img = np.load('sfo_tile.npy')
    print(np_img.shape)

# now, break that image up into smaller tiles. Get metadata of these tiles, place them into df for traceability.
df = init_db(name='metadata_df.pkl')

for i in range(scale_factor_y):
    for j in range(scale_factor_x):
        scaled_i = i * 165
        scaled_j = j * 165
        print(scaled_i, scaled_j)
        # get i,j to i + 175, j + 175
        temp_tile = np_img[scaled_i : scaled_i + 165, scaled_j : scaled_j + 165, :]
        print(temp_tile.shape)
        # save tile as numpy and jpg:
        # numpy:
        file_name = f's2_tiles/tile_row{scaled_i}_tile_col_{scaled_j}'
        np.save(file_name + '.npy', temp_tile)
        # jpg:
        im = Image.fromarray(temp_tile)
        im.save(f's2_tiles/tile_row{scaled_i}_tile_col_{scaled_j}.jpg')

        # add relevant info to metadata df:
        insert_row(df = df, tile_id = None, chip_row = scaled_i, chip_col = scaled_j, file_name = file_name)

# save dataframe
df.to_pickle('metadata_df.pkl')

print(df)
