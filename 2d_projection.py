import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import matplotlib.offsetbox as offsetbox
from PIL import Image
import numpy as np
from mdp_setup import mdp_setup



def render_2d_projection(proj_model, img_dict, grid=True, num_states=16, title="PCA on ViT"):
    """proj_model can be any dimensionality reduction method like PCA, Umap, etc
       The img_dict contains filenames of the images so that we can easily visualize a low dimensional clustering 
    """
    x_min, x_max = proj_model[:, 0].min(), proj_model[:, 0].max()
    y_min, y_max = proj_model[:, 1].min(), proj_model[:, 1].max()    


    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(x_min-1, x_max)
    ax.set_ylim(y_min-1, y_max)
    for i in range(len(img_dict)):
        x = proj_model[i, 0]
        y = proj_model[i, 1]
        

        filename = img_dict[i][2]
        img = Image.open(filename) 
        img = np.uint8(img)
        img = offsetbox.OffsetImage(img, zoom=0.04)
        
        ab = offsetbox.AnnotationBbox(img, (x,y), xycoords="data", frameon=False)
        ax.add_artist(ab)
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel('PC1', fontsize = 20)
    ax.set_ylabel('PC2', fontsize = 20)
    ax.grid(visible=True)

    if grid:
        x_states, y_states, actions = mdp_setup(proj_model, num_states = num_states)
        ax.set_xticks(x_states)
        ax.set_yticks(y_states)