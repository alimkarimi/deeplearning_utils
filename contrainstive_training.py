from vector_dataloading import VectorDataset
from model import MLP
from losses import PairwiseL2ContrastiveLoss
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt



pcl = PairwiseL2ContrastiveLoss(margin=10, debug=False)
torch.manual_seed(0)
epochs = 300
dataset = VectorDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=75,
                                              num_workers = 1, drop_last=True, shuffle=True)
model = MLP()

running_loss = []
running_loss_pos = []
running_loss_neg = []

optimizer = optim.Adam(model.parameters(), lr = 0.005, betas = (0.9, 0.99))

for epoch in range(epochs):
    for n, loaded_data in enumerate(dataloader):
        optimizer.zero_grad()
        #print(n)
        vectors, x_tile, y_tile, sensor, tile_metadata = loaded_data # unpack
        print(tile_metadata)

        # push vectors through MLP:
        vectors = model(vectors)
        #print(x_tile, y_tile, vectors.shape, sensor, tile_metadata)
        
        #print(tile_metadata)
        labels = np.array(tile_metadata)
        pairs_mask = labels == labels.T
        pairs_mask_positive = torch.tensor(pairs_mask)
        pairs_mask_negative = torch.tensor(~pairs_mask)
        pairs_mask_negative.fill_diagonal_(0)
        #print('converted pairs_mask to tensor')
        pairs_mask_positive.fill_diagonal_(0) # this gets rid of the notion that every vecor is similar to itself.
        #print(pairs_mask)  # pairs mask tells us which pairs are positive pairs and which pairs are negative pairs
        #print(vectors.shape)
        # get distance matrix:
        # we want the pairwise distance between ALL vectors. ||x - y|| = (x - y)@(x - y).T = ||x|| - 2x @ y.T + ||y||, where || is 
        # the L2 (euclidean) norm. 

        #print(torch.max(vectors[0]), 'max!!')
        pairwise_dot_products = vectors @ vectors.T
        #print(pairwise_dot_products)
        # diagonal of dot product is the L2 norm of all the x vectors.
        squared_norms_embedding_vecs = torch.diagonal(pairwise_dot_products)
        s = squared_norms_embedding_vecs.shape[0]

        #print(squared_norms_embedding_vecs.view(1,s))
        #print(squared_norms_embedding_vecs.view(s,1))

        distance_matrix = squared_norms_embedding_vecs.view(1,s) - 2.0 * pairwise_dot_products + squared_norms_embedding_vecs.view(s,1)
        #print(distance_matrix)

        # get positive pairs from True mask
        #print(pairs_mask)
        pos_pairs_extracted = pairs_mask_positive * distance_matrix # 1 is true. So, this will element wise multiply 1 by every pos pair
        # in distance matrix, 0 by every negative pair.
        #print(pos_pairs_extracted, 'pos pairs !')
        # get negative paris from False mask
        neg_pairs_extracted = pairs_mask_negative * distance_matrix # we have reversed the pairs mask. We now multiply 1 with every
        # negative pair, 0 by every positive pair. 
        #print(neg_pairs_extracted, 'negative pairs !!')
        print('pos pairs:', pos_pairs_extracted)
        print('neg pairs', neg_pairs_extracted)

        # compute loss:
        loss, positves_loss, negatives_loss = pcl(pos_pairs_extracted, neg_pairs_extracted, pairs_mask_positive,
                                                  pairs_mask_negative, tile_metadata)
        print('batch loss is ', loss)
        
        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.

        running_loss.append(loss.item())
        running_loss_pos.append(positves_loss.item())
        running_loss_neg.append(negatives_loss.item())


    
plt.plot(running_loss[50:], label='total loss')
plt.plot(running_loss_neg[50:], label='neg loss')
plt.plot(running_loss_pos[50:], label='pos loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over training - Contrastive Learning')
plt.savefig('Loss_plot.jpg')


# save model
torch.save(model.state_dict(), 'contrastive_MLP.pth')

        
