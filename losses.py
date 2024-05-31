import torch
import torch.nn as nn

class PairwiseL2ContrastiveLoss(nn.Module):
    def __init__(self, margin = 100, debug=False):
        super(PairwiseL2ContrastiveLoss, self).__init__()
        #self.cos_sim = nn.CosineSimilarity(dim=0)
        self.margin = margin
        self.debug = debug


    def forward(self, pos_pairs_extracted, neg_pairs_extracted, pairs_mask_positive, pairs_mask_negative, labels):
        # square positive pairs loss:
        pos_pairs_extracted_squared = torch.square(pos_pairs_extracted)
        
        pos_pairs_extracted_squared_summed = torch.sum(pos_pairs_extracted_squared)

        if self.debug:
            print('shape of positive pairs', pos_pairs_extracted.shape)
            print('shape of pos_pairs extracted and summed', pos_pairs_extracted_squared_summed.shape)
            print('pos pairs extracted:',pos_pairs_extracted)

        # in neg_pairs_extracted, positive pairs should be 0
        if self.debug:
            print('original negative pairs', neg_pairs_extracted)

        #pairs_mask.fill_diagonal_(0) # set diagonal to 0 b/c self to self should be positive. We do not want to count the 
        # positives in the negative pairs distance calculation, so, set to 0. 
        print(self.margin  - neg_pairs_extracted)
        print('above is margin - distance')
        max_margin_distance = torch.maximum(input= self.margin - neg_pairs_extracted, other=torch.tensor([0.0]))
        max_margin_distance = pairs_mask_negative * max_margin_distance # only use negative pairs for the loss computation,
        # zero out the rest of the pairs. 
        neg_pairs_extracted_and_squared = torch.square(max_margin_distance)
        neg_pairs_extracted_squared_and_summed = torch.sum(neg_pairs_extracted_and_squared)
        
        # apply mask so that only negative pairs are counted towards negative pairs loss:
        print('can this be used for the negative mask?', pairs_mask_negative)
        print('this is labels', labels)


        if self.debug:
            print('shape of neg pairs', neg_pairs_extracted.shape)
            print('result of distance with margin (max_margin_distance):', neg_pairs_extracted_and_squared)

        print('FINAL inputs to loss func:', pos_pairs_extracted_squared_summed, neg_pairs_extracted_squared_and_summed)

        loss = pos_pairs_extracted_squared_summed + neg_pairs_extracted_squared_and_summed
        return loss, pos_pairs_extracted_squared_summed, neg_pairs_extracted_squared_and_summed

# cos_sim = nn.CosineSimilarity(dim=0)
# x = torch.tensor([1.0, 2.0])
# y = torch.tensor([1.0, 1.0])
# print(1- cos_sim(x,y))
