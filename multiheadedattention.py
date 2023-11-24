# multiheaded attention:
import torch
import torch.nn as nn

"""
Goal of this script was to reproduce multiheaded attention in my own code
"""

torch.manual_seed(0)
embed_dim = 4
num_heads = 1
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)
print(multihead_attn.in_proj_bias, multihead_attn.bias_k, multihead_attn.bias_v, multihead_attn.add_zero_attn,
                multihead_attn.dropout, 'out_proj_weight', multihead_attn.out_proj.weight, multihead_attn.out_proj.bias)
print('multihead_attn.out_proj.weight.shape', multihead_attn.out_proj.weight.shape)
print('multihead_attn.in_proj_weight.shape', multihead_attn.in_proj_weight.shape)
qkv_matrix = torch.ones((4,4)) 
#Stack them vertically (along dimension 0) to create a 12x4 matrix
stacked_qkv_matrix = torch.cat([qkv_matrix, qkv_matrix, qkv_matrix], dim=0) # assume same weights for Wk, Wq, Wk

multihead_attn.in_proj_weight = nn.Parameter(stacked_qkv_matrix)
multihead_attn.out_proj.weight = nn.Parameter(qkv_matrix)
query = key = value = torch.ones(( 2,4))
print('multihead_attn.out_proj.weight.shape', multihead_attn.out_proj.weight.shape)
print('multihead_attn.in_proj_weight.shape', multihead_attn.in_proj_weight.shape)
print('actual weights are printed below:')
print('weights for in_proj_weight:\n', multihead_attn.in_proj_weight)
print('weights for out_proj.weight:\n', multihead_attn.out_proj.weight)

# Run Attention:
attn_output, attn_output_weights = multihead_attn(query, key, value)

# Print Attention output
print('attn_output:\n', attn_output)
print('attn_output_weights\n', attn_output_weights) # attn_output weights are simply the dot product of Q @ K.T * scale_factor, where scale_factor is q_dim

# custom attention:
data = torch.ones((2,4))
weight_matrix = qkv_matrix
print("weight_matrix\n", weight_matrix)
query = torch.matmul(data, multihead_attn.in_proj_weight[:4])
key = torch.matmul(data,multihead_attn.in_proj_weight[4:8])
print('key shape', key.shape)
value = torch.matmul(data, multihead_attn.in_proj_weight[8:])
q_dim = torch.tensor(query.shape[1])
scale = 1/torch.sqrt(q_dim)
print('scale is:', scale)
softmax = nn.Softmax(dim=-1)
key = torch.transpose(key, 0, 1)
query = query
print('key is of shape after transpose:', key.shape)
attn_weighted = torch.matmul(query, key) * scale
print("attn_weighted", attn_weighted)
attn_weighted_softmax = softmax(attn_weighted)
print('attn_weighted_softmax:\n', attn_weighted_softmax)
output = torch.matmul(attn_weighted_softmax , value)
output = torch.matmul(output, multihead_attn.out_proj.weight)
print(output)