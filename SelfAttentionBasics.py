import torch
import torch.nn as nn
''' 
Input - A matrix - Batch first data - 3D 
Output - Attention score , Attention output
Key terms - Query, Key, Value
Self Attention Architecture - Calcuate attention within itself
Steps
    1.Given input convert into Query, Key, Value using weighted matrix - Linear transformation
    2. Multiply Query with Key - Get Attention score
    3. Multiply Attention score with Value - Get Attention Output
'''
# prepare input
input_matrix = torch.rand(2,3,4) # batch, sequence, feature
print("input")
print(input_matrix)
embed_dim = 4 # of feature
num_heads = 1 # should divide embed_dim
# using pytorch prebuilt 
query = key = value = input_matrix
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads,batch_first= True)
attn_output, attn_output_weights = multihead_attn(query, key, value)
print("..................................")
print("attn_shape")
print(attn_output)
print(attn_output.shape, "- Output shape")
print("..................................")
print("attn_output_weights")
print(attn_output_weights)
print(attn_output_weights.shape, "- Attention score shape")
row_sum = torch.sum(input_matrix,dim=2)
print(row_sum)

# TODO: Check the sum of attention row wise and column wise and check
# TODO: What is Dk?
# TODO: Implement algorithm from scract
# code from scratch
print(key, "key")
key_transpose = torch.transpose(key, 1, 2)
result = torch.matmul(query, key_transpose)
print(result)
