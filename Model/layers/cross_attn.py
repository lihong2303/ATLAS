import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_attn_noproj(q,k,v,num_heads):
    q, k, v = [x.transpose(1, 0) for x in (q, k, v)]
    src_len,bsz,embed_dim = k.shape
    tgt_len,_,_ = q.shape
    head_dim = embed_dim // num_heads
    
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    
    q_scaled = q / math.sqrt(embed_dim)
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    # attn_output_weights = F.dropout(attn_output_weights, p=0.1)
    
    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = attn_output.view(tgt_len, bsz, embed_dim).transpose(1,0)
    
    return attn_output