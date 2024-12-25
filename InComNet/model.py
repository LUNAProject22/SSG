
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor, Generator
import numpy as np

from config import parse_args

args = parse_args()

def getPositionEncoding(seq_len, d):
    n=10000
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)

    return P
 

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

def normal_(self, mean: float=0, std: float=1, *, generator: Optional[Generator]=None) -> Tensor: ...


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.drop_rate = drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(self.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y, mask=None):
              
        B1, N1, C = x.shape # bs, roles, dimension
        B2, N2, C = y.shape # bs, image_tokens, dimension
    
        q = self.q(x).reshape(
            B1, N1, 1, 
            self.num_heads, 
            C // self.num_heads #self.num_heads
        ).permute(2, 0, 3, 1, 4)
        
        kv = self.kv(y).reshape(
            B2, N2, 2, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Cross-Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C)

        x = self.proj(x)

        if mask is not None:
            if B1 > 1:
                
                x = mask*x
            else:
                
                x = mask.unsqueeze(1).float()*x    
        
            return x, y, attn
        else:
            return x, y, attn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
    
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_rate,
        )

    def forward(self, x, y, mask=None):
                        
        x_block, y, attn = self.attn(
                    self.norm1(x), 
                    self.norm2(y), 
                    mask=mask
                )
        x_norm = self.norm3(x_block)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, y, attn


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super(MultiheadAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        attn_output, attn_weights = self.multihead_attention(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)
        output = self.layer_norm(attn_output)
        return output, attn_weights
    

class InComNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_ans_classes = [364,17, 125] 
        num_layers = 4
        num_heads = 4       

        if args.clip_model == 'ViT-B/32':
            self.proj_dim = 512
        elif args.clip_model == 'ViT_L_14_336' or args.clip_model == 'ViT_L_14_336_sft':
            self.proj_dim = 768    

        # obj SR 
        self.obj_sr_q_proj = nn.Linear(self.proj_dim, self.proj_dim) 
        self.obj_sr_kv_proj = nn.Linear(self.proj_dim, self.proj_dim)      
        self.obj_sr_output_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.obj_sr_classifier = nn.Linear(self.proj_dim, self.num_ans_classes[0])

        self.obj_sr_q = nn.Parameter(torch.zeros(1,7,self.proj_dim))
        self.obj_sr_q.data.normal_(mean=0.0, std=0.02) 
        
        self.obj_sr_pos_emb = getPositionEncoding(7, self.proj_dim) 
        
        obj_sr_attention_layer = MultiheadAttentionLayer(embed_dim=self.proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.obj_sr_atts = nn.ModuleList([obj_sr_attention_layer for i in range(num_layers)])

        # verb
        self.verb_q_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_kv_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_output_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_classifier = nn.Linear(self.proj_dim, self.num_ans_classes[1])
        self.verb_cls =  nn.Parameter(torch.randn(1, 1, self.proj_dim)) 
        
        self.verb_q = nn.Parameter(torch.zeros(1,1,self.proj_dim)) 
        self.verb_q.data.normal_(mean=0.0, std=0.02) 

        verb_attention_layer = MultiheadAttentionLayer(embed_dim=self.proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.verb_atts = nn.ModuleList([verb_attention_layer for i in range(num_layers)])


        # verb SR
        self.verb_sr_q_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_sr_kv_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_sr_output_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.verb_sr_classifier = nn.Linear(self.proj_dim, self.num_ans_classes[2])

        self.verb_sr_q = nn.Parameter(torch.zeros(1, 7, self.proj_dim))
        self.verb_sr_q.data.normal_(mean=0.0, std=0.02) 

        self.verb_sr_pos_emb = getPositionEncoding(7, self.proj_dim)

        verb_sr_attention_layer = MultiheadAttentionLayer(embed_dim=self.proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.verb_sr_atts = nn.ModuleList([verb_sr_attention_layer for i in range(num_layers)])

    def forward(self, obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_frame, obj_sr_mask, verb, verb_mask, verb_sr_frame, verb_sr_mask):  

        #### Object SR
        if (obj_sr_flag and not verb_flag and not verb_sr_flag):           

            obj_sr_kv = obj_sr_frame
            obj_sr_kv = self.obj_sr_kv_proj(obj_sr_kv)
            obj_sr_q = self.obj_sr_q.repeat(obj_sr_frame.size(0), 1, 1)

            self.obj_sr_pos_emb = self.obj_sr_pos_emb.to(args.device)
            obj_sr_q = obj_sr_q + self.obj_sr_pos_emb

            if obj_sr_mask is not None:
                obj_sr_mask = obj_sr_mask.unsqueeze(-1).repeat(1,1,self.proj_dim)
           
            for layer in self.obj_sr_atts:
                obj_sr_q, obj_sr_attn = layer(obj_sr_q, obj_sr_kv, obj_sr_kv)

            obj_sr_q = obj_sr_q*obj_sr_mask
            obj_sr_q = self.obj_sr_output_proj(obj_sr_q)
            obj_sr_logits = self.obj_sr_classifier(obj_sr_q)

            return obj_sr_logits, obj_sr_q

        #### Verb predicate
        if (not obj_sr_flag and verb_flag and not verb_sr_flag):

            verb_kv = verb
            verb_kv = self.verb_kv_proj(verb_kv)
            verb_q = self.verb_q.repeat(verb.size(0), 1, 1)

            verb_mask = None
            for layer in self.verb_atts:
                verb_q, verb_attn = layer(verb_q, verb_kv, verb_kv)
            
            verb_q = self.verb_output_proj(verb_q)
            verb_logits = self.verb_classifier(verb_q)
                
            return verb_logits, verb_q

        #### Verb predicate SR
        if (not obj_sr_flag and not verb_flag and verb_sr_flag):

            verb_sr_kv = verb_sr_frame
            verb_sr_kv = self.verb_sr_kv_proj(verb_sr_kv)
            verb_sr_q = self.verb_sr_q.repeat(verb_sr_frame.size(0), 1, 1)

            self.verb_sr_pos_emb = self.verb_sr_pos_emb.to(args.device)
            verb_sr_q = verb_sr_q + self.verb_sr_pos_emb

            if verb_sr_mask is not None:
                verb_sr_mask = verb_sr_mask.unsqueeze(-1).repeat(1,1,self.proj_dim)
            
            for layer in self.verb_sr_atts:
                verb_sr_q, verb_sr_attn = layer(verb_sr_q, verb_sr_kv, verb_sr_kv)

            verb_sr_q = verb_sr_q*verb_sr_mask
            verb_sr_q = self.verb_sr_output_proj(verb_sr_q)
            verb_sr_logits = self.verb_sr_classifier(verb_sr_q)

            return verb_sr_logits, verb_sr_q


class InComNetPerson(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_ans_classes = [364,17, 125] 
        num_layers = 4
        num_heads = 4       

        if args.clip_model == 'ViT-B/32':
            self.proj_dim = 512
        elif args.clip_model == 'ViT_L_14_336' or args.clip_model == 'ViT_L_14_336_sft':
            self.proj_dim = 768

        self.pos_emb = torch.nn.Parameter(torch.randn(7, self.proj_dim)) # learned pos emb
       

        self.person_sr_kv_proj = nn.Linear(self.proj_dim, self.proj_dim)  
        self.person_sr_output_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.person_sr_classifier = nn.Linear(self.proj_dim, self.num_ans_classes[0])

        self.person_sr_q = nn.Parameter(torch.zeros(1,5,self.proj_dim))
        self.person_sr_q.data.normal_(mean=0.0, std=0.02) # std = 1
            
        self.person_sr_q_pos_emb = getPositionEncoding(5, self.proj_dim)
        # self.person_sr_kv_pos_emb = getPositionEncoding(7, self.proj_dim)

        person_sr_attention_layer = MultiheadAttentionLayer(embed_dim=self.proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.person_sr_atts = nn.ModuleList([person_sr_attention_layer for i in range(num_layers)])
        
    def forward(self, person_sr_feat):  

        person_sr_kv = self.person_sr_kv_proj(person_sr_feat)
        person_sr_q = self.person_sr_q.repeat(person_sr_feat.size(0), 1, 1)
        person_sr_q = person_sr_q + self.person_sr_q_pos_emb.to(args.device)

        for layer in self.person_sr_atts:
            person_sr_q, person_sr_attn = layer(person_sr_q, person_sr_kv, person_sr_kv)
        
        person_sr_q = self.person_sr_output_proj(person_sr_q)
        person_sr_logits = self.person_sr_classifier(person_sr_q)

        return person_sr_logits, person_sr_q
