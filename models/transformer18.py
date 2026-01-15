import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        # dim=64,depth=18，heads=，dim_head=64,mlp_dim=128
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 64 x 8
        self.heads = heads # 8
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads # n=50,h=8
        # self.to_qkv(x)得到的尺寸为[b,50,64x8x3],然后chunk成3份
        # 也就是说，qkv是一个三元tuple,每一份都是[b,50,64x8]的大小
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 把每一份从[b,50,64x8]变成[b,8,50,64]的形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
		# 这一步不太好理解，q和k都是[b,8,50,64]的形式，50理解为特征数量，64为特征变量
        # dots.shape=[b,8,50,50]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # 不考虑mask这一块的内容
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
		# 对[b,8,50,50]的最后一个维度做softmax
        attn = dots.softmax(dim=-1)

		# 这个attn就是计算出来的自注意力值，和v做点乘，out.shape=[b,8,50,64]
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # out.shape变成[b,50,8x64]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out.shape重新变成[b,60,128]
        out =  self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
    # dim=128,fn=Attention/FeedForward
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
    # dim=128,hidden_dim=128
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class ViT(nn.Module):
    def __init__(self, signal_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
    # signal_size=1024,patch_size=64,num_classes=3,channels=1，dim=64
        super().__init__()
        assert signal_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_pathes = 1024/64 = 16
        num_patches = signal_size // patch_size
        # patch_dim = 64*1 = 64
        patch_dim = channels * patch_size 
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
		# self.patch_size = 64
        self.patch_size = patch_size
        # self.pos_embedding是一个形状为（1，17，64）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding是一个一维的卷积，最后输出的形状为（1，64, 16）后面要对最后两个维度互换
        self.patch_to_embedding = nn.Conv1d(1, dim, kernel_size = patch_size, stride = patch_size)
        # self.cls_token是一个随机初始化的形状为（1，1，64）这样的变量
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer后面会讲解
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask = None):
 		# p=32
        p = self.patch_size
        x = self.patch_to_embedding(x) # x.shape=[b,64,16]
        x = torch.transpose(x, 2, 1)
        b, n, _ = x.shape # n = 17

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        #print(x.shape)
        #print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1) # x.shape=[b,17,64]
        x += self.pos_embedding[:, :(n + 1)] # x.shape=[b,17,64]
        x = self.dropout(x) 

        x = self.transformer(x, mask) # x.shape=[b,17,64],mask=None

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature = x
        x = self.mlp_head(feature)
        return x, feature
    
# device = torch.device('cpu')
# model = ViT(1024,64,3,64,18,8,128).to(device)
# summary(model, input_size = (1,1024),batch_size=100)
