class NeuImgField(nn.Module):

    def __init__(self, n_hidden, dim_hidden, n_heads, dim_head, dim_out, L=10, do_bn=True, act=nn.Tanh()):
        super().__init__()
        assert dim_out>=1
        self.L = L
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.n_heads = n_heads

        self.net = nn.ModuleList([
            nn.Conv1d(2 * (self.L * 2 + 1) * n_heads, dim_hidden * n_heads, 1, groups=n_heads)
        ] + [
            nn.Conv1d(dim_hidden * n_heads, dim_hidden * n_heads, 1, groups=n_heads) for _ in range(n_hidden-1)
        ] + [
            nn.Conv1d(dim_hidden * n_heads, dim_head * n_heads, 1, groups=n_heads)
        ])

        self.bn_acts = nn.ModuleList([nn.Sequential(*[
            nn.BatchNorm1d(dim_hidden * n_heads) if do_bn else nn.Identity(), act
        ]) for _ in range(n_hidden)])
        self.out = None if dim_out==1 else nn.Linear(dim_head * n_heads, dim_out, bias=False)

    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x * torch.pi))
            out.append(torch.cos(2 ** j * x * torch.pi))
        return torch.cat(out, dim=2)

    def forward(self, xs, feats=None):
        batch_size, num_points = xs.shape[:2]
        xs = self.positional_encoding(xs, self.L)
        hs = xs[:,:,None].repeat(1,1,self.n_heads,1)
        hs = hs.reshape([-1, self.n_heads * 2 * (self.L * 2 + 1), 1])

        for i_layer, layer in enumerate(self.net):
            hs = layer(hs)
            if i_layer<self.n_hidden:
                if feats is not None and feats[i_layer] is not None:
                    scale, bias = [feats[i_layer][i][:,None,:,None].repeat(1,num_points,1,1).reshape(
                        [batch_size * num_points, -1, 1]
                    ) for i in range(2)]
                    hs = (hs - torch.mean(hs, dim=1, keepdim=True)) / torch.std(hs, dim=1, keepdim=True)
                    hs = hs * scale + bias
                hs = self.bn_acts[i_layer](hs)
        hs = hs.squeeze(2)
        if self.out is None:
            hs = torch.mean(hs, dim=1)
        else:
            hs = self.out(hs)
        hs = hs.reshape([batch_size, num_points, -1])
        return hs

class GroupEmbeddingSum(nn.Module):
    
    def __init__(self, dim_embedding, n_head, act=nn.Tanh()):
        super().__init__()
        self.n_head = n_head
        self.dim_embedding = dim_embedding
        
        self.net_embed = nn.Sequential(*[
            nn.Conv1d(n_head, dim_embedding*n_head, 1, groups=n_head), act,
            nn.Conv1d(dim_embedding*n_head, dim_embedding*n_head, 1, groups=n_head), act,
            nn.Conv1d(dim_embedding*n_head, dim_embedding*n_head, 1, groups=n_head)
        ])
        self.net_post = nn.Sequential(*[
            nn.Linear(dim_embedding, dim_embedding*2), act,
            nn.Linear(dim_embedding*2, dim_embedding*2), act,
            nn.Linear(dim_embedding*2, 1), nn.Sigmoid()
        ])
    
    def forward(self, xs, mask=None):
        hs = self.net_embed(xs).reshape([xs.shape[0], self.n_head, -1])
        if mask is not None:
            hs = hs * mask
        hs = self.net_post(torch.sum(hs, dim=1))
        return hs
