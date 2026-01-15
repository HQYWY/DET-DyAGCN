import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


def norm(adj: Tensor, add_loop):
    if add_loop:
        adj = adj.clone()
        idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
        adj[..., idx, idx] += 1
    deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
    adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
    return adj


class MultiShallowEmbedding(nn.Module):

    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs
        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))

    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)

    def forward(self, device):
        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        # remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        # top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)
        idx = torch.tensor([i // self.k for i in range(indices.size(0))], device=device)
        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)
        return adj


class GroupLinear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups,
                                   bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()

    def forward(self, x: Tensor, is_reshape: False):
        # x (Tensor): [B, C, N, F] or [B, C, G, N, F//G] (is_reshape)
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups
        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)
        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        # out: [B, C_out, G, N, F//G]
        return out


# class MultiHeadSlidlAtentModule(nn.Module):
#     def __init__(self, multi_head_embed_dim, num_heads):
#         super().__init__()
#         self.multi_self_attn = nn.MultiheadAttention(multi_head_embed_dim, num_heads)
#
#     def reset_parameters(self):
#         # default
#         pass
#
#     def forward(self, inputs: list):
#         # window attention
#         B, N, L = inputs[0].size()
#         re_out = []
#         for window in inputs:
#             re_shape_win = window.transpose(0, 1)
#             out, attn = self.multi_self_attn(re_shape_win, re_shape_win, re_shape_win)
#             re_out.append(out.transpose(0, 1))
#         return re_out


# class Nodes_Init_Embedding(nn.Module):
#     def __init__(self, nodes, node_embed_dim, device):
#         super().__init__()
#         self.device = device
#         self.node_embedding = Parameter(Tensor(nodes, node_embed_dim)).to(device)
#         self.nodes = nodes
#
#     def reset_parameters(self):
#         init.xavier_uniform_(self.node_embedding)
#
#     def forward(self, topk: int, batch_size: int):
#         # construct cosine similarity matrix
#         num_embeddings = self.nodes
#         A = torch.zeros(num_embeddings, num_embeddings)  # initialize similarity matrix
#         # set the gradients of the parameter tensor to None to ensure
#         # gradients are not computed during cosine similarity calculation.
#         with torch.no_grad():
#             # compute cosine similarity between each pair of embeddings
#             print(self.node_embedding.shape)
#             for i in range(num_embeddings):
#                 for j in range(num_embeddings):
#                     # cosine similarity
#                     cosine_sim = torch.dot(self.node_embedding[i], self.node_embedding[j]) / \
#                                  (torch.norm(self.node_embedding[i]) * torch.norm(self.node_embedding[j]))
#                     A[i][j] = cosine_sim.item()
#         sparse_A = torch.zeros_like(A)
#         # apply sparsification to each row
#         for i in range(num_embeddings):
#             # column indices corresponding to the topk maximum values
#             topk_indices = torch.topk(A[i], k=topk).indices
#             sparse_A[i][topk_indices] = 1
#         self.node_embedding = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
#         sparse_A = sparse_A.unsqueeze(0).expand(batch_size, -1, -1)
#         return self.node_embedding, sparse_A.to(self.device)


class StackConv(nn.Module):
    def __init__(self, input_size, output_size, kern_size_first, kern_size_second, padding_first,
                 padding_second, eps=0, train_eps=True):
        super().__init__()
        self.time_conv_first = nn.Conv2d(input_size, output_size, kern_size_first, padding=padding_first)
        self.time_conv_second = nn.Conv2d(input_size, output_size, kern_size_second, padding=padding_second)
        self.relu = nn.ReLU()
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.time_conv_first.reset_parameters()
        self.time_conv_second.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def forward(self, x: Tensor):
        output_first_tensor = self.time_conv_first(x)
        # [B,output_size, 1, T]
        output_second_tensor = self.time_conv_second(x).expand(output_first_tensor.shape)
        # [B,2*output_size, N, T]
        output_tensor = torch.cat((output_first_tensor, output_second_tensor), dim=1)
        # output_tensor * self.eps + self.time_res(res_x) * (1 - self.eps)
        return output_tensor


class GraphAttentionModule(nn.Module):
    def __init__(self, in_features, out_features, device, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(4 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, inputs: Tensor, nodes_embedding: Tensor):
        # torch.Size([15, 2, 160]) torch.Size([15, 2, 2])
        batch_size = nodes_embedding.size(0)
        x_w = torch.matmul(inputs, self.W)
        h = torch.cat([nodes_embedding, x_w], dim=2)
        N = h.size(1)
        # extend to compute attention scores
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1),
                             h.repeat(1, N, 1)], dim=2).view(batch_size, N, -1, 4 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # softmax
        attention = F.softmax(e, dim=2)
        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, x_w)
        return F.relu(h_prime), attention


class GAT(nn.Module):
    def __init__(self, slot_length, nodes_gat_hidden, nodes_embedding, device, dropout=0.6, alpha=0.2):
        super().__init__()
        self.attention1 = GraphAttentionModule(slot_length, nodes_gat_hidden, device, dropout=dropout, alpha=alpha)
        self.attention2 = GraphAttentionModule(nodes_gat_hidden, nodes_embedding, device, dropout=dropout, alpha=alpha)

    def reset_parameters(self):
        self.attention1.reset_parameters()
        self.attention2.reset_parameters()

    def forward(self, inputs: list, nodes_embedding: Tensor):
        # torch.Size([15, 2, 160]) torch.Size([15, 2, 16]) torch.Size([15, 2, 2])
        re_nodes_embedding = []
        re_nodes_attention_in = []
        re_nodes_attention_out = []
        for slot in inputs:
            slot, attention1 = self.attention1(F.dropout(slot, 0.6, training=self.training),
                                               nodes_embedding)
            slot, attention2 = self.attention2(F.dropout(F.relu(slot), 0.6, training=self.training),
                                               nodes_embedding)
            slot = F.log_softmax(slot, dim=1)
            re_nodes_embedding.append(slot)
            re_nodes_attention_in.append(attention1)
            re_nodes_attention_out.append(attention2)
        return re_nodes_embedding, re_nodes_attention_in, re_nodes_attention_out


class NodePooling(nn.Module):
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding, eps=0, train_eps=True):
        super().__init__()
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        self.hid_param = Parameter(Tensor(pre_nodes, kern_size))
        self.re_param = Parameter(Tensor(kern_size, 1))
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        init.kaiming_uniform_(self.hid_param, nonlinearity='relu')
        self.eps.data.fill_(self.init_eps)

    def forward(self, t_mixed: Tensor, win_fuse_latent, nodes_embed: Tensor):
        # torch.Size([16, 64, 24, 56]) torch.Size([16, 24, 8]) torch.Size([16, 4, 24, 24])
        # temporal pooling
        t_mixed = t_mixed.transpose(1, 2)
        t_mixed_out = self.time_conv(t_mixed)
        t_mixed_out = t_mixed_out.transpose(1, 2)
        # latent relation mixed pooling
        win_fuse_latent_out = [self.time_conv(slot.unsqueeze(dim=1).transpose(1, 2)).transpose(1, 2).squeeze(dim=1)
                               for slot in win_fuse_latent]
        # node embedding vector pooling
        nodes_embed_out = nodes_embed.unsqueeze(dim=1).transpose(1, 2)
        nodes_embed_out = self.time_conv(nodes_embed_out)
        nodes_embed_out = nodes_embed_out.transpose(1, 2).squeeze(dim=1)
        return t_mixed_out, win_fuse_latent_out, nodes_embed_out


class NETFConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0, train_eps=True):
        super().__init__()
        self.mlp = GroupLinear(2*in_channels, out_channels, bias=False)
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def forward(self, x: Tensor, adj: Tensor):
        # x (Tensor): [B, C, N, F], adj (Tensor): [B, G, N, N]
        B, C, N, _ = x.size()
        S = adj.size(1)
        adj = norm(adj, add_loop=False).unsqueeze(dim=1)
        # x --> [B, C, S, N, F//S]
        x = x.reshape(B, C, N, S, -1).transpose(2, 3)
        out = torch.matmul(adj, x)
        # x_pre = x[:, :, :-1, ...]
        # out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        out = self.eps * x + out * (1 - self.eps)
        # torch.cat([x, out], dim=1)
        # # out: [B, C, G, N, F//G]
        # out = self.mlp(torch.cat([x, out], dim=1), True)
        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        return out