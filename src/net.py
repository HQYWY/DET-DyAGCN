from layer import *


def similarity_matrix(nodes_embedding_ls: list):
    re_simil_adj_ls = []
    for node_embedding in nodes_embedding_ls:
        # calculate sparse topk
        nodes = node_embedding.size(1)
        if nodes > 5:
            top_k = nodes // 2
        else:
            top_k = nodes
        # construct cosine similarity matrix
        x_normalized = F.normalize(node_embedding, p=2, dim=2)
        simil_matrix = torch.matmul(x_normalized, x_normalized.transpose(1, 2))
        # sparsification
        _, indices = torch.topk(simil_matrix, k=top_k, dim=-1)
        sparse_matrix = torch.zeros_like(simil_matrix)
        sparse_matrix.scatter_(-1, indices, 1)
        re_simil_adj_ls.append(sparse_matrix)
    return re_simil_adj_ls


class GNNStack(nn.Module):
    def __init__(self, slots, multi_heads, device, layers, pool_ratio,
                 seq_len, nodes, node_embed_dim, kern_size,
                 in_dim, hidden_dim, out_dim,
                 classes, dropout=0.5, activation=nn.ReLU()):
        super().__init__()

        # self.g_constr = MultiShallowEmbedding(nodes, nodes, slots)
        # remaining nodes
        self.left_num_nodes = []
        for layer in range(layers + 1):
            left_node = round(nodes * (1 - (pool_ratio * layer)))
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.activation = activation
        self.slots = slots
        self.node_embed_dim = node_embed_dim
        # initialize global static matrix
        self.nodes_embedding = Parameter(Tensor(nodes, node_embed_dim)).to(device)
        slot_length = (seq_len // slots)
        if multi_heads == 0:
            max_num_heads = min(slot_length // 2, slot_length)
            possible_num_heads = [i for i in range(1, max_num_heads + 1)]
            multi_heads = possible_num_heads[-1]
        # padding
        paddings = [(k - 1) // 2 for k in kern_size]

        # self.multi_head_attns = nn.ModuleList(
        #     [MultiHeadSlidlAtentModule(slot_length, multi_heads) for _ in range(layers)]
        # )

        self.t_convs = nn.ModuleList(
            [StackConv(1, in_dim // 2, (1, kern_size[0]),
                                 (self.left_num_nodes[0], kern_size[0]), (0, paddings[0]), (0, paddings[0]))] +
            [StackConv(in_dim, hidden_dim // 2, (1, kern_size[layer + 1]),
                                 (self.left_num_nodes[layer + 1], kern_size[layer + 1]), (0, paddings[layer + 1]),
                                 (0, paddings[layer + 1])) for layer in range(layers - 2)] +
            [StackConv(hidden_dim, out_dim // 2, (1, kern_size[-1]),
                                 (self.left_num_nodes[-2], kern_size[-1]), (0, paddings[-1]), (0, paddings[-1]))]
        )

        # self.t_convs = nn.ModuleList(
        #     [StackConv(1, in_dim // 2, (1, kern_size[0]),
        #                 (self.left_num_nodes[0], kern_size[0]), (0, paddings[0]), (0, paddings[0]))] +
        #     [StackConv(in_dim, hidden_dim // 2, (1, kern_size[1]),
        #                 (self.left_num_nodes[1], kern_size[1]), (0, paddings[1]),
        #                 (0, paddings[1]))] +
        #     [StackConv(hidden_dim, hidden_dim // 2, (1, kern_size[2]),
        #                 (self.left_num_nodes[2], kern_size[2]), (0, paddings[2]),
        #                 (0, paddings[2]))] +
        #     [StackConv(hidden_dim, out_dim // 2, (1, kern_size[-1]),
        #                 (self.left_num_nodes[-2], kern_size[-1]), (0, paddings[-1]), (0, paddings[-1]))]
        # )

        # self.t_convs = nn.ModuleList(
        #     [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] +
        #     [nn.Conv2d(in_dim, hidden_dim, (1, kern_size[layer + 1]), padding=(0, paddings[layer + 1])) for layer in range(layers - 2)] +
        #     [nn.Conv2d(hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        # )

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(in_dim)] +
            [nn.BatchNorm2d(hidden_dim) for _ in range(layers - 2)] +
            [nn.BatchNorm2d(out_dim)]
        )

        self.nodes_poolings = nn.ModuleList(
            [NodePooling(self.left_num_nodes[layer], self.left_num_nodes[layer + 1], kern_size[layer],
                          paddings[layer]) for layer in range(layers - 1)] +
            [NodePooling(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )

        self.gats = nn.ModuleList(
            [GAT(seq_len // slots, node_embed_dim, node_embed_dim, device) for _ in range(layers)]
        )

        self.netf_convs = nn.ModuleList(
            [NETFConvModule(in_dim, in_dim)] +
            [NETFConvModule(hidden_dim, hidden_dim) for _ in range(layers - 2)] +
            [NETFConvModule(out_dim, out_dim)]
        )

        self.dropout = dropout
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(out_dim, classes)
        self.reset_parameters()

    def reset_parameters(self):
        for gat, t_conv, netf_conv, nodes_pooling, bn in zip(self.gats, self.t_convs,
                                                             self.netf_convs, self.nodes_poolings, self.bns):
            # multi_head_attn.reset_parameters()
            nodes_pooling.reset_parameters()
            t_conv.reset_parameters()
            bn.reset_parameters()
            gat.reset_parameters()
            netf_conv.reset_parameters()
        self.linear.reset_parameters()
        init.xavier_uniform_(self.nodes_embedding)

    def forward(self, inputs: Tensor):
        # adj = self.g_constr(inputs.device)
        B, N, L = inputs.size()
        # sliding window stride
        window_size = L // self.slots
        stride = window_size
        # sliding latent relations
        win_fuse_latent = []
        for i in range(0, L - window_size + 1, stride):
            win_fuse_latent.append(inputs[:, :, i:i + window_size])
        t_inputs = inputs.unsqueeze(dim=1)
        nodes_embedding = self.nodes_embedding.repeat(B, 1, 1)
        for gat, t_conv, netf_conv, nodes_pooling, bn in zip(self.gats, self.t_convs,
                                                             self.netf_convs, self.nodes_poolings, self.bns):
            # win_fuse_latent = multi_head_attn(win_fuse_latent)
            # nodes_embedding=[[B,N,L//S],...] nodes_attention_in=[[B,N,N],...] nodes_attention_out=[[B,N,N],...]
            nodes_slots_embedding, _, re_nodes_attention = gat(win_fuse_latent, nodes_embedding)
            nodes_simil_adj = similarity_matrix(nodes_slots_embedding)
            node_attn_adj = torch.stack(re_nodes_attention, dim=1)
            t_c = t_conv(t_inputs)
            t_bn = bn(t_c)
            t_mixed = netf_conv(t_bn, F.relu(node_attn_adj+torch.stack(nodes_simil_adj, dim=1)))
            t_pooling, win_fuse_latent, nodes_embedding = nodes_pooling(
                t_mixed, win_fuse_latent, nodes_embedding)
            t_inputs = F.dropout(self.activation(t_pooling), p=self.dropout, training=self.training)
        out = self.global_pool(t_inputs).view(t_inputs.size(0), -1)
        out = self.linear(out)
        return out
