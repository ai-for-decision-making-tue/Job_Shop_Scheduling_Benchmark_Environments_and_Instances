import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        """
            The implementation of Operation Message Attention Block
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()

        """
        super(SingleOpAttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, op_mask):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        with shape [sz_b, N, 3]
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        Wh = torch.matmul(h, self.W)
        sz_b, N, _ = Wh.size()

        Wh_concat = torch.stack([Wh.roll(1, dims=1), Wh, Wh.roll(-1, dims=1)], dim=-2)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        Wh2_concat = torch.stack([Wh2.roll(1, dims=1), Wh2, Wh2.roll(-1, dims=1)], dim=-1)

        # broadcast add: [sz_b, N, 1, 1] + [sz_b, N, 1, 3]
        e = Wh1.unsqueeze(-1) + Wh2_concat
        e = self.leaky_relu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh_concat).squeeze(-2)

        return h_new


class MultiHeadOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Operation Message Attention Block with multi-head attention
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadOpAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [
            SingleOpAttnBlock(input_dim, output_dim, dropout_prob) for
            _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, op_mask):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :return: output feature vectors with shape
                [sz_b, N, num_heads * output_dim] (if concat == true)
                or [sz_b, N, output_dim]
        """
        h = self.dropout(h)

        # shape: [ [sz_b, N, output_dim], ... [sz_b, N, output_dim]]
        h_heads = [att(h, op_mask) for att in self.attentions]

        if self.concat:
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, N, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            # h.shape : [sz_b, N, output_dim]
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)


class SingleMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob):
        """
            The implementation of Machine Message Attention Block
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()
        """
        super(SingleMchAttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        # Wh.shape: [sz_b, M, out_features]
        # W_edge.shape: [sz_b, M, M, out_features]

        Wh = torch.matmul(h, self.W)
        W_edge = torch.matmul(comp_val, self.W_edge)

        # compute attention matrix

        e = self.get_attention_coef(Wh, W_edge)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mch_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def get_attention_coef(self, Wh, W_edge):
        """
            compute attention coefficients using node and edge features
        :param Wh: transformed node features
        :param W_edge: transformed edge features
        :return:
        """

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [sz_b, M, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:2 * self.out_features, :])  # [sz_b, M, 1]
        edge_feas = torch.matmul(W_edge, self.a[2 * self.out_features:, :])  # [sz_b, M, M, 1]

        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2) + edge_feas.squeeze(-1)

        return self.leaky_relu(e)


class MultiHeadMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Machine Message Attention Block with multi-head attention
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadMchAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads

        self.attentions = [SingleMchAttnBlock
                           (node_input_dim, edge_input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] (any i) corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape
                [sz_b, M, num_heads * output_dim] (if concat == true)
                or [sz_b, M, output_dim]
        """
        h = self.dropout(h)

        h_heads = [att(h, mch_mask, comp_val) for att in self.attentions]

        if self.concat:
            # h.shape : [sz_b, M, output_dim*num_heads]
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, M, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)
