import torch


class RestrictedBoltzmanMachines():
    def __init__(self, num_visible_nodes, num_hidden_nodes):
        self.W = torch.randn(num_hidden_nodes, num_visible_nodes)
        self.a = torch.randn(1, num_hidden_nodes)  # bias for hidden nodes
        self.b = torch.randn(1, num_visible_nodes)  # bias for visiable nodes

    def sample_hidden_nodes(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        prob_hidden_given_visible = torch.sigmoid(activation)
        return prob_hidden_given_visible, torch.bernoulli(prob_hidden_given_visible)

    def sample_visible_nodes(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        prob_visible_given_hidden = torch.sigmoid(activation)
        return prob_visible_given_hidden, torch.bernoulli(prob_visible_given_hidden)

    def train(self, visible_0_iter, visible_k_iter, prob_hidden_0, prob_hidden_k):
        self.W += torch.mm(visible_0_iter.t(), prob_hidden_0) - torch.mm(visible_k_iter.t(), prob_hidden_k)
        self.b += torch.sum((visible_0_iter - visible_k_iter), 0)
        self.a += torch.sum((prob_hidden_0 - prob_hidden_k), 0)
