from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor, nn


class MNRLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, tau=None, norm_dim=1, use_rescale=False, mu=1):
        super(MNRLoss, self).__init__()
        self.model = model
        self.tau = tau
        self.norm_dim = norm_dim
        self.use_rescale = use_rescale
        self.mu = mu

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.tau is None:
            alpha = self.model._first_module().alpha
            # print('tau: {}'.format(1 / self.model._first_module().alpha), flush=True)
        else:
            alpha = 1 / self.tau

        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        rep_a, rep_b = reps[0], reps[1]

        if self.norm_dim == 1:  # or self.norm_dim == 0
            rep_an = rep_a / torch.norm(rep_a, p=2, dim=self.norm_dim, keepdim=True)
            rep_bn = rep_b / torch.norm(rep_b, p=2, dim=self.norm_dim, keepdim=True)
        elif self.norm_dim == 0:
            rep_an = rep_a / (rep_a.max(axis=0)[0] - rep_a.min(axis=0)[0])
            rep_bn = rep_b / (rep_b.max(axis=0)[0] - rep_b.min(axis=0)[0])
        else:
            rep_an = rep_a
            rep_bn = rep_b

        if labels is not None:
            dot_pr_mat = torch.mm(rep_an, rep_bn.T) * alpha

            if self.use_rescale:
                labels = labels**2 / (labels**2).sum()

            loss_mnr = -torch.mean(
                torch.diag(F.log_softmax(dot_pr_mat, dim=1)) * labels.float()
            ) - torch.mean(torch.diag(F.log_softmax(dot_pr_mat, dim=0)) * labels.float())
            loss_mnr = loss_mnr / 2
            return loss_mnr
        else:
            output = torch.cosine_similarity(rep_an, rep_bn)
            return reps, output
