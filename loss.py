import torch.nn as nn


class CosineLoss(nn.Module):

    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, input, target):
        if len(input.size()) > 1:
            loss = 0
            for i in range(input.size()[0]):
                loss += 1.0 - input[i][:].dot(target[i][:]) / (input[i][:].norm()*target[i][:].norm())
            loss /= input.size()[0]
            return loss
        else:
            return 1.0 - input.dot(target) / (input.norm()*target.norm())


def loss_from_name(name: str):
    if name == 'BCE':
        loss= nn.BCEWithLogitsLoss(reduction='elementwise_mean')
        return loss
    elif name == 'Cosine':
        loss = CosineLoss()
        return loss
    elif name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(reduction='elementwise_mean')
        return loss
    else:
        raise NotImplementedError
