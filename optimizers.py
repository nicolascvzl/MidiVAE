from torch import optim


def optimizer_from_name(name: str, **kwargs):
    if name == 'adam':
        return lambda params: optim.Adam(params, **kwargs)
    elif name == 'sgd':
        return lambda params: optim.SGD(params, **kwargs)
    else:
        raise NotImplementedError
