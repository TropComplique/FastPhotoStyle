import torch


def wct(content, style):
    """
    Arguments:
        content: a float tensor with shape [d, h * w].
        style: a float tensor with shape [d, h' * w'].
    Returns:
        a float tensor with shape [d, h * w].
    """

    _, area = content.size()
    # area is equal to h * w

    mean = torch.mean(content, 1, keepdim=True)  # shape [d, 1]
    content -= mean

    covariance = torch.mm(content, content.t()).div(area - 1) + torch.eye(c)  # shape [d, d]
    _, D_c, E_c = torch.svd(covariance, some=False)
    # they have shapes [d] and [d, d]

    k_c = (D_c >= 1e-5).long().sum()
    E_c = E_c[:, 0:k_c]  # shape [d, k_c]

    x = torch.diag(D_c[0:k_c].pow(-0.5))
    x = torch.mm(E_c, x)
    x = torch.mm(x, E_c.t())  # shape [d, d]
    whitened_content = torch.mm(x, content)  # shape [d, h * w]

    mean = torch.mean(style, 1, keepdim=True)  # shape [d, 1]
    style -= mean

    _, area = style.size()
    # area is equal to h' * w'

    covariance = torch.mm(style, style.t()).div(area - 1)
    _, D_s, E_s = torch.svd(covariance, some=False)
    # they have shapes [d] and [d, d]

    k_s = (D_s >= 1e-5).long().sum()
    E_s = E_s[:, 0:k_s]  # shape [d, k_s]

    x = torch.diag(D_s[0:k_s].pow(0.5))
    x = torch.mm(E_s, x)
    x = torch.mm(x, E_s.t())  # shape [d, d]
    colored_content = torch.mm(x, whitened_content)  # shape [d, h * w]
    colored_content += mean

    return colored_content
