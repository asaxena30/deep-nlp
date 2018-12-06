from torch.nn.init import xavier_normal_


def xavier_normal_weight_init(named_parameters) -> None:
    for name, param in named_parameters:
        if 'weight' in name:
            xavier_normal_(param)

