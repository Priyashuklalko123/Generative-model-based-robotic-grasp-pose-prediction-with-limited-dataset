def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ginnet':
        from .GINNet import GenerativeInceptionNN
        return GenerativeInceptionNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
