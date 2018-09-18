import reveal.parameters.constants as consts
import reveal.networks.feedforward.ff_factory

def resolve_network_factory(network_type):
    if network_type == consts.NETWORK_TYPE_FEEDFORWARD:
        return reveal.networks.feedforward.ff_factory.factory
