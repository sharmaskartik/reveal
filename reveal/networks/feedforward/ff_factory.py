import reveal.networks.feedforward

structures = {
    2  : reveal.networks.feedforward.TwoLayerFF,
    3  : reveal.networks.feedforward.ThreeLayerFF,
    4  : reveal.networks.feedforward.FourLayerFF,
    5  : reveal.networks.feedforward.FiveLayerFF
}

def factory(structure):
    net_class = structures.get(len(structure), None)

    if net_class is None:
        raise Exception("Network for the network structure :", structure,
         " has not been defined in the package")

    return net_class
