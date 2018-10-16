import reveal.networks.feedforward

structures = {
    2  : reveal.networks.feedforward.TwoLayerFF,
    3  : reveal.networks.feedforward.ThreeLayerFF,
    4  : reveal.networks.feedforward.FourLayerFF
}

def factory(structure):
    return structures[len(structure)]
