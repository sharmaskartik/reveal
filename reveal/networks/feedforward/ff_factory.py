import reveal.networks.feedforward

structures = {
    2  : reveal.networks.feedforward.TwoLayerFF,
    3  : reveal.networks.feedforward.ThreeLayerFF,
}

def factory(structure):
    return structures[len(structure)]
