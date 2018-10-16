import reveal.networks.feedforward

structures = {
    2  : reveal.networks.feedforward.TwoLayerFF,
    3  : reveal.networks.feedforward.ThreeLayerFF,
    4  : reveal.networks.feedforward.FourLayerFF,
    5  : reveal.networks.feedforward.FiveLayerFF
}

def factory(structure):
    return structures.get(len(structure), None)
