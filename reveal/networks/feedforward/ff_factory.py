import networks.feedforward.*
structures = {
    2  : TwoLayerFF,
    3  : TwoLayerFF,
}

def factory(structure):
    return structures[len(structure)]
