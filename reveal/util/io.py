import pickle

def save(filePath, object):
    with open(filePath, 'wb') as f:
        pickle.dump(object, f)

def load(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)
