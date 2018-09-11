import torch.nn.functional as F
import data_utils.data_processors as dp

def compareActivations(X, T, list_of_activations, n_iterations, batch_size):

    net = TwoLayerFF([X.shape[1],10,10,T.shape[1]], F.relu)
    print(net)

    X = torch.from_numpy(dp.normalize(X))
    T = torch.from_numpy(dp.normalize(T))
    X,T = X.type(torch.FloatTensor), T.type(torch.FloatTensor)

    nSamples = X.size()[0]

    for epoch in range(n_iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(0, nSamples, batch_size):
            if nSamples - i < batch_size:
                end = nSamples
            else:
                end = i + batch_size

            X_iter = X[i:end,:].unsqueeze(0)
            T_iter = T[i:end,:].unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, act = net(X_iter)
            loss = criterion(outputs, T_iter)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(running_loss)
        running_loss = 0.0

    print('Finished Training')


    print('beginning testing')
