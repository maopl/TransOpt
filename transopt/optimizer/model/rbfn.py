from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)


class RegressionDataset(Dataset):
    """create a dataset that complies with PyTorch """
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y


class RbfNet(nn.Module):
    def __init__(self, centers, beta):
        super(RbfNet, self).__init__()
        self.num_centers = centers.size(0)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(beta)
        self.linear = nn.Linear(self.num_centers, 1)
        nn.init.xavier_uniform_(self.linear.weight)

    def kernel_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdims=False).sqrt()))
        return C

    def forward(self, x):
        x = self.kernel_fun(x)
        x = self.linear(x)
        return x


class rbfn(object):
    def __init__(self, dataset, max_epoch=30, batch_size=5, lr=0.01, num_centers=5):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.num_centers = num_centers
        self.dim = dataset.inputs.shape[1]

        # create the DataLoader for training
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=1)

        # cluster
        self.centers = self.cluster()
        self.beta = self.calculate_beta()
        # create Rbf network
        self.model = RbfNet(self.centers, self.beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()
        self.avg_loss = 0

    def train(self):
        self.model.train()
        for epoch in range(self.max_epoch):
            self.avg_loss = 0
            total_batch = len(self.dataset) // self.batch_size

            for i, (input, output) in enumerate(self.data_loader):
                X = Variable(input.view(-1, self.dim))
                Y = Variable(output)

                self.optimizer.zero_grad()
                Y_prediction = self.model(X)
                loss = self.loss_fun(Y_prediction, Y)
                loss.backward()
                self.optimizer.step()
                self.avg_loss += loss / total_batch

            print("[Epoch: {:>4}] loss = {:>.9}".format(epoch + 1, self.avg_loss))
        print("[*] Training finished!")

    def predict(self, x):
        self.model.eval()
        x = torch.from_numpy(x)
        x = Variable(x)
        y = self.model(x)
        return y.data.numpy()

    def cluster(self):
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(self.dataset.inputs)
        centers = kmeans.cluster_centers_
        return torch.from_numpy(centers)

    def calculate_beta(self):
        r2 = torch.ones(1, self.num_centers)
        for i, center in enumerate(self.centers):
            distances = torch.linalg.norm(self.centers - center, axis=1)
            nearest_two_neighbors_indices = torch.argsort(distances)[:2]
            r2[0][i] = torch.sum(distances[nearest_two_neighbors_indices]**2) / 2
        beta = 1 / r2
        return beta

    def update_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=1)
