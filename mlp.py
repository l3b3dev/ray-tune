from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear (256) -> ReLU -> Dropout-> Linear(64) -> ReLU -> Dropout -> Linear(10) -> ReLU-> LogSoftmax
    """

    def __init__(self, l1=256, l2=64, dr=.25):
        super().__init__()
        self.fc1 = nn.Linear(784, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # Apply dropout
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)

        return x
