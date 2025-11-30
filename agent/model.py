import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_stack, n_actions):
        super(DQN, self).__init__()

        # Shape 4 x 84 x 84 -> 4

        self.network = nn.Sequential(
            nn.Conv2d(num_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.network(x)

class DuellingDQN(nn.Module):
    def __init__(self, num_stack, n_actions):
        super(DuellingDQN, self).__init__()

        # Shape 4 x 84 x 84 -> 4

        self.conv1 = nn.Conv2d(num_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Value stream            
        self.fc_v1 = nn.Linear(3136, 512)
        self.fc_v2 = nn.Linear(512, 1)

        # Action stream            
        self.fc_a1 = nn.Linear(3136, 512)
        self.fc_a2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        # Value Stream
        v = F.relu(self.fc_v1(x))
        v = self.fc_v2(v)

        # Action Stream
        a = F.relu(self.fc_a1(x))
        a = self.fc_a2(a)

        # Mean instead of max for gradient smoothing
        return v + a - a.mean(dim=1, keepdim=True)

class C51DQN(nn.Module):
    def __init__(self, num_stack, n_actions, n_atoms=51):
        super(C51DQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        # Convolutions
        self.conv1 = nn.Conv2d(num_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions * n_atoms)

    def forward(self, x, log=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape [batch, n_actions, n_atoms]
        x = x.view(-1, self.n_actions, self.n_atoms)

        if log:
            return F.log_softmax(x, dim=2)
        
        return F.softmax(x, dim=2)
