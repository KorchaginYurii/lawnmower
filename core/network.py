import torch
import torch.nn as nn

from core.config import DEVICE, ACTIONS, VISION_SIZE, STATE_CHANNELS

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(STATE_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1,STATE_CHANNELS,VISION_SIZE,VISION_SIZE)
            out = self.conv(dummy)
            self.conv_out = out.view(1,-1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_out, 256),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(256, 4)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        policy_logits = self.policy_head(x)
        v = torch.tanh(self.value_head(x))

        return policy_logits, v

#        pi = torch.softmax(self.policy_head(x), dim=1)
#        v = torch.tanh(self.value_head(x))

#        return pi, v