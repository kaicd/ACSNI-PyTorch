from torch import nn


class auto_encoder(nn.Module):
    def __init__(self, nc, __a):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nc, __a),
            nn.ReLU(),
            nn.Linear(__a, __a // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(__a // 2, __a),
            nn.Sigmoid(),
            nn.Linear(__a, nc),
        )

    def forward(self, data):
        out = self.encoder(data)
        out = self.decoder(out)
        return out
