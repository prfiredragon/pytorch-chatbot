import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

class IntentDLModel(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dim: int,
        padding_idx: int,
        n_hidden_layer: int,
        n_hidden_layer_neurons: int,
        n_classes: int,
    ) -> None:
        super(IntentDLModel, self).__init__()
        self.embedding_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=n_embeddings,
                embedding_dim=n_embedding_dim,
                padding_idx=padding_idx,
            ),
            nn.Dropout(0.10, inplace=True),
            nn.Linear(in_features=n_embedding_dim, out_features=n_hidden_layer_neurons),
            nn.ReLU(inplace=True),
        )
        self.linear_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 9216),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.05),
                )
                for _ in range(n_hidden_layer)
            ]
        )
        self.classifier = nn.Linear(9216, n_classes)

    def forward(self, token_type_ids, tag=None):
        out = self.embedding_layer(token_type_ids)
        out = out.view(-1, out.size(1) * out.size(2))
        out = self.classifier(out)
        return out


class IntentRnnModel(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        n_embedding_dim: int,
        padding_idx: int,
        n_hidden_layer: int,
        n_hidden_layer_neurons: int,
        n_classes: int,
    ) -> None:
        super(IntentRnnModel, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=n_embedding_dim,
            padding_idx=padding_idx,
        )
        self.lstm_layer = nn.LSTM(
            input_size=n_embedding_dim,
            hidden_size=n_hidden_layer_neurons,
            batch_first=True,
            dropout=0.10,
        )
        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=9216, out_features=1024),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, token_type_ids, tag=None):
        embeds = self.embedding_layer(token_type_ids)
        lstm_out, hidden = self.lstm_layer(embeds)
        out = self.dense_layer(
            torch.clone(lstm_out.reshape(-1, lstm_out.size(1) * lstm_out.size(2)))
        )
        out = self.classifier(torch.clone(out))
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.device = device

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        #print(h0.shape)
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
