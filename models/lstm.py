import copy
import matplotlib.pyplot as plt

from IPython.display import clear_output

import torch
import torch.utils.data as data
import torch, torch.nn as nn
import torch.nn.functional as F


class LSTM():
    def __init__(self, window_size, hidden_size=64, batch_size=2, output_size=2): 
        self.window_size = window_size
        self.batch_size = batch_size
        self.model = CharLSTMLoop(input_size=self.window_size, output_size=output_size, hidden_size=hidden_size)

    def init_data(self, X_train, y_train, X_val, y_val):
        X_train = X_train.reshape(-1, self.window_size, 8)
        y_train = y_train.reshape(-1, self.window_size, 2)

        X_val = X_val.reshape(-1, self.window_size, 8)
        y_val = y_val.reshape(-1, self.window_size, 2)

        train_data = torch.FloatTensor(X_train)
        train_pred = torch.FloatTensor(y_train)
        val_data = torch.FloatTensor(X_val)
        val_pred = torch.FloatTensor(y_val)
        self.train_loader = self.create_dataloader(train_data, train_pred, self.batch_size, shuffle=True)
        self.val_loader = self.create_dataloader(val_data, val_pred, self.batch_size, shuffle=False)

    def create_dataloader(self, inputs, labels, batch_size, shuffle):
        loader = data.DataLoader(data.TensorDataset(inputs, labels), shuffle=shuffle, batch_size=batch_size, drop_last=True)
        return loader


    def predict(self, batch):
        self.model.eval()
        preds, _, _ = self.model(batch)
        preds = preds.detach()

        return preds

    def train(self, verbose=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_fn = nn.MSELoss()

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model = self.model.to(device)
        train_history = []
        val_history = []

        best_loss = 5252525

        best_model_wts = copy.deepcopy(self.model.state_dict())
        EPOCHS = 100

        for epoch in range(EPOCHS):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch.to(device)
                preds, _, _ = self.model(X_batch)

                loss = loss_fn(preds, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.cpu().data.numpy()

            epoch_loss /= len(self.train_loader)
            train_history.append(epoch_loss)

            # Validation
            self.model.eval()
            epoch_loss = 0
            for X_batch, y_batch in self.val_loader:
                y_pred, _, _ = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_loss += loss.cpu().data.numpy()
            epoch_loss /= len(self.val_loader)

            if epoch_loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

            val_history.append(epoch_loss)
            if verbose:
                clear_output(True)
                plt.plot(train_history, label='train_loss')
                plt.plot(val_history, label='val_loss')
                plt.legend()
                plt.show()
        self.model.load_state_dict(best_model_wts)


class CharLSTMLoop(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
        super(self.__class__, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.hid_to_target = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None, c=None):
        if h is not None and c is not None:
            out_put, (h_new, c_new) = self.LSTM(x, (h, c))
        else:
            out_put, (h_new, c_new) = self.LSTM(x)
            
        next_logits = self.hid_to_target(out_put)
        
        return next_logits, h_new, c_new