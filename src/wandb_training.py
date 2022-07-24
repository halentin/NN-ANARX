import wandb
import scipy.io
import torch
from torch import nn
from utilities import lag_matrix
from NN_ANARX.ANARX import ANARX
import matplotlib.pyplot as plt
from tqdm import tqdm


""" This script is used to perform distributed training and analysis of NN-ANARX-models using Weights and Biases.
    It was adapted from a WandB example-script.
    It contains data-preparation, model-creation, open-loop-training, closed-loop-training and logging.
    The quality of a model is assessed according to the MSE-Loss of its closed-loop-prediction on validation data.
    Thats why closed-loop-validation loss is logged every 100th step of open-loop-training.
"""

def main():
    config = dict(
        epochs=10,
        epochs_cl = 10,
        n_hidden = 3,
        layersize = 10,
        afunc = "tanh",
        bias = True, 
        y_lags = 15,
        u1_lags = 10,
        u2_lags = 10,
        batch_size=100,
        learning_rate=1e-3,
        architecture="ANARX")
    
    model_pipeline(config)





def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

      # make the model, data, and optimization problem
        model, train_loader, train_data, valid_data, criterion, optimizer = make(config)
        torch.save(model, 'model.pt')
        print(model)

      # and use them to train the model
        train(model, train_loader, train_data, valid_data, criterion, optimizer, config)

    

    return model


def make(config):
    # Make the data
    data = scipy.io.loadmat("data/1803")
    u1_t, u1_v =  torch.Tensor(data["u1_t"]).squeeze(),  torch.Tensor(data["u1_v"]).squeeze()
    u2_t, u2_v =  torch.Tensor(data["u2_t"]).squeeze(),  torch.Tensor(data["u2_v"]).squeeze()
    y_t, y_v =  torch.Tensor(data["y_t"]).squeeze(),  torch.Tensor(data["y_v"]).squeeze()

    train_ds = torch.utils.data.TensorDataset(lag_matrix(u1_t, config.u1_lags), lag_matrix(u2_t, config.u2_lags), lag_matrix(y_t, config.y_lags), y_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = config.batch_size, shuffle=False)

    valid_data = [u1_v, u2_v, y_v]
    train_data = [u1_t, u2_t, y_t]
    # Make the model
    if config.afunc == "tanh":
        afunc = torch.tanh
    elif config.afunc == "relu":
        afunc = torch.relu
    else:
        afunc = torch.relu
    
    model = ANARX(config.y_lags, [config.u1_lags, config.u2_lags], n_hidden=config.n_hidden, layersize=config.layersize, afunc=afunc, bias=config.bias)

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, train_data, valid_data, criterion, optimizer

def train(model, loader, train_data, valid_data, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
     # number of examples seen
    batch_ct = 0
    # open loop training
    for epoch in tqdm(range(config.epochs)):
        for _, (u1_l, u2_l, y_l, y) in enumerate(loader):

            loss = train_batch(u1_l, u2_l, y_l, y, model, optimizer, criterion)
            batch_ct += 1

        if (epoch % 10 == 0):
            train_log(epoch, loss)
        if (epoch % 100 == 0):
            log_closed_loop_loss(epoch, model, criterion, valid_data, train_data)
    # closed loop training
    for epoch_cl in tqdm(range(config.epochs_cl)):
        log_epoch = epoch_cl+config.epochs+1
        prediction = model.predict([train_data[0], train_data[1]])
        loss = criterion(prediction, train_data[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch_cl % 10 == 0):
            prediction_valid = model.justpredict([valid_data[0], valid_data[1]])
            cl_valid_loss = criterion(prediction_valid, valid_data[2])
            wandb.log({"epoch":log_epoch, "cl_valid_loss": cl_valid_loss, "cl_train_loss": loss}, step=log_epoch)
            plt.plot(prediction_valid)
            plt.plot(valid_data[2])
            plt.ylabel("Model Prediction vs Output on Valid Data")
            wandb.log({"Valid Chart": plt})
            plt.plot(prediction.detach().numpy())
            plt.plot(train_data[2])
            plt.ylabel("Model Prediction vs Output on Train Data")
            wandb.log({"Train Chart": plt})
        else:
            wandb.log({"epoch": log_epoch, "cl_valid_loss": cl_valid_loss, "cl_train_loss": loss}, step=log_epoch)


def train_batch(u1_l, u2_l, y_l, y, model, optimizer, criterion):
    # Forward pass ➡
    outputs = model(y_l, [u1_l, u2_l])
    loss = criterion(outputs, y)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(epoch, loss):
    # Where the magic happens

    wandb.log({"epoch": epoch, "ol_train_loss": loss}, step=epoch)

def log_closed_loop_loss(epoch, model, criterion, valid_data, train_data):
    prediction_valid = model.justpredict([valid_data[0], valid_data[1]])
    prediction_train = model.justpredict([train_data[0], train_data[1]])
    cl_valid_loss = criterion(prediction_valid, valid_data[2])
    cl_train_loss = criterion(prediction_train, train_data[2])
    wandb.log({"epoch": epoch, "cl_valid_loss": cl_valid_loss, "cl_train_loss": cl_train_loss}, step=epoch)
    plt.plot(prediction_valid)
    plt.plot(valid_data[2])
    plt.ylabel("Model Prediction vs Output on Valid Data")
    wandb.log({"Valid Chart": plt})
    plt.plot(prediction_train)
    plt.plot(train_data[2])
    plt.ylabel("Model Prediction vs Output on Train Data")
    wandb.log({"Train Chart": plt})



if __name__ == "__main__":
    main()