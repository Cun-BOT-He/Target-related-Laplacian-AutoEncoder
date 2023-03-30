import time
from utils import timeSince

# 单层自编码器训练函数
def ss_trainTAE(model, trainloader, epochs, trainlayer, use_label=False, 
               s_lambda=0, lr=1e-1, weight_decay=0, plot_every=1):

    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr)
    loss_func_x = nn.MSELoss()
    loss_func_y = nn.MSELoss()
    plot_loss = []

    start = time.time()
    for j in range(epochs):
        sum_loss = 0
        for X, y in trainloader:
            if use_label is True:
                Hidden, Hidden_reconst = model(X.to(dev), trainlayer, PreTrain=True)
                y = y.squeeze()
                y_pred = Hidden_reconst[:, -1]
                Hidden_reconst = Hidden_reconst[:, :-1]
                loss = loss_func_x(Hidden, Hidden_reconst) + s_lambda * loss_func_y(y.to(dev), y_pred)
            else:
                Hidden, Hidden_reconst = model(X.float().to(dev), trainlayer, PreTrain=True)
                loss = loss_func_x(Hidden, Hidden_reconst[:, :-1])                

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        if j % plot_every == 0:
            plot_loss.append(sum_loss/len(trainloader))
        print('{} ({}, {}%)'.format(timeSince(start,(j+1)/epochs), (j+1), (j+1) / epochs * 100),'无监督预训练第{}层的第{}个epoch:'.format(trainlayer+1, j + 1), ',Loss:{}'.format(sum_loss/len(trainloader)))

    return model, plot_loss
    
    
    
def ss_trainAE(model, trainloader, epochs, trainlayer, 
               s_lambda=0, lr=1e-1, weight_decay=0, plot_every=1):

    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr)
    loss_func_x = nn.MSELoss()
    loss_func_y = nn.MSELoss()
    plot_loss = []

    for j in range(epochs):
        sum_loss = 0
        for X, y in trainloader:
            Hidden, Hidden_reconst = model(X.to(dev), trainlayer, PreTrain=True)
            loss = loss_func_x(Hidden, Hidden_reconst) + s_lambda * loss_func_y(y.to(dev), y_pred)               

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        if j % plot_every == 0:
            plot_loss.append(sum_loss/len(trainloader))
        print('{} ({}, {}%)'.format(timeSince(start,(j+1)/epochs), (j+1), (j+1) / epochs * 100),'无监督预训练第{}层的第{}个epoch:'.format(trainlayer+1, j + 1), ',Loss:{}'.format(sum_loss/len(trainloader)))

    return model, plot_loss