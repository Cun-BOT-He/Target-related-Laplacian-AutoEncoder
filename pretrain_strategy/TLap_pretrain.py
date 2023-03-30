def TLap_trainAE(model, trainloader, epochs, trainlayer, 
               gemb_lambda=0, lr=1e-1, weight_decay=0, plot_every=1):
    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr, weight_decay = weight_decay)
    gemb_dist = nn.PairwiseDistance(p=2)
    loss_x = nn.MSELoss()
    plot_loss = []

    for j in range(epochs):
        sum_loss = 0
        for X, y, X_neighbor, y_neighbor, weight in trainloader:
            y = y.view(-1).to(dev)
            unlabel_index = y<0
            label_index = y>=0
            Xu = X[unlabel_index]
            Xu_neighbor = X_neighbor[unlabel_index]
            yu_neighbor = y_neighbor[unlabel_index]
            Xu_weight = weight[unlabel_index]
            Xl = X[label_index]
            Xl_neighbor = X_neighbor[label_index]
            yl_neighbor = y_neighbor[label_index]
            Xl_weight = weight[label_index]
            
            Hidden, Hidden_reconst = model(X.to(dev), trainlayer, PreTrain=True)
            loss = 0            
            # 对无标签数据样本计算
            if Xu.shape[0]>0:
                yu_neighbor = yu_neighbor.view(-1).to(dev)
                Xu_unlabel_index = yu_neighbor.view(-1)<0
                Xu_label_index = yu_neighbor.view(-1)>0
                Xu_sample = torch.repeat_interleave(Xu,X_neighbor.shape[1],0)
                Xu_neighbor = Xu_neighbor.view(-1,Xu_neighbor.shape[-1])
                Xu_weight = Xu_weight.view(-1)
                Xu_sample_label = Xu_sample[Xu_label_index].to(dev)
                Xu_neighbor_label = Xu_neighbor[Xu_label_index].to(dev)
                Xu_weight_label = Xu_weight[Xu_label_index].to(dev)
                Xu_sample_unlabel = Xu_sample[Xu_unlabel_index].to(dev)
                Xu_neighbor_unlabel = Xu_neighbor[Xu_unlabel_index].to(dev)
                Xu_weight_unlabel = Xu_weight[Xu_unlabel_index].to(dev)
                    
                # 计算无标签样本的有标签近邻的 Loss
                Hidden_label, Hidden_label_reconst = model(Xu_sample_label, trainlayer, PreTrain=True)
                y_sample_pred_label = Hidden_label_reconst[:, -1]
                loss += gemb_lambda * torch.mul(gemb_dist(y_sample_pred_label.unsqueeze(-1), 
                                                         yu_neighbor[Xu_label_index].unsqueeze(-1).detach()),
                                               Xu_weight_label).sum()
                # 计算无标签样本的无标签近邻的 Loss
                Hidden_unlabel, Hidden_unlabel_reconst = model(Xu_sample_unlabel, trainlayer, PreTrain=True)
                yu_sample_pred_unlabel = Hidden_unlabel_reconst[:, -1]
                neighbor_Hidden_unlabel, neighbor_Hidden_unlabel_reconst = model(Xu_neighbor_unlabel,
                                                                                 trainlayer, PreTrain=True)
                yu_neighbor_pred_unlabel = neighbor_Hidden_unlabel_reconst[:, -1]
                loss += gemb_lambda * torch.mul(gemb_dist(yu_sample_pred_unlabel.unsqueeze(-1),
                                                          yu_neighbor_pred_unlabel.unsqueeze(-1)), Xu_weight_unlabel).sum()
            # 对有标签数据样本进行计算
            if Xl.shape[0]>0:
                Xl_sample = torch.repeat_interleave(Xl,X_neighbor.shape[1],0).to(dev)
                yl_sample = torch.repeat_interleave(y[label_index],y_neighbor.shape[1],0).to(dev)
                neighbor_Hidden, neighbor_Hidden_reconst = model(Xl_neighbor.view(-1,X_neighbor.shape[-1]).to(dev),
                                                                 trainlayer, PreTrain=True)
                yl_neighbor_pred = neighbor_Hidden_reconst[:, -1]
                loss += gemb_lambda * torch.mul(gemb_dist(yl_sample.unsqueeze(-1).detach(),
                                                            yl_neighbor_pred.unsqueeze(-1)), 
                                                        Xl_weight.view(-1).to(dev)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        if j % plot_every == 0:
            plot_loss.append(sum_loss/len(trainloader))
        print('{} ({}, {}%)'.format(timeSince(start,(j+1)/epochs), (j+1), (j+1) / epochs * 100),'无监督预训练第{}层的第{}个epoch:'.format(trainlayer+1, j + 1), ',Loss:{}'.format(sum_loss/len(trainloader)))

    return model, plot_loss