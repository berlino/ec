import random
import torch
from torch.utils.data import DataLoader
from neural_seq.larcDataset import *

def train_imitiation_learning(model, train_loader, test_loader, batch_size, lr, weight_decay, num_epochs, earlyStopping=True):

    print("Training for {} epochs on {} batches of size {}".format(num_epochs, len(train_loader), batch_size))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    epoch_train_scores = []
    test_scores = []

    n_epochs_stop = 10
    if earlyStopping:
        ep = EarlyStopping(patience=10, n_epochs_stop=n_epochs_stop, init_best_val_loss=float('INF'))
        assert len(test_dataset) > 0
    
    # Imitation learning training
    for epoch in range(num_epochs):
        
        epoch_score = 0.0
        
        for i,batch in enumerate(train_loader):

            print("Epoch: {}, Batch: {}".format(epoch, i))
            # the sequence will always be the ground truth since we run forward in "score" mode
            scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="score", targets=batch['program'])
            weighted_scores = torch.dot(scores, batch["program_weight"])
            batch_score = (weighted_scores / batch_size)
            epoch_score += batch_score

            batch_score.backward()
            optimizer.step()
            optimizer.zero_grad()
            

        epoch_score = epoch_score / len(train_loader)
        epoch_train_scores.append(epoch_score)
        print("Training score at epoch {}: {}".format(epoch, epoch_score))

    
    if earlyStopping:
        # Get test set performance
        if epoch % 5 == 0:

            test_score = 0.0
            num_batches = 0
            for batch in test_loader:
                # the sequence will always be the ground truth since we run forward in "score" mode
                token_sequences, scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="score", targets=batch['programs'])
                batch_score = (torch.sum(scores) / test_batch_size)
                test_score += batch_score
                num_batches += 1

            epoch_test_score = test_score / num_batches
            print("Test score at epoch {}: {}".format(epoch, epoch_test_score))
            
            test_scores.append(test_score / num_batches)

            if earlyStopping:
                shouldStop, bestModel = ep.should_stop(epoch, epoch_test_score, model)
                if shouldStop:
                    print("Holdout loss stopped decreasing after {} epochs".format(n_epochs_stop))
                    return bestModel, epoch_train_scores, test_scores

    return model, epoch_train_scores, test_scores

def getKfoldSplit(taskNames, trainRatio, k):
    
    totalNumTasks = len(set(taskNames))
    numTrain = int(trainRatio * totalNumTasks)

    for i in range(k):
        trainTaskNames = random.sample(taskNames, numTrain)
        testTaskNames = list(set(taskNames).difference(trainTaskNames))

        yield trainTaskNames, testTaskNames

def train_experience_replay(model, larc_train_dataset, num_epochs, lr, weight_decay, batch_size):

    train_loader = DataLoader(larc_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: collate(x, True), drop_last=False)

    model, epoch_train_scores, test_scores = train_imitiation_learning(model, train_loader, test_loader=None, batch_size=batch_size, 
        lr=lr, weight_decay=weight_decay, num_epochs=num_epochs, earlyStopping=False)

    return model


    # model = train_imitiation_learning(model, train_loader, test_loader, batch_size=batch_size, lr=1e-3, weight_decay=0.0, num_epochs=3)


