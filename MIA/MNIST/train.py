import torch
import numpy as np
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
import matplotlib.pyplot as plt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, data_loader, test_loader, optimizer, criterion, n_epochs, classes=None, verbose=False):
    losses = []
    for epoch in range(n_epochs):
        net.train()
        for i, batch in enumerate(data_loader):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if verbose:
                print("[%d/%d][%d/%d] loss = %f" % (epoch, n_epochs, i, len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch
        print("[%d/%d]" %(epoch, n_epochs))
        print("Training:")
        train_accuracy = eval_target_net(net, data_loader, classes=classes)
        print("Test:")
        test_accuracy = eval_target_net(net, test_loader, classes=classes)
        #plt.plot(losses)
        #plt.show()
    return train_accuracy, test_accuracy


def eval_target_net(net, testloader, classes=None):

    if classes is not None:
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
    total = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (imgs, lbls) in enumerate(testloader):

            imgs, lbls = imgs.to(device), lbls.to(device)

            output = net(imgs)

            predicted = output.argmax(dim=1)

            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()

            if classes is not None:
                for prediction, lbl in zip(predicted, lbls):

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1
                    
    accuracy = 100*(correct/total)
    if classes is not None:
        for i in range(len(classes)):
            print('Accuracy of %s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("\nAccuracy = %.2f %%\n\n" % (accuracy) )
    
    return accuracy


def train_attacker(attack_net, shadow, shadow_train, shadow_out, optimizer, criterion, n_epochs, k):
    
    """
    Trains attack model (classifies a sample as in or out of training set) using
    shadow model outputs (probabilities for sample class predictions). 
    The type of shadow model used can vary.
    """
        
    in_predicts=[]
    out_predicts=[]
    losses = []
    
    if type(shadow) is not Pipeline:
        shadow_net=shadow
        shadow_net.eval()

    for epoch in range(n_epochs):
       
        total = 0
        correct = 0

        #train_top = np.array([])
        #train_top = []
        train_top = np.empty((0,1))
        out_top = np.empty((0,1))
        for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(shadow_train, shadow_out)):

            if train_imgs.shape[0] != out_imgs.shape[0]: 
                break
                
            #######out_imgs = torch.randn(out_imgs.shape)
            mini_batch_size = train_imgs.shape[0]
            
            if type(shadow) is not Pipeline:
                train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)

                train_posteriors = F.softmax(shadow_net(train_imgs.detach()), dim=1)
                
                out_posteriors = F.softmax(shadow_net(out_imgs.detach()), dim=1)

                
            else:
                traininputs= train_imgs.view(train_imgs.shape[0],-1)
                outinputs=out_imgs.view(out_imgs.shape[0], -1)
                
                in_preds=shadow.predict_proba(traininputs)
                train_posteriors=torch.from_numpy(in_preds).float()
                #for p in in_preds:
                 #   in_predicts.append(p.max())
                
                out_preds=shadow.predict_proba(outinputs)
                out_posteriors=torch.from_numpy(out_preds).float()
                #for p in out_preds:
                 #   out_predicts.append(p.max())
                            

            train_sort, _ = torch.sort(train_posteriors, descending=True)
            train_top_k = train_sort[:,:k].clone().to(device)
            for p in train_top_k:
                in_predicts.append((p.max()).item())
            out_sort, _ = torch.sort(out_posteriors, descending=True)
            out_top_k = out_sort[:,:k].clone().to(device)
            for p in out_top_k:
                out_predicts.append((p.max()).item())

            train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))


            train_lbl = torch.ones(mini_batch_size).to(device)
            out_lbl = torch.zeros(mini_batch_size).to(device)

            optimizer.zero_grad()

            train_predictions = torch.squeeze(attack_net(train_top_k))
            out_predictions = torch.squeeze(attack_net(out_top_k))

            print(train_predictions)
            print(train_lbl)
            loss_train = criterion(train_predictions, train_lbl)
            loss_out = criterion(out_predictions, out_lbl)

            loss = (loss_train + loss_out) / 2
            
            if type(shadow) is not Pipeline:
                loss.backward()
                optimizer.step()

            
            correct += (F.sigmoid(train_predictions)>=0.5).sum().item()
            correct += (F.sigmoid(out_predictions)<0.5).sum().item()
            total += train_predictions.size(0) + out_predictions.size(0)


            print("[%d/%d][%d/%d] loss = %.2f, accuracy = %.2f" % (epoch, n_epochs, i, len(shadow_train), loss.item(), 100 * correct / total))
            
        #Plot distributions for target predictions in training set and out of training set
        """
        fig, ax = plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.hist(in_predicts, bins='auto')
        plt.title('In')
        plt.subplot(2,1,2)
        plt.hist(out_predicts, bins='auto')
        plt.title('Out')
        """

        '''
        plt.scatter(out_top.T[0,:], out_top.T[1,:], c='b')
        plt.scatter(train_top.T[0,:], train_top.T[1,:], c='r')
        plt.show()
        '''        

def eval_attack_net(attack_net, target, target_train, target_out, k):
    """Assess accuracy, precision, and recall of attack model for in training set/out of training set classification.
    Edited for use with SVCs."""
    
    in_predicts=[]
    out_predicts=[]
    losses = []
    
    if type(target) is not Pipeline:
        target_net=target
        target_net.eval()
        
    attack_net.eval()

    
    precisions = []
    recalls = []
    accuracies = []

    #for threshold in np.arange(0.5, 1, 0.005):
    thresholds = np.arange(0.5, 1, 0.005)

    total = np.zeros(len(thresholds))
    correct = np.zeros(len(thresholds))

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds))   
 
    train_top = np.empty((0,1))
    out_top = np.empty((0,1))
    
    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)
        
        #[mini_batch_size x num_classes] tensors, (0,1) probabilities for each class for each sample)
        if type(target) is Pipeline:
            traininputs=train_imgs.view(train_imgs.shape[0], -1)
            outinputs=out_imgs.view(out_imgs.shape[0], -1)
            
            train_posteriors=torch.from_numpy(target.predict_proba(traininputs)).float()
            out_posteriors=torch.from_numpy(target.predict_proba(outinputs)).float()
            
        else:
            train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
            out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)
        

        #[k x mini_batch_size] tensors, (0,1) probabilities for top k probable classes
        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:,:k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:,:k].clone().to(device)
        
        #Collects probabilities for predicted class.
        for p in train_top_k:
            in_predicts.append((p.max()).item())
        for p in out_top_k:
            out_predicts.append((p.max()).item())
        
        if type(target) is not Pipeline:
            train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))

        #print("train_top_k = ",train_top_k)
        #print("out_top_k = ",out_top_k)
        
        #print(train_top.shape)
        
        train_lbl = torch.ones(mini_batch_size).to(device)
        out_lbl = torch.zeros(mini_batch_size).to(device)
        
        #Takes in probabilities for top k most likely classes, outputs ~1 (in training set) or ~0 (out of training set)
        train_predictions = F.sigmoid(torch.squeeze(attack_net(train_top_k)))
        out_predictions = F.sigmoid(torch.squeeze(attack_net(out_top_k)))


        for j, t in enumerate(thresholds):
            true_positives[j] += (train_predictions >= t).sum().item()
            false_positives[j] += (out_predictions >= t).sum().item()
            false_negatives[j] += (train_predictions < t).sum().item()
            #print(train_top >= threshold)


            #print((train_top >= threshold).sum().item(),',',(out_top >= threshold).sum().item())

            correct[j] += (train_predictions >= t).sum().item()
            correct[j] += (out_predictions < t).sum().item()
            total[j] += train_predictions.size(0) + out_predictions.size(0)

    #print(true_positives,',',false_positives,',',false_negatives)

    for j, t in enumerate(thresholds):
        accuracy = 100 * correct[j] / total[j]
        precision = true_positives[j] / (true_positives[j] + false_positives[j]) if true_positives[j] + false_positives[j] != 0 else 0
        recall = true_positives[j] / (true_positives[j] + false_negatives[j]) if true_positives[j] + false_negatives[j] !=0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))
        

        
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
