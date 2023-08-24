import math
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Movie_genre_classification:
    def __init__(self,train_file,validation_file,test_file):
        # Hyperparams
        n_inputs=300
        num_classes=6
        num_epochs=35
        # batch_size=50
        batch_size=33
        learning_rate=0.001

        class MovieDataset(Dataset):
            def __init__(self, file):
                file_=os.path.basename(file)
                file_name=os.path.splitext(file_)[0]
                file_df=pd.read_csv(file)
                file_genre_count=file_df['genre'].value_counts()
                file_le=preprocessing.LabelEncoder()
                file_le.fit(file_df['genre'])
                encoded_labels=file_le.transform(file_df['genre'])
                one_hot_labels=LabelBinarizer().fit_transform(encoded_labels)
                self.file_label_tens=torch.tensor(one_hot_labels,dtype=torch.float32)
                file_feat=file_df.iloc[:,:-1]
                self.file_feat_tensor=torch.tensor(file_feat.values, dtype=torch.float32)
                self.n_samples=file_feat.shape[0]
                print(f'There are total {len(file_genre_count)} classes in {file_name} set')
                for i,val in file_genre_count.items():
                    print(f'Genre {i} has {val} instances in {file_name} set')
                print('\n')

            def __getitem__(self, index):
                return self.file_feat_tensor[index], self.file_label_tens[index]

            def __len__(self):
                return self.n_samples

        #Class encoder dictionary
        class_dic={}
        class_dic[0]='action'
        class_dic[1]='comedy'
        class_dic[2]='documentary'
        class_dic[3]='drama'
        class_dic[4]='horror'
        class_dic[5]='thriller'

        train_dataset=MovieDataset(train_file)
        validation_dataset=MovieDataset(validation_file)
        test_dataset=MovieDataset(test_file)

        train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        validation_loader=DataLoader(dataset=validation_dataset,batch_size=batch_size,shuffle=False)
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

        class NeuralNet(nn.Module):
            def __init__(self,n_inputs,num_neurons_per_layer,num_classes,num_hidden_layers):
                super(NeuralNet, self).__init__()
                self.hidden_layers=nn.ModuleList()
                for i in range(num_hidden_layers):
                    if i==0:
                        input_size=n_inputs
                    else:
                        input_size=num_neurons_per_layer
                    self.hidden_layers.append(nn.Linear(input_size,num_neurons_per_layer))
                self.output_layer=nn.Linear(num_neurons_per_layer,num_classes)

            def forward(self,x):
                for hid_layer in self.hidden_layers:
                    x=torch.relu(hid_layer(x))
                x=self.output_layer(x)
                return x

        for iter in range(4):
            print('----------------------------------------')
            num_neurons_per_layer = [64,64,128,128]
            num_hidden_layers = [1,2,1,2]
            print(f'For {num_hidden_layers[iter]} hidden layer/s and {num_neurons_per_layer[iter]} neurons per layer')
            model=NeuralNet(n_inputs,num_neurons_per_layer[iter],num_classes,num_hidden_layers[iter]).to(device)

            #loss and criterion
            criterion=nn.CrossEntropyLoss()
            optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

            # training the model and validating to get best model
            best_val_loss=np.inf
            best_model=None
            num_class=len(class_dic)
            train_loss_per_epoch=[]
            valid_loss_per_epoch = []
            for epoch in range(num_epochs):
                for i, (feat,lab) in enumerate(train_loader):
                    batch_correct=0
                    feat=feat.to(device)
                    lab=lab.to(device)
                    #forward pass
                    op=model(feat)
                    loss=criterion(op,lab)
                    #Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # if (i+1)%100==0:
                    #     print(f'epoch {epoch+1}/{num_epochs}, Step [{i+1}/{n_total_steps}], loss: {loss.item(): .4f}')

                # Evaluating on training set
                train_loss=0.0
                all_batch_mean_acc_tr=[]
                all_batch_mean_per_class_acc_tr=[0] * num_class
                count=0
                for f,l in train_loader:
                    mean_per_class_acc_per_batch_tr=[0] * num_class
                    class_correct_per_batch_tr = [0] * num_class
                    class_total_per_batch_tr = [0] * num_class
                    f=f.to(device)
                    l=l.to(device)
                    inv_l=torch.where(l==1)[1]
                    train_output=model(f)
                    train_loss+=criterion(train_output,l).item()
                    # train_loss += criterion(train_output, inv_l).item()

                    _,train_pred=torch.max(train_output.data,1)
                    correct_batch_samples_tr=(train_pred == inv_l).sum().item()
                    for c in range(num_class):
                        class_correct_per_batch_tr[c]=((train_pred==inv_l)&(inv_l==c)).sum().item()
                        class_total_per_batch_tr[c]=(inv_l==c).sum().item()
                        if class_total_per_batch_tr[c]!=0:
                            mean_per_class_acc_per_batch_tr[c]= class_correct_per_batch_tr[c] / class_total_per_batch_tr[c]

                    for m in range(num_class):
                        all_batch_mean_per_class_acc_tr[m]+=mean_per_class_acc_per_batch_tr[m]

                    mean_acc_per_batch_tr=correct_batch_samples_tr/l.size(0)
                    all_batch_mean_acc_tr.append(mean_acc_per_batch_tr)

                all_batch_mean_per_class_acc_tr=[ele/len(train_loader) for ele in all_batch_mean_per_class_acc_tr]
                train_loss=train_loss/len(train_loader)
                all_batch_mean_acc_tr=sum(all_batch_mean_acc_tr)/len(train_loader)

                # if epoch == math.floor(num_epochs/2):
                #     print(f'Epoch {epoch}, Mean classification accuracy, training set: {(all_batch_mean_acc_tr)*100 :.4f}%')
                #     print(f'Epoch {epoch},Mean per class accuracy, training set: {class_dic[0]}: {all_batch_mean_per_class_acc_tr[0]*100 : .4f}%, {class_dic[1]}: {all_batch_mean_per_class_acc_tr[1]*100 : .4f}%, {class_dic[2]}: {all_batch_mean_per_class_acc_tr[2]*100 : .4f}%,'
                #           f' {class_dic[3]}: {all_batch_mean_per_class_acc_tr[3]*100 : .4f}%, {class_dic[4]}: {all_batch_mean_per_class_acc_tr[4]*100 : .4f}%, {class_dic[5]}: {all_batch_mean_per_class_acc_tr[5]*100 : .4f}%')
                # if epoch==num_epochs-1:
                #     print(f'Epoch {epoch}, Mean classification accuracy, training set: {(all_batch_mean_acc_tr) * 100 :.4f}%')
                #     print(f'Epoch {epoch},Mean per class accuracy, training set: {class_dic[0]}: {all_batch_mean_per_class_acc_tr[0] * 100 : .4f}%, {class_dic[1]}: {all_batch_mean_per_class_acc_tr[1] * 100 : .4f}%, {class_dic[2]}: {all_batch_mean_per_class_acc_tr[2] * 100 : .4f}%,'
                #         f' {class_dic[3]}: {all_batch_mean_per_class_acc_tr[3] * 100 : .4f}%, {class_dic[4]}: {all_batch_mean_per_class_acc_tr[4] * 100 : .4f}%, {class_dic[5]}: {all_batch_mean_per_class_acc_tr[5] * 100 : .4f}%')
                if epoch%10==0:
                    print(f'Epoch {epoch}, Mean classification accuracy, training set: {(all_batch_mean_acc_tr)*100 :.4f}%')
                    print(f'Epoch {epoch}, Mean per class accuracy, training set: {class_dic[0]}: {all_batch_mean_per_class_acc_tr[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_tr[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_tr[2] * 100 : .4f}% |'
                        f' {class_dic[3]}: {all_batch_mean_per_class_acc_tr[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_tr[4] * 100 : .4f}% | {class_dic[5]}: {all_batch_mean_per_class_acc_tr[5] * 100 : .4f}%')


                valid_loss = 0.0
                with torch.no_grad():
                    all_batch_mean_acc_valid = []
                    all_batch_mean_per_class_acc_valid = [0]*num_class
                    for features,labels in validation_loader:
                        mean_per_class_acc_per_batch_valid = [0]*num_class
                        class_correct_per_batch_valid = [0] * num_class
                        class_total_per_batch_valid = [0] * num_class
                        features = features.to(device)
                        labels = labels.to(device)
                        valid_output = model(features)
                        inv_labels=torch.where(labels==1)[1]
                        valid_loss += criterion(valid_output, labels).item()
                        _, valid_pred = torch.max(valid_output.data, 1)
                        correct_batch_samples_valid = (valid_pred == inv_labels).sum().item()

                        for c in range(num_class):
                            class_correct_per_batch_valid[c]=((valid_pred==inv_labels)&(inv_labels==c)).sum().item()
                            class_total_per_batch_valid[c]=(inv_labels==c).sum().item()
                            if class_total_per_batch_valid[c] != 0:
                                mean_per_class_acc_per_batch_valid[c] = class_correct_per_batch_valid[c]/class_total_per_batch_valid[c]

                        for m in range(num_class):
                            all_batch_mean_per_class_acc_valid[m] += mean_per_class_acc_per_batch_valid[m]

                        mean_acc_per_batch_valid = correct_batch_samples_valid / labels.size(0)
                        all_batch_mean_acc_valid.append(mean_acc_per_batch_valid)

                    all_batch_mean_per_class_acc_valid=[ele/len(validation_loader) for ele in all_batch_mean_per_class_acc_valid]
                    valid_loss = valid_loss / len(validation_loader)
                    all_batch_mean_acc_valid = sum(all_batch_mean_acc_valid) / len(validation_loader)

                    # if epoch == math.floor(num_epochs / 2):
                    #     print(f'Epoch {epoch}, Mean classification accuracy, validation set: {(all_batch_mean_acc_valid)*100 :.4f}%')
                    #     print(f'Epoch {epoch}, Mean per class accuracy, validation set: {class_dic[0]}: {all_batch_mean_per_class_acc_valid[0] * 100 : .4f}%, {class_dic[1]}: {all_batch_mean_per_class_acc_valid[1] * 100 : .4f}%, {class_dic[2]}: {all_batch_mean_per_class_acc_valid[2] * 100 : .4f}%,'
                    #         f' {class_dic[3]}: {all_batch_mean_per_class_acc_valid[3] * 100 : .4f}%, {class_dic[4]}: {all_batch_mean_per_class_acc_valid[4] * 100 : .4f}%, {class_dic[5]}: {all_batch_mean_per_class_acc_valid[5] * 100 : .4f}%')
                    # if epoch== num_epochs-1:
                    #     print(f'Epoch {epoch}, Mean classification accuracy, validation set: {(all_batch_mean_acc_valid) * 100 :.4f}%')
                    #     print(f'Epoch {epoch}, Mean per class accuracy, validation set: {class_dic[0]}: {all_batch_mean_per_class_acc_valid[0] * 100 : .4f}%, {class_dic[1]}: {all_batch_mean_per_class_acc_valid[1] * 100 : .4f}%, {class_dic[2]}: {all_batch_mean_per_class_acc_valid[2] * 100 : .4f}%,'
                    #         f' {class_dic[3]}: {all_batch_mean_per_class_acc_valid[3] * 100 : .4f}%, {class_dic[4]}: {all_batch_mean_per_class_acc_valid[4] * 100 : .4f}%, {class_dic[5]}: {all_batch_mean_per_class_acc_valid[5] * 100 : .4f}%')
                    if epoch%10==0:
                        print(f'Epoch {epoch}, Mean classification accuracy, validation set: {(all_batch_mean_acc_valid)*100 :.4f}%')
                        print(f'Epoch {epoch}, Mean per class accuracy, validation set: {class_dic[0]}: {all_batch_mean_per_class_acc_valid[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_valid[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_valid[2] * 100 : .4f}% |'
                            f' {class_dic[3]}: {all_batch_mean_per_class_acc_valid[3] * 100 : .4f}% |, {class_dic[4]}: {all_batch_mean_per_class_acc_valid[4] * 100 : .4f}% | {class_dic[5]}: {all_batch_mean_per_class_acc_valid[5] * 100 : .4f}%')

                    if valid_loss<best_val_loss:
                        best_val_loss=valid_loss
                        best_model=model.state_dict()

                train_loss_per_epoch.append(train_loss)
                valid_loss_per_epoch.append(valid_loss)

            # Testing best model
            model.load_state_dict(best_model)
            # #Evaluating best model on test set
            with torch.no_grad():
                all_batch_mean_acc_test = []
                all_batch_mean_per_class_acc_test = [0]*num_class
                for features, label in test_loader:
                    mean_per_class_acc_per_batch_test = [0]*num_class
                    class_correct_per_batch_test = [0] * num_class
                    class_total_per_batch_test = [0] * num_class
                    features = features.to(device)
                    label = label.to(device)
                    inv_labels_test=torch.where(label==1)[1]
                    # print(inv_labels_test)
                    test_output = model(features)
                    _, test_pred = torch.max(test_output.data, 1)
                    correct_batch_samples_test = (test_pred == inv_labels_test).sum().item()

                    for c in range(num_class):
                        class_correct_per_batch_test[c] = ((test_pred == inv_labels_test) & (inv_labels_test == c)).sum().item()
                        class_total_per_batch_test[c] = (inv_labels_test == c).sum().item()
                        if class_total_per_batch_test[c]!=0:
                            mean_per_class_acc_per_batch_test[c]= class_correct_per_batch_test[c] / class_total_per_batch_test[c]

                    for m in range(num_class):
                        all_batch_mean_per_class_acc_test[m]+=mean_per_class_acc_per_batch_test[m]

                    mean_acc_per_batch_test = correct_batch_samples_test / label.size(0)
                    all_batch_mean_acc_test.append(mean_acc_per_batch_test)

                all_batch_mean_per_class_acc_test = [ele / len(test_loader) for ele in all_batch_mean_per_class_acc_test]
                all_batch_mean_acc_test = sum(all_batch_mean_acc_test) / len(test_loader)

            # print(f'Mean classification accuracy, test data: {all_batch_mean_acc_test*100 :.4f}%')
            # print(f'Mean per class accuracy, test set: {class_dic[0]}: {all_batch_mean_per_class_acc_test[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_test[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_test[2] * 100 : .4f}% |'
            #     f' {class_dic[3]}: {all_batch_mean_per_class_acc_test[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_test[4] * 100 : .4f}% | {class_dic[5]}: {all_batch_mean_per_class_acc_test[5] * 100 : .4f}%')

            # plotting training and validation losses
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(range(num_epochs), train_loss_per_epoch, label='Training Loss')
            ax.plot(range(num_epochs), valid_loss_per_epoch, label='Validation Loss')
            ax.legend()
            ax.grid()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            plt.title(f'Training and Validation losses for {num_hidden_layers[iter]} hidden layer/s and {num_neurons_per_layer[iter]} nerons per hidden layer')
            plt.show()

if __name__=='__main__':
    train_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\train.csv'
    validation_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\validation.csv'
    test_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\test.csv'
    print('Movie Genre Classification')
    movie_classification=Movie_genre_classification(train_file,validation_file,test_file)
