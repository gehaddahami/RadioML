# Imports
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
from sklearn.metrics import accuracy_score


# TODO: after making sure the functions are properly working, remove all extra priniting statements 
def train_loop(model, train_loader, optimizer, criterion):
    losses = []
    model.train()

    for (inputs, labels, snr) in tqdm(train_loader):
        print('break')
        # Forward pass 
        # print('training input shape: ', inputs.shape)
        # print('training labels shape: ', labels.shape)
        output = model(inputs) 
        print('training output shape: ', output.shape)

        output_modified = output.argmax(dim = 1, keepdim = True)
        print('training output_mod shape: ', output_modified.shape)

        loss = criterion(output, labels)

        # Backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Use .item() to get the scalar value of the loss

    return losses
    

def test_loop(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        
        for (inputs, labels, snr) in tqdm(test_loader):

            print('validation input shape: ', inputs.shape)
            print('validation labels shape: ', labels.shape)
            print('labels unsqueezed shape:', labels.unsqueeze(1).shape)

            outputs = model(inputs)
            print('validation outputs shape: ', outputs.shape)
            pred = outputs.argmax(dim=1, keepdim=True)
            print('validation -pred shape: ', pred.shape)
            y_true.append(labels.numpy())  # Move labels back to CPU for concatenation
            y_pred.append(pred.reshape(-1).numpy())


            # trying another method to reshape the outputs 
            #pred_indices = outputs.argmax(dim=1, keepdim=True)
            #pred2 = torch.gather(outputs, 1, pred_indices)

            #print('before squeeze : ',pred2.shape) 
            #pred2 = torch.gather(outputs, 1, pred_indices).squeeze() 
            #pred2 = pred2.argmax(dim=1)
            #print('after squeeze : ', pred2.shape)
            # another method to calculate the accuracy 
            #y_true.append(labels.reshape(-1).numpy())  # Move labels back to CPU for concatenation
            #y_pred.append(pred2.reshape(-1).numpy())

    # printing the accuracy of the model (2nd method)
    y_true = np.concatenate(y_true)
    print(y_true.shape)
    y_pred = np.concatenate(y_pred)
    print(y_pred.shape)

    print(y_true)
    print(y_pred)

    # Calculate and print F1 score
    #f1 = f1_score(y_true, y_pred, average='weighted')
    #print(f'F1 Score: {f1}')

    return accuracy_score(y_true, y_pred), y_pred, y_true 


def display_loss(losses, title = 'Training loss', xlabel= 'Iterations', ylabel= 'Loss'):
    x_axis = [i for i in range(len(losses))] 
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()