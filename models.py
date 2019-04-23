import torch, os
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


class DeepNet(nn.Module):
    def __init__(self, base, n_outputs, full_backprop=False):
        super().__init__()
        # Use an existing architecture to extract image features
        if base == 'vgg':
            self.basemodel = models.vgg11(pretrained=True)
            self.num_ftrs = 512
        elif base == 'resnet':
            self.basemodel = models.resnet18(pretrained=True)
            self.num_ftrs = 512
        elif base == 'alexnet':
            self.basemodel = models.alexnet(pretrained=True)
            self.num_ftrs = 1024
        else:
            raise AttributeError('base arg should be one of vgg, resnet or alexnet')

        # Freeze the weights. We do not always want to backpropagate through ALL the network
        if not full_backprop:
            for p in self.parameters():
                p.requires_grad = False

        # Change the last layer of classification for our purposes. (our number of classes for example)
        if base == 'resnet':
            self.basemodel.fc = nn.Linear(self.num_ftrs, n_outputs)
        else:
            self.basemodel.classifier[6].out_features = n_outputs
            for p in self.basemodel.classifier[6].parameters():
                p.requires_grad = True
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Set the device to cuda if a GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.basemodel.to(self.device)

    def forward(self, input):
        """ Forward propagation through the network"""
        input = Variable(input)
        return self.basemodel(input)

    def forward_with_hooks(self, input, layer_index):
        """ Forward propagation with hooks (for feature vizualization """
        self.activations = SaveFeatures(list(self.basemodel.children())[layer_index])
        self.forward(input.unsqueeze(0))
        return self.activations.features

    def predict(self, input):
        """ Forward propagation with prediction """
        output = self.forward(input)
        output = torch.exp(self.logsoftmax(output))
        return torch.max(output, dim=1)

    def fit(self, training_data, validation_data, criterion, optimizer, num_epochs=16, verbose=True):
        """ Training function for the deepnet model """
        self.tr_accuracy, self.val_accuracy = [], []
        for epoch in range(num_epochs):
            self.train()
            tr_acc = []
            for inputs, labels in training_data:
                # 1 : Convert input tensor and target tensors to GPU if available
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # 2 : Reset the optimizer for the batch
                optimizer.zero_grad()
                # 3 : Go through the network. Prediction : Highest value in output (no probas in training)
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                # 4 : Compute the error, between our outputs and the ground truth (labels)
                loss = criterion(outputs, labels)
                # 5 : Back propagate the errors and adjust the weight according to the optimizer
                loss.backward()
                optimizer.step()

                tr_acc.append(1.0 * sum(torch.eq(preds, labels).tolist()) / len(preds))
            self.tr_accuracy.append(np.mean(tr_acc))
            self.eval()
            val_acc = []
            for inputs, labels in validation_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)

                val_acc.append(1.0 * sum(torch.eq(preds, labels).tolist()) / len(preds))
            self.val_accuracy.append(np.mean(val_acc))
            if(verbose):
                print('EPOCH #{} ... Validation accuracy : {}'.format(epoch, np.mean(val_acc)))


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True).to(self.device)
    def close(self):
        self.hook.remove()


