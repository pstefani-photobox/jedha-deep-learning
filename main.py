import torch
from models import DeepNet, Dataset

BATCH_SIZE = 32
NB_CLASSES = 16
NUM_EPOCHS = 32
MODELPATH = 'models/'

DatasetCreator = Dataset(BATCH_SIZE)
training_data, validation_data = DatasetCreator.create('images/')

deep_net = DeepNet('resnet', NB_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_net.parameters())
deep_net.fit(training_data, validation_data, criterion, optimizer, num_epochs=NUM_EPOCHS)
deep_net.save(MODELPATH)
