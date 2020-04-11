device = 'cuda'
model = models.densenet121(pretrained=True)

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
                
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
       
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:   
            x = F.relu(linear(x),inplace = True)
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
('fc1', nn.Linear(1024, 200)),
('relu', nn.ReLU()),
('fc2', nn.Linear(200, 102)),
('drop', nn.Dropout(p=0.5)),
('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model.classifier)

def validation(model, valid_dataloaders, criterion):
    test_loss = 0
    accuracy = 0
    model = model.to('cuda')
    for images, labels in valid_dataloaders:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy
    
    
epochs =3
steps = 0
running_loss = 0
print_every = 10
print("hello")

for e in range(epochs):
    model.train()
    model = model.to('cuda')
    for images, labels in train_dataloaders:
        
        steps += 1
        images, labels = images.to(device), labels.to(device)
#         print(images.size())
        # Flatten images into a 784 long vector
#         print(images.size())
#         print(images.size())
        
#         print(images.size())
        optimizer.zero_grad()
        
        output = model.forward(images)        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, valid_dataloaders, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()    
