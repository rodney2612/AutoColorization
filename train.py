import os
import traceback
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors

from myimgfolder import TrainImageFolder
from colornet import ColorNet

original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

have_cuda = torch.cuda.is_available()
epochs = 20

data_dir = "cifar/train/"
train_set = TrainImageFolder(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
batch_size = 2
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
color_model = ColorNet()

# Weights of our model are saved in colornet_params.pkl
if os.path.exists('./colornet_params.pkl'):
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())

'''
    Stores representative points (central points) of each of the 313 bins.
    Then we can map any (a,b) value to a bin by computing its nearest neighbor from these 313
    central points.
'''
bins = np.load("pts_in_hull.npy")

''' 
    Stores the empirical probabilties of each bin.
    Empirical probabilities are simply calculated by mapping each pixel to 1 of the 313 bins.
    Then the count of the bin in the whole image dataset divided by the total no of pixels gives
    the empirical probability of the bin.
'''
prior_probs = np.load("prior_probs.npy")

'''
    algorithm = 'auto' means the nearest neighbors library will use either of 'ball_tree', 'kd_tree' or
    'brute_force' algorithm. The chosen algorithm will be optimised for our data points.
'''
neighbor = NearestNeighbors(n_neighbors=1,algorithm='auto', leaf_size=30)

'''
    The fit function will preprocess the central points(cluster centres of the bins) in a data structure
    so that we can calculate the bin corresponding to (a,b) value of any pixel quickly during training. 
'''
neighbor.fit(bins)

# Dimensions of our input image
h = w = 224
    
def cal_weights(img_ab):
    print("In calculate weights")
    #img_ab = torch.from_numpy(img_ab.transpose((1, 2, 0)))
    img_ab = img_ab.transpose(1,2)
    img_ab = img_ab.transpose(2,3)
    #print("img_ab size after 1st transpose",img_ab.size())
    weights = np.zeros([batch_size,h,w])

    for img_no in range(len(img_ab)): #batch size
        for i in range(len(img_ab[0])): # row
            for j in range(len(img_ab[0][0])): # col values
                #print("a value",img_ab[img_no][0][i][j].item())
                '''
                    The multiplication by 255 and subtraction by 128 is done because
                    the code of the original model mapped both the a and b value to between 0 and 1.
                    But the bins have been created considering a and b values between 0 to 128.
                '''
                a = (img_ab[img_no][i][j][0] * 255 - 128).item()
                b = (img_ab[img_no][i][j][1] * 255 -128).item()
                '''
                    Here, we map (a,b) value of each pixel to its corresponding bin
                '''
                index = neighbor.kneighbors([[a,b]])[1][0][0]
                #print("bin: ",index)
                '''
                    Storing weights for each pixel (which we will multiply with the mean squared error loss.)
                '''
                weights[img_no][i][j] = prior_probs[index]

    
    '''
        Dimensions of img_ab at this point is batch_size * height * width * 2
        Dimenisons of weights is batch_size * height * width
        Using weights.expand_as(img_ab) requires dimensions of image to be 2 * batch_size * height * width
    '''
    img_ab = img_ab.transpose(0,3)
    img_ab = img_ab.transpose(1,3)
    img_ab = img_ab.transpose(2,3)
    #print("img_ab size after transpose",img_ab.size())
    #weights = weights.transpose((0,3,1,2))
    weights = torch.from_numpy(weights)
    #print("weights size after from_numpy",weights.size())
    weights = weights.expand_as(img_ab)
    #print("weights size after expand_as",weights.size())
    weights = weights.transpose(0,1)
    #print("Weights size after transpose",weights.size())

    img_ab = img_ab.transpose(0,1)
    #print("img_ab size after transpose",img_ab.size())
    return Variable(weights)

def train(epoch):
    color_model.train()

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            print("Batch id",batch_idx)
            messagefile = open('./message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            class_output, output = color_model(original_img, original_img)
            weights = cal_weights(img_ab)
            squared = torch.pow((img_ab - output), 2)
            ms = (weights.float() * squared.float()).sum()
            '''
                This is the loss from our colorization network.
            '''
            ems_loss = ms / torch.from_numpy(np.array(list(output.size()))).float().prod()
            print("ems loss",ems_loss.item())
            '''
                This is the loss from our classification network.
            '''
            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
            loss = ems_loss + cross_entropy_loss
            lossmsg = 'loss: %.9f\n' % (loss.item())
            messagefile.write(lossmsg)
            ems_loss.backward(retain_graph=True)
            cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print("size of weight",weights.size())
                print("size of img_ab",img_ab.size())
                print("size of output",output.size())
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params.pkl')
            messagefile.close()
                # print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')


for epoch in range(1, epochs + 1):
    print("Epoch no",epoch)
    train(epoch)
