import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from dataset import *
from network import *
from pytorch_metric_learning import losses
import torch


resnet_transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


resnet_transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                if i == 0:
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) 
    
    return [torch.stack(A[i]) for i in range(len(A))]



###
def evaluate_cos(model, dataset_type, dataloader, writer, epoch, k_values=[1, 3, 5, 10, 20]):
    nb_classes = dataloader.dataset.get_unique_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    T = T.cuda()

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    print('{} values:'.format(dataset_type))
    for k in k_values:
        r_at_k = calc_recall_at_k(T.float().cpu(), Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
        writer.add_scalar("{}/R@{}".format(dataset_type, k), r_at_k, epoch)

    return recall
