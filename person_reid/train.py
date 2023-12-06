from dataset import *
from network import *
from utils import *
from pytorch_metric_learning import losses
import torch
import torch.nn as nn
import os
from tqdm import *
import datetime
from torch.utils.tensorboard import SummaryWriter
import typer



def run_code(
    experiment_name: str = typer.Option(
        'experiment',
        "--experiment-name",
        help="The name of the experiment (used for the TensorBoard)",
    ),
    dataset_path: str = typer.Option(
        '/home/anca/work/RAG/Computer Vision/data/reid',
        "--dataset-path",
        help="Path to the Market 101 dataset",
    ),
    model_dir: str = typer.Option(
        'model_best_ckpt_directory',
        "--model-dir",
        help="The output directory that saves the best ckpts",
    ),
    model_name: str = typer.Option(
        'resnet50',
        "--model-name",
        help="The name of the model",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        help="Batch size value",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        help="Number of workers",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        help="Number of epochs",
    ),
    embedding_shape: int = typer.Option(
        2048,
        "--embedding-shape",
        help="The size of the embedding shape",
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="The learning rate value",
    ),):


    experiment = experiment_name
    writer = SummaryWriter(experiment)


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    train_dataset = PersonReIdDataset(dataset_path, 'train', transform=resnet_transform_train)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            drop_last = True,
            pin_memory = True
        )

    val_dataset = PersonReIdDataset(dataset_path, 'val', transform=resnet_transform_eval)

    validation_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
    )

    model = Resnet(emb=embedding_shape, pretrained=True, is_norm=1, bn_freeze =1, type=model_name).cuda()

    num_classes =  train_dataset.get_unique_classes()
    criterion = losses.ProxyAnchorLoss(num_classes, embedding_shape, margin = 0.1, alpha = 32)




    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters())))},
        {'params': model.model.embedding.parameters(), 'lr':float(lr) * 1},
        {'params': criterion.parameters(), 'lr':float(lr) * 100}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=float(lr), weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)

    best_recall=0
    bn_freeze = True

    for epoch in range(0, epochs):
        model.train()

        if bn_freeze:
                modules = model.model.modules()
                for m in modules: 
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == 1:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True


        losses_epoch = []
        for batch_idx, (images, labels) in tqdm(enumerate(train_dataloader)): 

            optimizer.zero_grad()

            m = model(images.squeeze().cuda())
            loss = criterion(m, labels.squeeze().cuda())
            
            loss.backward()
        

            losses_epoch.append(loss.data.cpu().numpy())
            optimizer.step()

            if batch_idx % 50 == 0:
                print('\nEpoch: {} [{}/{}] Loss: {}'.format(
                    epoch, batch_idx + 1, len(train_dataloader),
                    loss.item()))

        scheduler.step()

        
        
        with torch.no_grad():
            print("Evaluation:")
            recall_train_scores = evaluate_cos(model, "Train", train_dataloader, writer, epoch)
            recall_val_scores = evaluate_cos(model, "Validation", validation_dataloader, writer, epoch)
        if best_recall < recall_val_scores[0]:
            best_recall = recall_val_scores[0]
            best_epoch = epoch
            
            torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_best.pth'.format(model_dir, model_name, epoch))


    writer.close()


if __name__ == "__main__":
    typer.run(run_code)
