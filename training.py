import torch
import torchvision
import dataload
import utils
#from train_utils import train_eval_utils as utils
#from engine import train_one_epoch
#from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trdataset = dataload.TvidDataset(root=r'./', mode='train')
tedataset = dataload.TvidDataset(root=r'./', mode='test')
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    trdataset, batch_size=10, shuffle=True, num_workers=0)
#    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    tedataset, batch_size=10, shuffle=False, num_workers=0)
#    collate_fn=utils.collate_fn)


#weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
# For training
num_classes = 4

# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# training for 10 epochs
num_epochs = 10



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return 0


for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# def train(batch,model)
#     model.train()
#     images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
#     boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
#     labels = torch.randint(1, 91, (4, 11))
#     images = list(image for image in images)
#     targets = []
#     for i in range(len(images)):
#         d = {}
#         d['boxes'] = boxes[i]
#         d['labels'] = labels[i]
#         targets.append(d)
#     output = model(images, targets)
# def train_one_epoch(model,optimizer, dataloader,  device,epoch,print_freq ):
#     model.train()
#     correct, total = 0, 0
#     for x, y in dataloader:
#         temp_tr_loss_sum = 0
#         x = x.to(device)
#
#         y = y.to(device)
#         temp_y_pred = model(x)
#
#
#         temp_loss = criterion['cls'](temp_y_pred, y)
#
#
#         optimizer.zero_grad()
#         temp_loss.backward()
#         optimizer.step()
#         temp_tr_loss_sum += temp_loss.cpu().item()
#         arg = temp_y_pred.argmax(dim=1)
#         y = y.argmax(dim = 1)
#         correct += (arg == y).sum().cpu().item()
#
#
#         total += batch
#
#        scheduler.step()
#     return correct / total * 100.0
#
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400)]
# predictions = model(x)
# print(predictions)
# print(x)
# # optionally, if you want to export the model to ONNX:
# torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)


