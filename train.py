import torch
import numpy as np
import os
import logging
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

from configs import get_cfg
from core.model.e2flow.e2flow import E2Flow
from core.loss import supervised_loss
from core.dataset.FlyingChairsDark import FlyingChairsData

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(cfgs):
    # Load Model
    model = E2Flow(cfgs.model)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    logging.info("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info(cfgs)

    # Load Data
    dataset_train = FlyingChairsData(cfgs.dataset.FlyingChairs, 'train', True)
    train_dataloader = DataLoader(dataset_train, batch_size=cfgs.training.batch_size, pin_memory=True, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=2)

    # Load optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.training.learning_rate, weight_decay=cfgs.training.adamw_decay, eps=cfgs.training.epsilon)
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=cfgs.training.learning_rate, 
                           epochs=cfgs.training.epochs, steps_per_epoch=train_dataloader.__len__(), 
                           pct_start=cfgs.training.pct_start)

    # Load autocast
    autocast = torch.cuda.amp.autocast
    scaler = GradScaler(enabled=cfgs.training.mixed_precision)

    # Training Loop
    steps = 0
    e_start = 0
    loss_records = 0.
    loss_step = 0
    if cfgs.training.checkpoint is not None:
        checkpoint = torch.load(cfgs.training.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        e_start = checkpoint['epoch'] + 1
        steps = 0
    for epoch in range(e_start, cfgs.training.epochs):
        # 1. One Epoch
        for data_blob in tqdm(train_dataloader):
            # Input
            image1, image2, voxel, flow, valid = [x.to(device, non_blocking=True) for x in data_blob]

            with autocast(enabled=cfgs.training.mixed_precision):
                flow_predictions = model(image1, image2, voxel)
                loss = supervised_loss(flow_predictions, flow, valid, cfgs.loss.gamma)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if not( loss > 0 and  loss < 100):
                logging.info('Step='+str(steps)+' Invalid Loss=' + str(loss))
                loss = 0
                loss_step -= 1            
            else:
                # grad clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfgs.training.grad_clip)

                # optimizie
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            steps += 1
            loss_step += 1
            if loss_step % 1000 == 0:
                print('Step='+str(steps)+' Loss='+str(float(loss_records/1000)) + ' LR='+str(scheduler.get_last_lr()[0]))
                logging.info('Step='+str(steps)+' Loss='+str(float(loss_records/1000)) + ' LR='+str(scheduler.get_last_lr()[0]))
                loss_records = 0.
                loss_step = 0
            else:
                loss_records = loss_records + loss

        # 2. Save
        if epoch % 5 == 4:
            if not os.path.exists(cfgs.training.save_path):
                os.makedirs(cfgs.training.save_path)
            torch.save({
                'epoch': epoch,
                'step': steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(cfgs.training.save_path, cfgs.training.prefix+'_'+str(epoch)+'.tar'))


def setSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    cfgs = get_cfg()
    setSeed(cfgs.seed)

    # Start logging
    logging.basicConfig(filename=os.path.join(cfgs.training.save_path, cfgs.training.prefix+'_log.txt'), level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    train(cfgs)
