import numpy as np
import torch
import torch.optim as optim
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

#from data.data_loader import GMDataset, get_dataloader
from data.sg_dataset import SGDataset, get_dataloader
from GMN.displacement_layer import Displacement
from utils.offset_loss import RobustLoss
from utils.permutation_loss import CrossEntropyLoss
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model, save_model
from eval import eval_model
from utils.hungarian import hungarian

from utils.config import cfg

def train_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     num_epochs=1200,
                     resume=False,
                     start_epoch=0):
    print('Start training...')

    flag_30 = False
    flag_50 = False
    flag_70 = False

    since = time.time()
    dataset_size = len(dataloader)
    displacement = Displacement()
    lap_solver = hungarian

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in tqdm(range(start_epoch, num_epochs)):

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0


        # Training
        model.train()  # Set model to training mode
        acc = 0
        np.random.seed(None)
        for idx, inputs in enumerate(dataloader):
            if idx > 10000:
                #print(idx, inputs)
                if inputs == []:
                    continue
                #if idx % 100 == 0:
                #    print(f"\ton image {idx}")
                
                inp_type = 'img'
                data1, data2 = [_.cuda() for _ in inputs['images']]
                P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
                n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
                e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
                G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
                H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
                KG, KH = [_.cuda() for _ in inputs['Ks']]
                perm_mat = inputs['gt_perm_mat'].cuda()
                
                iter_num = iter_num + 1

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward
                    s_pred, d_pred = \
                        model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

                    multi_loss = []
                    if cfg.TRAIN.LOSS_FUNC == 'offset':
                        d_gt, grad_mask = displacement(perm_mat, P1_gt, P2_gt, n1_gt)
                        loss = criterion(d_pred.double(), d_gt.double(), grad_mask.double())
                    elif cfg.TRAIN.LOSS_FUNC == 'perm':
                        loss = criterion(s_pred.float(), perm_mat.float(), n1_gt, n2_gt)
                    else:
                        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    #if cfg.MODULE == 'NGM.hypermodel':
                    #    tfboard_writer.add_scalars(
                    #        'weight',
                    #        {'w2': model.module.weight2, 'w3': model.module.weight3},
                    #        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    #    )

                    # training accuracy statistic
                    acc, _, __ = matching_accuracy(lap_solver(s_pred.double(), n1_gt, n2_gt), perm_mat, n1_gt)

                    # tfboard writer
                    loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                    loss_dict['loss'] = loss.item()
                    #tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                    #accdict = dict()
                    #accdict['matching accuracy'] = acc
                    #tfboard_writer.add_scalars(
                    #    'training accuracy',
                    #    accdict,
                    #    epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    #)

                    # statistics
                    running_loss += loss.item() * perm_mat.size(0)
                    epoch_loss += loss.item() * perm_mat.size(0)

                    #if iter_num % cfg.STATISTIC_STEP == 0:
                    #    running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                    #    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                    #          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                    #    tfboard_writer.add_scalars(
                    #        'speed',
                    #        {'speed': running_speed},
                    #        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    #    )
                    #    running_loss = 0.0
                    #    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size


        # evaluation
        model.eval()  
        acc = 0
        np.random.seed(123)
        for idx, inputs in enumerate(dataloader):
            if idx < 10000:
                #print(idx, inputs)
                if inputs == []:
                    continue
                #if idx % 100 == 0:
                #    print(f"\ton image {idx}")
                
                inp_type = 'img'
                data1, data2 = [_.cuda() for _ in inputs['images']]
                P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
                n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
                e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
                G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
                H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
                KG, KH = [_.cuda() for _ in inputs['Ks']]
                perm_mat = inputs['gt_perm_mat'].cuda()
                
                iter_num = iter_num + 1

                # zero the parameter gradients
                with torch.set_grad_enabled(True):
                    # forward
                    s_pred, d_pred = \
                        model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

                    # training accuracy statistic
                    acc, _, __ = matching_accuracy(lap_solver(s_pred.double(), n1_gt, n2_gt), perm_mat, n1_gt)


        if acc > 0.35 and flag_30 == False : 
            #save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            #torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
            flag_30 = True
            print(f"35 on epoch {epoch} with acc {acc}")
        if acc > 0.5 and flag_50 == False  : 
            #save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            #torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
            flag_50 = True
            print(f"50 on epoch {epoch} with acc {acc}")
        if acc > 0.75 and flag_70 == False :
            #save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            #torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
            flag_70 = True
            print(f"75 on epoch {epoch} with acc {acc}")
        if acc > 0.9 :
            #save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            #torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
            print(f"90 on epoch {epoch} with acc {acc}")

        scheduler.step()

        if epoch % 1 == 0:
            print('Learning_rate = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]), end=' ')
            print('Epoch: {:d} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, acc), end=' ')
            time_elapsed = time.time() - since
            print('Time_passed {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset = SGDataset('../scene-graph-IMP/data/vg/')
    dataloader = get_dataloader(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.cuda()

    if cfg.TRAIN.LOSS_FUNC == 'offset':
        criterion = RobustLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    print_easydict(cfg)
    model = train_model(model, criterion, optimizer, dataloader,
                        num_epochs=cfg.TRAIN.NUM_EPOCHS,
                        resume=cfg.TRAIN.START_EPOCH != 0,
                        start_epoch=cfg.TRAIN.START_EPOCH)
