import numpy as np
import torch
import torch.optim as optim
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import wandb

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

    flag_30 = False
    flag_50 = False
    flag_70 = False
    flag_90 = False

    since = time.time()
    dataset_size = len(dataloader)
    step_size = dataset_size // 10
    displacement = Displacement()
    lap_solver = hungarian

    print(f'Start training... {dataset_size}')


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

    max_acc = 0
    #prev_permat = None
    for epoch in tqdm(range(start_epoch, num_epochs)):
        evaluation_flag, training_flag = True, True
        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()

        train_accs = []
        valid_accs = []
        data_iterator = iter(dataloader)
        for idx in tqdm(range(dataset_size)):
            wandb.log({"idx" : idx})

            try:
                inputs = next(data_iterator)
            except:
                print(f"{idx} continue")
                continue


            inp_type = 'img'
            data1, data2 = [_.cuda() for _ in inputs['images']]
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].cuda()
            
            # Training
            if training_flag :
                print(f"training @ {epoch}")
                training_flag = False

            if idx in range(step_size*3 + 1) or idx in range(step_size*3, step_size*6 + 1) or idx in range(step_size*6, step_size*9 + 1):
                #print(f"training {idx}", end=" ")
                model.train()  # Set model to training mode
                acc = 0
                np.random.seed(None)

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

                    # training accuracy statistic
                    acc, _, __ = matching_accuracy(lap_solver(s_pred.double(), n1_gt, n2_gt), perm_mat, n1_gt)

                    # tfboard writer
                    loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                    loss_dict['loss'] = loss.item()

                    # statistics
                    running_loss += loss.item() * perm_mat.size(0)
                    epoch_loss += loss.item() * perm_mat.size(0)
                    
                    train_accs.append(acc)
            
            # evaluation
            if idx == step_size * 3 or idx == step_size * 6 or idx == step_size * 9: 

                training_acc = torch.mean(torch.FloatTensor(valid_accs))
                wandb.log({"Trainign Accuracy" : training_acc})
                train_accs = []

                
                # Evaluation
                if evaluation_flag :
                    print(f"evaluation @ {epoch}")
                    evaluation_flag = False
                    
                for i in tqdm(range(step_size*9, dataset_size)):
                    #print(f"evaluation {idx}", end=" ")
                    model.eval()  
                    acc = 0
                    np.random.seed(123)

                    #if idx == 24 and epoch == 0 :
                    #    print(perm_mat)
                    #    prev_permat = perm_mat
                    #elif idx == 24:
                    #    assert torch.all(prev_permat.eq(perm_mat))

                    # zero the parameter gradients
                    with torch.set_grad_enabled(True):
                        # forward
                        s_pred, d_pred = \
                            model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

                        # training accuracy statistic
                        acc, _, __ = matching_accuracy(lap_solver(s_pred.double(), n1_gt, n2_gt), perm_mat, n1_gt)
                        #print(acc)
                    valid_accs.append(acc)

                    epoch_loss = epoch_loss / dataset_size

                    valid_acc = torch.mean(torch.FloatTensor(valid_accs))
                    if acc > 0.30 and flag_30 == False : 
                        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
                        flag_30 = True
                        print(f"35 on epoch {epoch} with acc {valid_acc}")
                    if acc > 0.5 and flag_50 == False  : 
                        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
                        flag_50 = True
                        print(f"50 on epoch {epoch} with acc {valid_acc}")
                    if acc > 0.70 and flag_70 == False :
                        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
                        flag_70 = True
                        print(f"70 on epoch {epoch} with acc {valid_acc}")
                    if acc > 0.9 :
                        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
                        print(f"90 on epoch {epoch} with acc {valid_acc}")

                    scheduler.step()
                    if max_acc < valid_acc: max_acc = valid_acc

                    time_elapsed = time.time() - since
                    wandb.log({
                        "Epoch" : epoch,
                        "Epoch Loss": epoch_loss,
                        "Valid Accuracy": valid_acc,
                        "Max Accuracy": max_acc,
                        "Time": '{:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60),
                        "flag_30" : flag_30,
                        "flag_50":flag_50,
                        "flag_75":flag_70,
                        "flag_90":flag_90
                        })

    return model


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    wandb.init()

    args = parse_args('Deep learning of graph matching training & evaluation code.')
    wandb.config.update(args)

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    #torch.manual_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print('*** multi-gpu ***')
    model = model.cuda()
    wandb.watch(model)


    dataset = SGDataset('../scene-graph-TF-release/data_tools/')
    dataloader = get_dataloader(dataset)


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
