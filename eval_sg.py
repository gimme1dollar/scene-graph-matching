import torch
import time
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils.hungarian import hungarian
from data.sg_dataset import SGDataset, get_dataloader
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
import numpy as np
from utils.config import cfg
from GMN.displacement_layer import Displacement

def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(f'./output/vgg16_gmn_willow/params/params_0001.pt')
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()
    displacement = Displacement()
    lap_solver = hungarian

    # inference
    running_since = time.time()
    acc_match_num = torch.zeros(1, device=device)
    acc_total_num = torch.zeros(1, device=device)
    iter_num = 0
    for inputs in dataloader:
        if inputs == None:
            continue

        inp_type = 'img'
        data1, data2 = [_.cuda() for _ in inputs['images']]
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
        G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
        H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
        KG, KH = [_.cuda() for _ in inputs['Ks']]
        label = inputs['gt_perm_mat'].cuda()

        batch_num = data1.size(0)

        iter_num = iter_num + 1

        with torch.set_grad_enabled(False):
            s_pred, pred = \
                model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt,
                      n1_gt, n2_gt, KG, KH, inp_type)

        s_pred = s_pred.cpu().numpy()
        a_pred = np.zeros_like(s_pred)
        a_pred_= np.zeros((np.shape(a_pred)[0],np.shape(a_pred)[1]))
        for b in range(batch_num):
            for i in range(np.shape(a_pred)[1]):
                m = np.argmax(s_pred[b][i])
                a_pred_[b][i] = s_pred[b][i][m]
        for b in range(batch_num):
            k = min(n1_gt[b].item(), n2_gt[b].item())
            for i in a_pred_[b].argsort()[-k:][::-1]:
                m = np.argmax(s_pred[b][i])
                a_pred[b][i][m] = 1
        print(f"** Result **")
        #print(f"raw_pred :\n{s_pred}")
        print(f"pred :\n{a_pred}")
        print(f"label :\n{label.cpu().numpy()}")

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #model.train(mode=was_training)

if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)


    dataset = SGDataset('../scene-graph-proposal/data/vg/')
    dataloader = get_dataloader(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    #with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
    print_easydict(cfg)
    eval_model(model, dataloader, eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None, verbose=True)