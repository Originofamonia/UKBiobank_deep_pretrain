"""
use SFCN infer AD MRI images, then train features on LR or SVM
"""
import os
import torch
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from mri_data import Task, LoaderHelper


os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def evaluate(args, model, test_dl):
    model.eval()
    feats = []
    labels = []
    valid_loss = 0.
    with torch.no_grad():
        for ii, batch in tqdm(enumerate(test_dl)):
            # load the data to memory
            batch = tuple(item.to(args.device) for item in batch)
            batch_x, batch_y = batch
            # compute the outputs
            feat = model.get_feat(batch_x)
            # print(feat)
            feats.append(feat.squeeze().detach().cpu().numpy())
            labels.append(batch_y.squeeze().detach().cpu().numpy())

    feats = np.array(feats)
    labels = np.array(labels)
    clf = LogisticRegression(random_state=0).fit(feats, labels)
    preds = clf.predict(feats)
    print(f"preds: {preds}")
    print(f"labels: {labels}")
    acc = clf.score(feats, labels)
    print(f"LR acc: {acc}")


def main():
    model = SFCN()
    model = torch.nn.DataParallel(model)
    fp_ = './brain_age/run_20190719_00_epoch_best_mae.p'
    model.load_state_dict(torch.load(fp_))
    model.cuda()

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=444)
    parser.add_argument('--n_epochs', type=int, default=60)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(args.seed)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {args.device}')
    print('set up data loaders')
    ld_helper = LoaderHelper(task=Task.CN_v_AD)
    train_dl = ld_helper.get_train_dl(batch_size=args.batch_size)
    test_dl = ld_helper.get_test_dl(batch_size=1)

    evaluate(args, model, test_dl)
