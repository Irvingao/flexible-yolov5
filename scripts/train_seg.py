import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('.')
from od.models.modules.experimental import attempt_load
from od.models.seg_model import Model
from od import ComputeLoss
from utils.autoanchor import check_anchors
from od import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first
from utils.saver import Saver
from scripts.eval import test

from od.data import make_data_loader


logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # segmentation
    seg = opt.seg
    print("enable segmentation:", seg)

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    model = Model(opt.cfg).to(device)  # create

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Pretrained
    print("os.listdir(weights):", os.listdir(weights))
    pretrained = os.path.join(weights, "model_best.pth.tar") if "model_best.pth.tar" in os.listdir(weights) else None
    print("weights:", weights)
    print("pretrained:", pretrained)
    if pretrained != None:

        model_CKPT = torch.load(pretrained)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        logger.info('-------------------------------------------------------------------')  # report
        logger.info('load pretrained model sucessfully')  # report
        logger.info('-------------------------------------------------------------------')  # report

    else:
        model = Model(opt.cfg).to(device)  # create


    best_pred = 0.0
    if opt.resume:
        if not os.path.isfile(opt.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(opt.resume))
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch']
        if cuda:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        if not opt.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(opt.resume, checkpoint['epoch']))
    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False


    imgsz, imgsz_test = [(608, 608), (608, 608)]

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    
    # Trainloader
    kwargs = {'num_workers': opt.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(opt, **kwargs)

    from od.models.loss import SegmentationLosses
    weight = None
    criterion = SegmentationLosses(weight=weight, cuda=cuda).build_loss()

    # Define Evaluator
    from utils.metrics import Evaluator
    evaluator = Evaluator(nclass)
    
    # Define lr scheduler
    from utils.lr_scheduler import LR_Scheduler
    scheduler = LR_Scheduler("poly", hyp['lr0'], opt.epochs, len(train_loader))
    writer = SummaryWriter(opt.save_dir)  # Tensorboard

    # Define Saver
    saver = Saver(opt)
    saver.save_experiment_config()



    print('Starting Epoch:', 0)
    print('Total Epoches:', opt.epochs)
    for epoch in range(0, opt.epochs):
        train_loss = 0.0
        model.train()
        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if cuda:
                image, target = image.cuda(), target.cuda()
            scheduler(optimizer, i, epoch, best_pred)
            optimizer.zero_grad()
            output = model(image)
            # target = target.unsqueeze(1) # 
            # print("target:", target.shape)
            # print("output:", output.shape)
            loss = criterion(output, target)
            # input()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                # self.summary.visualize_image(writer, self.opt.dataset, image, target, output, global_step)

        writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * opt.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if opt.no_val:
            # save checkpoint every epoch
            is_best = False
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best)
        # eval
        if not opt.no_val and epoch % opt.eval_interval == (opt.eval_interval - 1):
            tbar = tqdm(val_loader, desc='\r')
            test_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                if cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(image)
                loss = criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                evaluator.add_batch(target, pred)

            # Fast test during the training
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            writer.add_scalar('val/mIoU', mIoU, epoch)
            writer.add_scalar('val/Acc', Acc, epoch)
            writer.add_scalar('val/Acc_class', Acc_class, epoch)
            writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * opt.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('Loss: %.3f' % test_loss)

            new_pred = mIoU
            if new_pred >= best_pred:
                is_best = True
                best_pred = new_pred
                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_pred,
                }, is_best)

    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload

        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):  # speed, mAP tests
                results, _, _ = test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=conf,
                                          iou_thres=iou,
                                          model=attempt_load(final, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=save_json,
                                          plots=False)

    else:
        dist.destroy_process_group()

    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', action='store_true', help='enable segmentation')
    parser.add_argument('--dataset', type=str, default='coco',choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: coco)')
    parser.add_argument('--dataset_dir', type=str, default='/home/pc/workspace/dataset/coco2017', help='dataset dir')
    parser.add_argument('--base-size', type=int, default=608, help='segmentation base image size')
    parser.add_argument('--crop-size', type=int, default=608, help='segmentation crop image size')
    parser.add_argument('--checkname', type=str, default='checkpoint', help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=True, help='skip validation during training')

    parser.add_argument('--weights', type=str, default='run/coco/checkpoint', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='configs/model_yolo_segmentation.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='configs/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='configs/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[608, 608], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=6, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)

    wandb = None
    prefix = colorstr('prefix: ')
    logger.info(f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer)
