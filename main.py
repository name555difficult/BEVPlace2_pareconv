import argparse
from math import ceil
import pdb
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss

from collate import precompute_neibors
import kitti_dataset
import nclt_dataset 
from model import PARE_Net

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description='BEVPlace++')
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    
    parser.add_argument('--batchSize', type=int, default=1, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--num_neg', type=int, default=2, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=4, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')

    parser.add_argument('--threads', type=int, default=0, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')


    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')


    parser.add_argument('--load_from', type=str, default='/home/ssd8t/wzb/BEVPlace2/kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='None', 
            help='Load_from from latest or best checkpoint.', choices=['latest', 'best'])
    

    opt = parser.parse_args()
    return opt

def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()

def train_epoch(epoch, model, train_set, opt):
    
    epoch_loss = 0

    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    criterion = TripletLoss().to(device)
    
    
    model.eval()
    

    # if epoch>=0:
    #     print('====> Building Cache for Hard Mining')
    #     train_set.mining=False
    #     train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
    #     with h5py.File(train_set.cache, mode='w') as h5: 
    #         pool_size = model.global_feat_dim

    #         h5feat = h5.create_dataset("features", 
    #                 [len(train_set), pool_size], 
    #                 dtype=np.float32)
    #         training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
    #             batch_size=opt.batchSize, shuffle=False, 
    #             collate_fn=kitti_dataset.pc_collate_fn)
    #         with torch.no_grad():
    #             for iteration, (data_dict, indices) in enumerate(training_data_loader, 1):
    #                 data_dict = to_cuda(data_dict)
    #                 query = query.to(device)
    #                 _, _, global_descs = model(query)
    #                 h5feat[indices, :] = global_descs.detach().cpu().numpy()
    #     train_set.mining=True
    #     train_set.refreshCache()
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=kitti_dataset.pc_collate_fn)
    
    model.train()

    for iteration, (data_dict, indices) in enumerate(training_data_loader):

        data_dict = to_cuda(data_dict)
        data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                              4,
                                              [35] * 4,
                                              )
        data_dict.update(data)
        # B, C, H, W = query.shape
        # input = torch.cat([query, positives, negatives])

        # input = input.to(device)
        
        global_descs = model(data_dict)

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [1, 1, opt.num_neg])
        

        optimizer.zero_grad()

        # no need to train the kps feature
        loss = 0
        num_negs = opt.num_neg
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            loss += max_loss
        
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * n_batches) + iteration)
            

    optimizer.zero_grad()    
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def infer(eval_set, return_local_feats = False):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, 
                batch_size=opt.cacheBatchSize, 
                collate_fn=kitti_dataset.infer_collate_fn,
                shuffle=False)

    model.eval()
    model.to('cuda')
    with torch.no_grad():
        
        all_global_descs = []
        all_local_feats = []
        for iteration, (data_dict, indices) in enumerate(tqdm(test_data_loader)):

            data_dict = to_cuda(data_dict)
            data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                                4,
                                                [35] * 4,
                                                )
            data_dict.update(data)
            # B, C, H, W = query.shape
            # input = torch.cat([query, positives, negatives])

            # input = input.to(device)
            
            global_descs = model(data_dict)
            local_feat = None
        # for _, (imgs, _) in enumerate(tqdm(test_data_loader)):
            # imgs = imgs.to(device)
            # _ , local_feat, global_desc = model(imgs)
            all_global_descs.append(global_descs.detach().cpu().numpy())
            if return_local_feats:
                all_local_feats.append(local_feat.detach().cpu().numpy())
           
    if return_local_feats:
        return np.concatenate(all_local_feats, axis=0), np.concatenate(all_global_descs, axis=0)
    else:
        return np.concatenate(all_global_descs, axis=0)

def testPCA(eval_set, epoch=0, write_tboard=False):
    # TODO global descriptor PCA for faster inference speed
    pass
    # return recalls


def getClusters(cluster_set):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, 128], 
                        dtype=np.float32)

            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), 128, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True)
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(128, 64, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')


if __name__ == "__main__":
    opt = get_args()

    device = torch.device("cuda")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Building model')

    from REIN import REIN

    model = PARE_Net(
        96, 64, 4, False, 'edge_conv', True, True
    )
    model = model.cuda()
    
    # initialize netvlad with pre-trained or cluster
    if opt.load_from:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pth.tar')
        else:
            resume_ckpt = opt.load_from

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            # model.load_state_dict(checkpoint['model'])
            keys_state_dict = set(ckpt['model'].keys())
            keys_model = set(model.state_dict().keys())
            unexpected_keys = keys_state_dict - keys_model
            missing_keys = keys_model - keys_state_dict
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
                exit(-1)
            else:
                print(f"unsurpported keys: {unexpected_keys}")
                model.load_state_dict(ckpt['model'], strict=False)

            model = model.to(device)
            # print("=> loaded checkpoint '{}' (epoch {})"
            #     .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    # else:
    #     initcache = join(opt.cachePath, 'desc_cen.hdf5')
    #     if not isfile(initcache):
    #         train_set = kitti_dataset.TrainingDataset()
    #         print('===> Calculating descriptors and clusters')
    #         getClusters(train_set)
    #     with h5py.File(initcache, mode='r') as h5: 
    #         clsts = h5.get("centroids")[...]
    #         traindescs = h5.get("descriptors")[...]
    #         model.pooling.init_params(clsts, traindescs) 
    #         model = model.cuda()

    if opt.mode.lower() == 'train':
        # preparing tensorboard
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))

        logdir = writer.file_writer.get_logdir()
        try:
            makedirs(logdir)
        except:
            pass

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)


        print('===> Loading dataset(s)')

        train_set = kitti_dataset.TrainingDataset(num_neg=opt.num_neg) 
        val_set={}
        for seq in ['00', '02', '05', '06']:   
        # for seq in ['2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28','2012-11-16','2013-02-23']:
            val_set[seq] = kitti_dataset.InferDataset(seq=seq)

        # initilize model weights
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
            model.parameters()), lr=opt.lr)   

        best_score = 0

        for epoch in range(opt.nEpochs):
            
            train_epoch(epoch, model, train_set, opt)
            
            print('===> Testing')
            recalls_kitti = []
            for seq in ['00', '02', '05', '06','08']:
                test_set = kitti_dataset.InferDataset(seq=seq)   
                global_descs = infer(test_set)
                recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
                recalls_kitti.append(recall_top1)

                writer.add_scalars('val', {'KITTI_'+seq: recall_top1}, epoch)

            eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
            eval_datasets = []
            eval_global_descs = []
            for seq in eval_seq:   

                test_set = nclt_dataset.InferDataset(seq=seq)   
                global_descs = infer(test_set)
                eval_global_descs.append(global_descs)
                eval_datasets.append(test_set)
            
            recalls_nclt = nclt_dataset.evaluateResults(eval_global_descs, eval_datasets)# (q_descs, db_descs, q_dataset, db_dataset)
            for ii in range(len(recalls_nclt)):
                writer.add_scalars('val', {'NCLT_'+eval_seq[ii+1]: recalls_nclt[ii]}, epoch)
            
            mean_recall = np.mean(recalls_nclt)

            print('===> Mean Recall on KITTI: %0.2f'%(np.mean(recalls_kitti)*100))
            print('===> Mean Recall on NCLT : %0.2f'%(np.mean(recalls_nclt)*100))

            is_best = mean_recall > best_score 
            if is_best:   best_score = mean_recall
            
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
            }, is_best, logdir)

        print('===> Best Recall: %0.2f'%(mean_recall*100))
        writer.close()

    elif opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        
        recalls_kitti = []
        print('====> Extracting Features of KITTI and calculating recalls')
        eval_seq =  ['08']

        for seq in eval_seq:   
            if seq=='08':
                test_set = kitti_dataset.InferDataset(seq=seq, sample_inteval=5)  #return a very large local feature mat could be very slow. sample the dataset to reduce ram and time cost
                global_descs = infer(test_set, return_local_feats=False)  
                recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set, None)
            else:
                test_set = kitti_dataset.InferDataset(seq=seq, sample_inteval=5)  
                global_descs = infer(test_set, return_local_feats=False)  
                recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set, None)
            recalls_kitti.append(recall_top1)

        mean_recall = np.mean(recalls_kitti)
        print('\n################# Recall @ top 1 on KITTI ########################\n')
        for ii in range(len(eval_seq)):
            print('%s: %0.2f'%(eval_seq[ii], recalls_kitti[ii]*100))

        print('mean: %0.2f'%(mean_recall*100))
        # print('################# Global Loc Results on KITTI 08  ##################\n')

        # print('Success rate: %0.2f; Mean Trans. Err.: %0.2f; Mean Rot. Err.: %0.2f'%(success_rate*100, mean_trans_err, mean_rot_err))

        
        print('\n')


        # print('====> Extracting Features of NCLT and calculating recalls')
        # eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
        # eval_datasets = []
        # eval_global_descs = []
        # for seq in eval_seq:   

        #     test_set = nclt_dataset.InferDataset(seq=seq)   
        #     global_descs = infer(test_set)
        #     eval_global_descs.append(global_descs)
        #     eval_datasets.append(test_set)
        
        # recalls_nclt = nclt_dataset.evaluateResults(eval_global_descs, eval_datasets)# (q_descs, db_descs, q_dataset, db_dataset)

        # print('\n################# Recall @ top 1 on NCLT ########################\n')
        # mean_recall = np.mean(recalls_nclt)


        # for ii in range(len(eval_seq[1:])):
        #     print('%s: %0.2f'%(eval_seq[ii+1], recalls_nclt[ii]*100))
        
        # print('mean: %0.2f'%(mean_recall*100))

