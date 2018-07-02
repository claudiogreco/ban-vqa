"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import _pickle as cPickle
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import base_model
import utils
from classifier import SimpleClassifier
from dataset import Dictionary, VisualGenomeFeatureDataset, FoilFeatureDataset
from dataset import tfidf_from_questions
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', type=bool, default=False, help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', type=bool, default=False, help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', type=bool, default=True, help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='saved_models/ban')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    adaptive = True

    img_id2train = cPickle.load(
        open(os.path.join('data', '%s%s_imgid2idx.pkl' % ("train", '' if adaptive else '36')), 'rb'))

    img_id2val = cPickle.load(
        open(os.path.join('data', '%s%s_imgid2idx.pkl' % ("val", '' if adaptive else '36')), 'rb'))

    img_id2test = cPickle.load(
        open(os.path.join('data', '%s%s_imgid2idx.pkl' % ("test2015", '' if adaptive else '36')), 'rb'))

    train_img_pos_boxes = None
    val_img_pos_boxes = None
    test2015_img_pos_boxes = None

    print('loading image features from h5 train file')
    with h5py.File(os.path.join('data', '%s%s.hdf5' % ("train", '' if adaptive else '36')), 'r') as hf:
        train_img_features = np.array(hf.get('image_features'))
        train_img_spatials = np.array(hf.get('spatial_features'))
        if adaptive:
            train_img_pos_boxes = np.array(hf.get('pos_boxes'))

    print('loading image features from h5 val file')
    with h5py.File(os.path.join('data', '%s%s.hdf5' % ("val", '' if adaptive else '36')), 'r') as hf:
        val_img_features = np.array(hf.get('image_features'))
        val_img_spatials = np.array(hf.get('spatial_features'))
        if adaptive:
            val_img_pos_boxes = np.array(hf.get('pos_boxes'))

    print('loading image features from h5 test2015 file')
    with h5py.File(os.path.join('data', '%s%s.hdf5' % ("test2015", '' if adaptive else '36')), 'r') as hf:
        test2015_img_features = np.array(hf.get('image_features'))
        test2015_img_spatials = np.array(hf.get('spatial_features'))
        if adaptive:
            test2015_img_pos_boxes = np.array(hf.get('pos_boxes'))

    train_dset = FoilFeatureDataset(
        'data/foilGWVQA.train.json',
        dictionary, img_id2train,
        img_id2val,
        img_id2test,
        train_img_features,
        train_img_spatials,
        train_img_pos_boxes,
        val_img_features,
        val_img_spatials,
        val_img_pos_boxes,
        test2015_img_features,
        test2015_img_spatials,
        test2015_img_pos_boxes,
        adaptive=True
    )

    val_dset = FoilFeatureDataset(
        'data/foilGWVQA.val.json',
        dictionary,
        img_id2train,
        img_id2val,
        img_id2test,
        train_img_features,
        train_img_spatials,
        train_img_pos_boxes,
        val_img_features,
        val_img_spatials,
        val_img_pos_boxes,
        test2015_img_features,
        test2015_img_spatials,
        test2015_img_pos_boxes,
        adaptive=True
    )

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, 3129, args.op, args.gamma).cuda()

    tfidf = None
    weights = None

    if args.tfidf:
        dict = Dictionary.load_from_file('data/dictionary.pkl')
        tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dict)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy', tfidf, weights)

    model = nn.DataParallel(model).cuda()

    batch_size = args.batch_size
    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

        # Fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        model.module.classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, train_dset.num_ans_candidates, .5)
        model.module.classifier = model.module.classifier.cuda()

    if args.use_both:  # use train & val splits to optimize
        if args.use_vg:  # use a portion of Visual Genome dataset
            vg_dsets = [
                VisualGenomeFeatureDataset('train', \
                                           train_dset.features, train_dset.spatials, dictionary, adaptive=True,
                                           pos_boxes=train_dset.pos_boxes),
                VisualGenomeFeatureDataset('val', \
                                           val_dset.features, val_dset.spatials, dictionary, adaptive=True,
                                           pos_boxes=val_dset.pos_boxes)]
            trainval_dset = ConcatDataset([train_dset, val_dset] + vg_dsets)
        else:
            trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    train(model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
