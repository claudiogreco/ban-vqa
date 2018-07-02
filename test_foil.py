"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import _pickle as cPickle
import argparse
import os

import h5py
import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import base_model
import utils
from classifier import SimpleClassifier
from dataset import Dictionary, FoilFeatureDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban/model_epoch12.pth')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logits', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx[0]]


def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    # qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(max_value=N)
    for v, b, q, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        logits, att = model(v, b, q, None)
        pred[idx:idx + batch_size, :].copy_(logits.data)
        # qIds[idx:idx + batch_size].copy_(i)
        idx += batch_size
        if args.debug:
            print(get_question(q.data[0], dataloader))
            print(get_answer(logits.data[0], dataloader))
        print(logits.data)
    bar.update(idx)
    return pred  # , qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i]
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


if __name__ == '__main__':
    args = parse_args()

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

    eval_dset = FoilFeatureDataset(
        'data/foilGWVQA.test.json',
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

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, 3129, args.op, args.gamma).cuda()
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)


    def process(args, model, eval_loader):
        model_path = args.input + '/model%s.pth' % ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

        print('loading %s' % model_path)
        model_data = torch.load(model_path)
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))
        model.train(False)

        model.module.classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, eval_dset.num_ans_candidates, .5)
        model.module.classifier = model.module.classifier.cuda()

        # logits, qIds = get_logits(model, eval_loader)
        logits = get_logits(model, eval_loader)
        print(logits)

        # results = make_json(logits, qIds, eval_loader)
        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)
        print("step 5")

        if args.logits:
            utils.create_dir('logits/' + model_label)
            torch.save(logits, 'logits/' + model_label + '/logits%d.pth' % args.index)

            # utils.create_dir(args.output)
            # if 0 <= args.epoch:
            #     model_label += '_epoch%d' % args.epoch

            # with open(args.output+'/%s_%s.json' \
            #     % (args.split, model_label), 'w') as f:
            #     json.dump(results, f)

            # process(args, model, eval_loader)
