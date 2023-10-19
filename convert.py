# python3 convert.py input-model.tar output-model.pth
import argparse

import torch

from backbone.repvgg import repvgg_model_convert
from model import RepNet6D, RepNet5D

parser = argparse.ArgumentParser(description='RepNet6D Conversion')
parser.add_argument('--load', dest='load', default='output/snapshots/AFLW2000_20231014151652_bs16/_epoch_4.tar', type=str, help='path to the weights file')
parser.add_argument('--save', dest='save', default='epoch_4.pth', type=str, help='path to the weights file')
parser.add_argument('--arch', dest='arch', help='Name of model snapshot.',  default='RepVGG-B1g4')



def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def convert():
    args = parser.parse_args()

    print('Loading model.')
    model = RepNet6D(backbone_name=args.arch,
                            backbone_file='',
                            deploy=False,
                            pretrained=False)

    # Load snapshot
    saved_state_dict = torch.load(args.load)

    load_filtered_state_dict(model, saved_state_dict['model_state_dict'])
    print('Converting model.')
    repvgg_model_convert(model, save_path=args.save)
    print('Done.')

if __name__ == '__main__':
    convert()