import argparse
import os
import random

import csv

target_csv = os.path.join(os.getcwd(), "predic/out/csv/large_outputs.csv")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from model import Net
from config import Base
from downloads import download_weights


parser = argparse.ArgumentParser()
 
parser.add_argument(
    '-I',
    '--img',
    type=str,
    default='',
    help='Image name'
)

parser.add_argument(
    '-N',
    '--num',
    type=str,
    default='',
    help='run number'
)

parser.add_argument(
    '-T', 
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='',
    help='Checkpoint name'
)

args = parser.parse_args()

imgpath = "./evaluate/img/"+ args.img
outpath = "./predic/out"
modelsize = "L"
checkpoint = args.checkpoint_name if args.checkpoint_name else ""
targetcaptioncsv = './evaluate/captions.csv' 

config = Base() 
# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

if __name__ == '__main__':
    ckp_path = os.path.join(config.weights_dir, checkpoint)

    if args.img:
        assert os.path.isfile(imgpath), 'Image does not exist'
        img_list = [imgpath]
    else:
        assert os.path.isdir(imgpath), 'Input folder does not exist'
        img_list = [
            os.path.join(imgpath, f)
            for f in sorted(os.listdir(imgpath))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        assert img_list, 'No images found in input folder'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path, modelsize)

    checkpoint = torch.load(ckp_path, map_location=device)
    # Handle both raw state_dict and full checkpoint formats
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    #print("Missing keys:", missing)
    #print("Unexpected keys:", unexpected)

    model.eval()

    counter=0
    for path in img_list:
        counter+=1
        img = Image.open(path).convert('RGB')

        with torch.no_grad():
            caption, _ = model(img, args.temperature)

        plt.imshow(img)
        plt.title(caption)
        plt.axis('off')

        img_save_path = f'{os.path.splitext(os.path.basename(path))[0]}.jpg'
        plt.savefig(os.path.join(outpath, img_save_path), bbox_inches='tight')

        plt.clf()
        plt.close()

        file_exists = os.path.isfile(target_csv)
        imgname =f'{os.path.basename(path)}'
        targetcaption = next((row['caption'] for row in csv.DictReader(open(targetcaptioncsv)) if row['image'] == imgname), "Caption not found")
        
        with open(target_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["img_save_path", "caption","targetcaption"])

            writer.writerow([img_save_path, caption,targetcaption])
        if args.num:
            print(f'{counter}/{args.num}: {img_save_path}, {caption},{targetcaption}')
            if counter >=int(args.num):
                break
        else:
            print(f'{counter}: {img_save_path}, {caption},{targetcaption}')
        