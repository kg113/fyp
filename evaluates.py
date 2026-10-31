import argparse
import os
import random

import csv

large_csv = os.path.join(os.getcwd(), "predic/out/csv/large_outputs.csv")
small_csv = os.path.join(os.getcwd(), "predic/out/csv/small_outputs.csv")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data import random_split

from dataset import MiniFlickrDataset
from model import Net
from config import Base
from downloads import download_weights



parser = argparse.ArgumentParser()

parser.add_argument(
    '-T',
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

args = parser.parse_args()

imgpath = "./evaluate/img/"
targetcaptioncsv = './evaluate/captions.csv' 
outpath = "./evaluate/out"
checkpoint = "model.pt"
modelsize = 'L'
config = Base()

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

if __name__ == '__main__':
    ckp_path = os.path.join(config.weights_dir, checkpoint)

    assert os.path.isdir(imgpath), 'Test image folder does not exist'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if not os.path.exists(os.path.dirname(large_csv)):
        os.makedirs(os.path.dirname(large_csv))

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

    dataset = MiniFlickrDataset(os.path.join('evaluate', 'processed', 'dataset.pkl'))

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    _, _, test_dataset = random_split(dataset, [config.train_size, config.val_size, config.test_size])

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path, modelsize)

    state = torch.load(ckp_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    #print("Missing keys:", missing)
    #print("Unexpected keys:", unexpected)

    model.eval()

    target_csv = large_csv if modelsize == "L" else small_csv
    file_exists = os.path.isfile(target_csv)

    for img_name, _, _ in test_dataset:
        img = Image.open(os.path.join(imgpath, img_name)).convert('RGB')

        with torch.no_grad():
            caption, _ = model(img, args.temperature)

        plt.imshow(img)
        plt.title(caption)
        plt.axis('off')

        img_save_path = f'{os.path.splitext(os.path.basename(img_name))[0]}.jpg'
        plt.savefig(os.path.join(outpath, img_save_path), bbox_inches='tight')

        plt.clf()
        plt.close()
        targetcaption = next(row['caption'] for row in csv.DictReader(open(targetcaptioncsv)) if row['image'] == img_name)
        with open(target_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["img_save_path", "caption","targetcaption"])
                file_exists = True

            writer.writerow([img_save_path, caption,targetcaption])
        print(f'{img_save_path}, {caption}')