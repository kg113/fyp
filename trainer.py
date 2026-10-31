'''
    Module contains Trainer used in training and testing processes.
'''

import io
import os

import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering globally
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scaler,
        scheduler, 
        train_loader, 
        valid_loader, 
        test_path='', 
        ckp_path='', 
        device='cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_path = test_path
        self.ckp_path = ckp_path
        self.device = device

        # Initialize attributes with defaults
        self.cur_lr = self.optimizer.param_groups[0]['lr']
        self.epoch = 0
        self.train_loss = []
        self.valid_loss = []
        self.test_result = None

        # load checkpoint (will override defaults if checkpoint exists)
        if os.path.isfile(ckp_path):
            self._load_ckp(
                ckp_path, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                scaler=scaler, 
                epoch=True,
                train_loss=True, 
                valid_loss=True, 
                device=device
            )
                
    def train_epoch(self):
        self.model.train()
        self.epoch += 1

        total_loss = 0

        loop = tqdm(self.train_loader, total=len(self.train_loader))
        loop.set_description(f'Epoch: {self.epoch} | Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

            img_emb, cap, att_mask = img_emb.to(self.device), cap.to(self.device), att_mask.to(self.device)

            with torch.amp.autocast('cuda'):
                loss = self.model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            total_loss += loss.item()

            loop.set_description(f'Epoch: {self.epoch} | Loss: {total_loss / (batch_idx + 1):.3f}')
            loop.refresh()

        self.cur_lr = self.optimizer.param_groups[0]['lr']
        self.train_loss.append(total_loss / (batch_idx + 1))

        self.scheduler.step()
    
    def valid_epoch(self):
        self.model.eval()

        total_loss = 0

        loop = tqdm(self.valid_loader, total=len(self.valid_loader))
        loop.set_description(f'Validation Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

            img_emb, cap, att_mask = img_emb.to(self.device), cap.to(self.device), att_mask.to(self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):

                    loss = self.model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)

                    total_loss += loss.item()
                    
                    loop.set_description(f'Validation Loss: {total_loss / (batch_idx + 1):.3f}')
                    loop.refresh()

        self.valid_loss.append(total_loss / (batch_idx + 1))

    def _clean_caption(self, text):
        """Post-process generated caption to remove artifact punctuation that hurts metrics."""
        if not isinstance(text, str):
            text = str(text)
        # Strip trailing loose periods/spaces
        text = text.replace('. .', '').strip()
        while text.endswith('.'):
            text = text[:-1].strip()
        if text:
            text = text[0].upper() + text[1:] + '.'
        return text

    def test_step(self):
        """Test on images in the folder with robust grid sizing and text cleanup."""
        self.model.eval()

        # Get all images
        image_files = sorted([f for f in os.listdir(self.test_path) if f.endswith('.jpg')])
        num_examples = min(len(image_files), 100)  # limit to 100 for safety
        
        if num_examples == 0:
            return

        ncols = 2
        nrows = (num_examples + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
        axs = np.atleast_2d(axs)

        for idx, img_name in enumerate(image_files[:num_examples]):
            img = Image.open(os.path.join(self.test_path, img_name))

            with torch.no_grad():
                caption, _ = self.model(img)

            cleaned_caption = self._clean_caption(caption)

            row, col = idx // ncols, idx % ncols
            axs[row, col].imshow(img)
            axs[row, col].set_title(cleaned_caption, usetex=False, fontsize=10)
            axs[row, col].axis('off')
            
        # Hide any unused subplots in the grid
        for idx in range(num_examples, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axs[row, col].axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        plt.close(fig)

        self.test_result = Image.open(io.BytesIO(buf.getvalue()))

    def get_training_data(self):
        return {
            'train_loss': self.train_loss, 
            'valid_loss': self.valid_loss, 
            'lr': self.cur_lr, 
            'examples': self.test_result    
        }

    def save_ckp(self, ckp_path):
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'tloss': self.train_loss,
                'vloss': self.valid_loss
            }, 
            ckp_path
        )   

    def _load_ckp(
        self, 
        checkpoint_fpath,
        optimizer=False, 
        scheduler=False, 
        scaler=False, 
        epoch=False, 
        train_loss=False, 
        valid_loss=False, 
        device='cpu'
    ):
        '''
            Loads entire checkpoint from file safely with weights_only=False.
        '''
        checkpoint = torch.load(checkpoint_fpath, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            if epoch and 'tloss' in checkpoint:
                self.epoch = len(checkpoint['tloss'])

            if train_loss and 'tloss' in checkpoint:
                self.train_loss = checkpoint['tloss']

            if valid_loss and 'vloss' in checkpoint:
                self.valid_loss = checkpoint['vloss']
        else:
            print(f"Warning: Checkpoint format not recognized. Loading only model weights.")
            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")