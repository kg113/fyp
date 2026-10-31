import os
import pickle
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

if __name__ == '__main__':
    SEED = 100
    DATA_PATH = os.path.join(os.getcwd(), 'evaluate')
    BATCH_SIZE = 32  # Adjust based on GPU memory

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA Available:", torch.cuda.is_available())

    preprocessor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').vision_model.to(device)
    model.eval()

    df = pd.read_csv(os.path.join(DATA_PATH, 'captions.csv'))
    df.columns = [col.strip() for col in df.columns]

    # Clean up NaN captions directly in pandas before grouping
    df = df.dropna(subset=['caption'])

    # Group captions by image name
    grouped = df.groupby('image')['caption'].apply(list).reset_index()

    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(grouped), BATCH_SIZE)):
        batch_df = grouped.iloc[i : i + BATCH_SIZE]
        
        valid_images = []
        valid_img_names = []
        valid_captions = []

        # Load images for current batch
        for img_name, caps in zip(batch_df['image'], batch_df['caption']):
            img_path = os.path.join(DATA_PATH,'img', img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                valid_images.append(img)
                valid_img_names.append(img_name)
                valid_captions.append(caps)
            except Exception as e:
                print(f'Lack of image {img_name}: {e}')

        if not valid_images:
            continue

        # Batch GPU inference
        with torch.no_grad():
            img_prep = preprocessor(images=valid_images, return_tensors='pt').to(device)
            img_features = model(**img_prep).pooler_output
            img_features = img_features.cpu().numpy()

        # Append results
        for img_name, embedding, caps in zip(valid_img_names, img_features, valid_captions):
            for c in caps:
                if pd.isna(c):
                    continue
                caption_text = str(c).strip()
                results.append((img_name, embedding, caption_text))

    # Save to pickle
    os.makedirs(os.path.join(DATA_PATH, 'processed'), exist_ok=True)
    with open(os.path.join(DATA_PATH, 'processed', 'dataset.pkl'), 'wb') as f:
        pickle.dump(results, f)