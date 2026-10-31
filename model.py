'''
    Module contains final Model and all pieces of it.
'''
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

def sanitize_caption(text):
    '''
        Escapes special LaTeX/matplotlib characters that can cause parsing errors.
        Characters like $, _, ^, {, }, %, &, # trigger special modes in matplotlib.
    '''
    special_chars = {
        '$': r'\$',
        '_': r'\_',
        '^': r'\^',
        '{': r'\{',
        '}': r'\}',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
    }
    for char, escaped in special_chars.items():
        text = text.replace(char, escaped)
    return text

class ImageEncoder(nn.Module):
    '''
        Encodes image and returns it's embedding.
    '''

    def __init__(self, model, device='cpu'):
        super(ImageEncoder, self).__init__()
        
        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)
        
        return image_features.pooler_output

class Mapping(nn.Module):
    '''
        Maps image embedding to GPT-2 embedding.
    '''
    
    def __init__(
        self, 
        ep_len,
        num_layers,
        embed_size, 
        n_heads, 
        forward_expansion, 
        dropout, 
        device='cpu'
    ):
        super(Mapping, self).__init__()
        
        self.ep_len = ep_len
        self.embed_size = embed_size

        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=n_heads, 
                dim_feedforward=embed_size*forward_expansion, 
                dropout=dropout, 
                batch_first=True, 
                device=device
            ),
            num_layers=num_layers
        ).to(self.device)

        self.mapper = nn.Linear(embed_size, ep_len * embed_size).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size] 
                if train_mode else 
                [self.ep_len, self.embed_size]
            )
        ) # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class TextDecoder(nn.Module):
    '''
        Processes embedding into caption.
    '''

    def __init__(self, model, device='cpu'):
        super(TextDecoder, self).__init__()
        
        self.device = device
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(inputs_embeds=embedding, attention_mask=attention_mask)
        
        return text_features.logits

class Net(nn.Module):
    '''
        Final Model class. Puts all pieces together and generates caption based on image.
    '''
    
    def __init__(self, clip_model, text_model, ep_len, num_layers, n_heads, forward_expansion, dropout, max_len, device='cpu'):
        super(Net, self).__init__()

        self.device = device
        self.ep_len = ep_len

        self.ie = ImageEncoder(model=clip_model, device=device)
        self.mp = Mapping(ep_len=self.ep_len, num_layers=num_layers, embed_size=self.ie.model.config.hidden_size, n_heads=n_heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
        self.td = TextDecoder(model=text_model, device=device)
        
        assert self.ie.model.config.hidden_size == self.td.model.config.n_embd, "Embedding size of models mismatch"

        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.td.tokenizer.pad_token_id)

        self.freeze_layers()

    def freeze_layers(self):
        for p in [*list(self.ie.parameters()), *list(self.td.parameters())[14:-14]]:
            p.requires_grad = False

    def forward(self, img, temperature=1.0, beam_size=3, repetition_penalty=1.2):
        '''
            Caption generation for a single image using Beam Search.
        '''
        if temperature <= 0.0:
            temperature = 1.0

        with torch.no_grad():
            img_embedded = self.ie(img)
            img_mapped = self.mp(img_embedded) # (ep_len, embed_size)

            sos_id = self.td.tokenizer.bos_token_id
            eos_id = self.td.tokenizer.eos_token_id

            sos_emb = self.td.model.transformer.wte(torch.tensor(sos_id).to(self.device)).unsqueeze(0)
            start_emb = torch.cat([sos_emb, img_mapped], dim=0)

            # Initialize beam: list of tuples (tokens, score)
            beams = [([], 0.0)]

            for _ in range(self.max_len):
                next_beams = []
                all_finished = True

                for tokens, score in beams:
                    if len(tokens) > 0 and tokens[-1] == eos_id:
                        next_beams.append((tokens, score))
                        continue

                    all_finished = False

                    if len(tokens) > 0:
                        tok_tensor = torch.tensor(tokens).to(self.device)
                        tok_emb = self.td.model.transformer.wte(tok_tensor)
                        emb = torch.cat([start_emb, tok_emb], dim=0)
                    else:
                        emb = start_emb

                    pos_emb = self.td.model.transformer.wpe(torch.arange(emb.shape[0]).to(self.device))
                    emb += pos_emb

                    pred = self.td(emb.unsqueeze(0)) # (1, seq_len, vocab_size)
                    logits = pred[0, -1, :] / temperature

                    # CORRECTED Repetition Penalty:
                    # Penalize already generated tokens properly whether logits are positive or negative
                    for t in set(tokens):
                        if logits[t] > 0:
                            logits[t] /= repetition_penalty
                        else:
                            logits[t] *= repetition_penalty

                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                    for i in range(beam_size):
                        next_token = topk_indices[i].item()
                        token_score = topk_log_probs[i].item()
                        next_beams.append((tokens + [next_token], score + token_score))

                # Sort beams by score descending and keep top beam_size
                next_beams.sort(key=lambda x: x[1], reverse=True)
                beams = next_beams[:beam_size]

                if all_finished:
                    break

            best_tokens = beams[0][0]
            # Strip trailing EOS if present
            if best_tokens and best_tokens[-1] == eos_id:
                best_tokens = best_tokens[:-1]

            decoded = self.td.tokenizer.decode(best_tokens)
            decoded = decoded.strip()
            decoded = decoded[0].upper() + decoded[1:] if decoded else "No caption generated"
            decoded = sanitize_caption(decoded)

            return decoded, best_tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_mapped = self.mp(img_emb, train_mode=True)
        text_emb = self.td.model.transformer.wte(x)

        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat([torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1)

        pos_emb = self.td.model.transformer.wpe(torch.arange(x.shape[1]).to(self.td.device))
        pos_emb = pos_emb.expand_as(x)

        x += pos_emb

        res = self.td(x, attention_mask=x_mask)

        loss = self.criterion(
            res[:, self.ep_len:, :].reshape(-1, res.shape[-1]), 
            y.reshape(-1)
        )
        
        return loss

if __name__ == '__main__':
    for clip, text in [['openai/clip-vit-base-patch32', 'gpt2']]:
        m = Net(
            clip_model=clip,
            text_model=text,
            ep_len=3,
            num_layers=6,
            n_heads=16, 
            forward_expansion=4, 
            dropout=0.1, 
            max_len=20
        )

        m.eval()
        r = m(torch.randn(3, 224, 224))
        print(r)