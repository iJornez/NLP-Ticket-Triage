
import json, os, argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class MultiTask(nn.Module):
    def __init__(self, enc: str = "distilbert-base-uncased", n_topic: int = 6, n_sent: int = 3, p: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(p)
        self.topic = nn.Linear(h, n_topic)
        self.sent = nn.Linear(h, n_sent)
    def forward(self, input_ids, attention_mask):
        x = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.drop(x)
        return self.topic(x), self.sent(x)

class TicketDS(Dataset):
    def __init__(self, df, tokenizer, max_len, topic2id, sent2id):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.topic2id = topic2id
        self.sent2id = sent2id
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        out = self.tok(row["text"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in out.items()}
        item["topic"] = torch.tensor(self.topic2id[row["topic"]], dtype=torch.long)
        item["sent"] = torch.tensor(self.sent2id[row["sentiment"]], dtype=torch.long)
        return item

def train_epoch(model, loader, optim, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_lab = batch["topic"].to(device)
        s_lab = batch["sent"].to(device)
        t_logits, s_logits = model(input_ids, attention_mask)
        loss = ce(t_logits, t_lab) + ce(s_logits, s_lab)
        loss.backward()
        optim.step()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for batch in tqdm(loader, desc="valid", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_lab = batch["topic"].to(device)
        s_lab = batch["sent"].to(device)
        t_logits, s_logits = model(input_ids, attention_mask)
        loss = ce(t_logits, t_lab) + ce(s_logits, s_lab)
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/seed/seed.csv")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--artifacts", default="artifacts/model")
    args = ap.parse_args()

    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.train_csv)
    topics = sorted(df["topic"].unique().tolist())
    sents = sorted(df["sentiment"].unique().tolist())
    topic2id = {t:i for i,t in enumerate(topics)}
    sent2id = {s:i for i,s in enumerate(sents)}
    with open(Path(args.artifacts)/"label_maps.json","w") as f:
        json.dump({"topics": topics, "sentiment": sents}, f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = TicketDS(df, tokenizer, args.max_len, topic2id, sent2id)
    n_train = int(0.8*len(ds))
    n_valid = len(ds)-n_train
    train_ds, valid_ds = torch.utils.data.random_split(ds, [n_train, n_valid], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTask(enc=args.model_name, n_topic=len(topics), n_sent=len(sents)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, optim, device)
        ev = eval_epoch(model, valid_loader, device)
        print(f"epoch {epoch} train_loss={tr:.4f} valid_loss={ev:.4f}")
        if ev < best:
            best = ev
            # save
            tokenizer.save_pretrained(args.artifacts)
            torch.save(model.state_dict(), Path(args.artifacts)/"pytorch_model.bin")
            with open(Path(args.artifacts)/"model_config.json","w") as f:
                json.dump({"model_name": args.model_name, "max_len": args.max_len}, f)

    print("done. best_valid_loss=", best)

if __name__ == "__main__":
    main()
