import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from torch.optim import AdamW
import logging

from pathlib import Path
import datetime
from math import cos, pi
import gc
import argparse
from CLSEnhancer import  CLSEnhancer


def get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        min_lr=1e-6,
        last_epoch=-1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            scale = (optimizer.defaults['lr'] - min_lr) * 0.5 * (1.0 + cos(pi * progress))
            return (scale + min_lr) / optimizer.defaults['lr']

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_dir = "./logs-hazy"
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(f"{log_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()



class CLSDataset(Dataset):
    def __init__(self, lq_dir, hq_dir):
        self.lq_files = sorted(Path(lq_dir).glob("*.pt"))
        self.hq_files = sorted(Path(hq_dir).glob("*.pt"))
        assert len(self.lq_files) == len(self.hq_files), "Mismatched number of LQ and HQ samples"

    def __len__(self):
        return len(self.lq_files)

    def __getitem__(self, idx):
        return (
            torch.load(self.lq_files[idx], weights_only=True).squeeze(),
            torch.load(self.hq_files[idx], weights_only=True).squeeze()
        )



class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.cos = nn.CosineEmbeddingLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * self.cos(
            pred, target, torch.ones(target.size(0), device=target.device)
        )



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "lq_dir": args.lq_dir,
        "hq_dir": args.hq_dir,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "min_lr": args.min_lr,
        "save_dir": args.save_dir,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers
    }

    dataset = CLSDataset(config["lq_dir"], config["hq_dir"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    model = CLSEnhancer(hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]).to(device)
    criterion = HybridLoss(alpha=0.7)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
        min_lr=config["min_lr"]
    )

    start_epoch = 0
    best_loss = float('inf')
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('loss', best_loss)
        scheduler.last_epoch = start_epoch * len(train_loader)
        logger.info(f"Loaded checkpoint from epoch {start_epoch}, loss={best_loss:.4f}")

    os.makedirs(config["save_dir"], exist_ok=True)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (lq, hq) in enumerate(train_loader):
            lq, hq = lq.to(device), hq.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = criterion(model(lq), hq)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if epoch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch [{epoch + 1}/{config['num_epochs']}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.7f} LR: {current_lr:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch + 1}/{config['num_epochs']}] Average Loss: {avg_loss:.7f} | LR: {current_lr:.2e}")

        if epoch % 1000 == 0:
            save_path = f"{config['save_dir']}/epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, save_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{config['save_dir']}/best_model.pth")
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLS Enhancer Model")
    parser.add_argument("--lq_dir", type=str, default=r"---",#LQ图片路径
                        help="Path to LQCLS folder")
    parser.add_argument("--hq_dir", type=str, default=r"",  #GT图片路径
                        help="Path to GTCLS folder")
    parser.add_argument("--save_dir", type=str, default=r"",#模型保存路径
                        help="Directory to save models")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0", help="GPU device id, e.g. '0' or '0,1'")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    train(args)

