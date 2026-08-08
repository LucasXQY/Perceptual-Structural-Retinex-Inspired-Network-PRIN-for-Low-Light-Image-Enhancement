"""Training script for PRIN (paper Sec. 4.1.2 training protocol).

Trains the final PRIN model (paper Table 5 variant V2) on one of the LOL
benchmarks. Protocol: Adam, lr 1e-4, batch size 2, 2500 epochs, seed 42,
full-resolution inputs (no random crops), a checkpoint saved every epoch,
plus a resumable "latest_resume.pth" (model + optimizer + RNG states).

NOTE: the `models/` and `losses/` packages are released upon paper
acceptance; until then this script's imports will not resolve.

Usage:
    python train.py
(Edit the CONFIG block at the top of train() to switch datasets.)
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.lol_dataset import LOLDataset
from datasets.lolv2real_dataset import LOLv2RealDataset
from datasets.lolv2syn_dataset import LOLv2SynDataset
from models.prin import PRIN
from losses.prin_loss import prin_loss


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_logs(avg_logs: dict) -> str:
    keys_order = [
        "total", "recon", "ssim", "edge", "wav",
        "base", "illum", "A_l1", "A_tv", "exp", "col", "oe"
    ]
    parts = []
    for k in keys_order:
        if k in avg_logs:
            parts.append(f"{k}:{avg_logs[k]:.6f}")
    return " | ".join(parts)


def save_resume_checkpoint(path, epoch, model, optimizer, best_loss, best_epoch):
    ckpt = {
        "epoch": epoch,  # last completed epoch (e.g. epoch=12 means epoch 12 finished)
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, path)


def try_load_resume(path, device, model, optimizer):
    """
    Returns: start_epoch, best_loss, best_epoch, loaded(bool)
    start_epoch: the next epoch to run (e.g. resume starts at epoch 13)
    """
    if not os.path.exists(path):
        return 1, float("inf"), -1, False

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    # The optimizer state may fail to load after structural changes; guard it.
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"[WARN] Optimizer state load failed, will continue with fresh optimizer. Reason: {e}")

    # Restore RNG states (optional but recommended).
    try:
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all", None) is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    except Exception as e:
        print(f"[WARN] RNG state restore failed: {e}")

    last_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_epoch + 1
    best_loss = float(ckpt.get("best_loss", float("inf")))
    best_epoch = int(ckpt.get("best_epoch", -1))
    return start_epoch, best_loss, best_epoch, True


def build_train_dataset(dataset_name: str, low_dir: str, high_dir: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "lolv1":
        return LOLDataset(low_dir=low_dir, high_dir=high_dir, mode="train")
    if dataset_name == "lolv2real":
        return LOLv2RealDataset(low_dir=low_dir, high_dir=high_dir, mode="train")
    if dataset_name == "lolv2syn":
        return LOLv2SynDataset(low_dir=low_dir, high_dir=high_dir, mode="train")
    raise ValueError(
        f"Unsupported dataset_name={dataset_name!r}. "
        "Expected one of: lolv1, lolv2real, lolv2syn."
    )


def train():
    # ----------------- CONFIG -----------------
    seed = 42
    batch_size = 2
    lr = 1e-4
    num_epochs = 2500

    dataset_name = "lolv1"  # one of: lolv1, lolv2real, lolv2syn
    dataset_root = os.path.join("data", dataset_name)
    train_low_dir = os.path.join(dataset_root, "train", "low")
    train_high_dir = os.path.join(dataset_root, "train", "high")

    log_path = "train_log.txt"
    ckpt_dir = "checkpoints"
    save_every = 1  # save a plain state_dict checkpoint every N epochs

    # ====== resume settings ======
    resume = False  # set True to resume from resume_path
    resume_path = os.path.join(ckpt_dir, "latest_resume.pth")  # resume file (includes optimizer etc.)
    # =============================

    # ----------------- Setup -----------------
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    print("[INFO] Loading training dataset...")

    train_set = build_train_dataset(dataset_name, train_low_dir, train_high_dir)


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(f"[INFO] Dataset loaded: {len(train_set)} training samples.")

    # Model
    print("[INFO] Initializing model...")
    model = PRIN().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Best-so-far tracking
    best_epoch = -1
    best_loss = float("inf")

    # --------- try resume ---------
    start_epoch = 1
    if resume:
        start_epoch, best_loss, best_epoch, loaded = try_load_resume(
            resume_path, device, model, optimizer
        )
        if loaded:
            print(f"[INFO] Resumed from: {resume_path}")
            print(f"[INFO] Start epoch = {start_epoch}, best_epoch={best_epoch}, best_loss={best_loss:.6f}")
        else:
            print(f"[INFO] Resume enabled, but no resume file found at: {resume_path}. Start from scratch.")

    # Logging: append if resuming and the log already exists; otherwise rewrite.
    log_mode = "a" if (resume and os.path.exists(log_path)) else "w"
    with open(log_path, log_mode, encoding="utf-8") as f:
        if log_mode == "w":
            f.write("===== Training Log =====\n")
            f.write(f"Device: {device}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"LR: {lr}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Train low dir: {train_low_dir}\n")
            f.write(f"Train high dir: {train_high_dir}\n")
            f.write("========================\n\n")
        else:
            f.write("\n===== RESUME TRAINING =====\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Resume path: {resume_path}\n")
            f.write(f"Start epoch: {start_epoch}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best loss: {best_loss:.6f}\n")
            f.write("===========================\n")

    # ----------------- Training -----------------
    if start_epoch > num_epochs:
        print(f"[INFO] start_epoch({start_epoch}) > num_epochs({num_epochs}), nothing to do.")
        return

    print("[INFO] Start Training...")
    interrupted = False

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()

        sum_logs = {}
        n_batches = 0

        try:
            for low, high in train_loader:
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                r_out, l_out, enhanced = model(low)

                loss, logs = prin_loss(
                    low=low, gt=high,
                    r_out=r_out, l_out=l_out,
                    enhanced=enhanced
                )

                loss.backward()
                optimizer.step()

                n_batches += 1
                for k, v in logs.items():
                    val = float(v.item())
                    sum_logs[k] = sum_logs.get(k, 0.0) + val

        except KeyboardInterrupt:
            interrupted = True
            print("\n[WARN] KeyboardInterrupt detected! Saving resume checkpoint...")
            # Save the current epoch number; the next run resumes from epoch+1
            # (edit the resume file manually if you want to redo the partial epoch).
            save_resume_checkpoint(resume_path, epoch, model, optimizer, best_loss, best_epoch)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[INTERRUPT] Saved resume checkpoint at epoch={epoch}: {resume_path}\n")
            break

        avg_logs = {k: (v / max(n_batches, 1)) for k, v in sum_logs.items()}

        msg = f"Epoch [{epoch}/{num_epochs}] | {format_logs(avg_logs)} | time:{time.time()-t0:.1f}s"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # Update best (best_model.pth stays a plain state_dict, compatible with test.py)
        avg_total = avg_logs.get("total", None)
        if avg_total is not None and avg_total < best_loss:
            best_loss = avg_total
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[BEST UPDATE] epoch={best_epoch}, best_avg_total={best_loss:.6f}\n")

        # Save a resumable "latest_resume.pth" every epoch
        save_resume_checkpoint(resume_path, epoch, model, optimizer, best_loss, best_epoch)

        # Periodic plain checkpoint (still a plain state_dict)
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

    # Training finished
    if not interrupted:
        end_msg = (
            f"\n===== TRAIN DONE =====\n"
            f"Best Epoch: {best_epoch}\n"
            f"Best Avg Total Loss: {best_loss:.6f}\n"
            f"Saved best: {os.path.join(ckpt_dir, 'best_model.pth')}\n"
            f"Saved resume: {resume_path}\n"
        )
        print(end_msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(end_msg)
    else:
        print(f"[INFO] Training interrupted. Resume file saved at: {resume_path}")


if __name__ == "__main__":
    train()
