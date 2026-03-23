"""Training loops for EnsembleModeDuhamel and DenoisingDNN."""
import os
import re
import sys
import glob
import numpy as np
import torch
import time as _time
from datetime import datetime, timedelta
from models.ensemble_model import EnsembleModeDuhamel
from models.denoising_dnn import DenoisingDNN
from .loss import custom_loss, denoising_loss


def _log(msg, newline=False):
    """Overwrite current line; newline=True to preserve the line."""
    sys.stdout.write('\r' + msg + ' ' * 10)
    if newline:
        sys.stdout.write('\n')
    sys.stdout.flush()


def _cleanup_checkpoints(checkpoint_dir, title, best_epoch):
    """Remove all checkpoints except the best one."""
    pattern = os.path.join(checkpoint_dir, f"{title}_checkpoint_*.pth")
    best_path = os.path.normpath(
        os.path.join(checkpoint_dir, f"{title}_checkpoint_{best_epoch}.pth")
    )
    removed = 0
    for f in glob.glob(pattern):
        if os.path.normpath(f) != best_path:
            os.remove(f)
            removed += 1
    if removed:
        print(f"[Cleanup] Removed {removed} checkpoints, kept best epoch {best_epoch}.")


def train(X_train, y_train, X_mask_train,
          X_val, y_val, X_mask_val,
          num_epochs, batch_size, validation_batch_size,
          freq_list, dt, xi_init=None, uj_u1=None, num_node=None,
          checkpoint_dir=None, title=None,
          checkpoint_epoch=None,
          existing_checkpoint=False, device_allocate=None, ma_window=5):
    """Train EnsembleModeDuhamel model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = device_allocate or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    time_win_size = min(X_train.shape[-1], y_train.shape[-1])
    X_mask_train = X_mask_train[:, :, :time_win_size]

    model = EnsembleModeDuhamel(freq_list, dt, xi_init, uj_u1, num_node,
                                 device_allocate, ma_window=ma_window)

    start_epoch = 0
    if existing_checkpoint:
        model.load_state_dict(torch.load(existing_checkpoint))
        match = re.search(r'checkpoint_(\d+)', existing_checkpoint)
        if match:
            start_epoch = int(match.group(1))

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_mask_train = torch.from_numpy(X_mask_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    X_mask_val = torch.from_numpy(X_mask_val).float().to(device)

    if existing_checkpoint:
        lossdata = np.load(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses.npz')))
        val_loss = lossdata["val_loss"].tolist()
        train_loss = lossdata["train_loss"].tolist()
        best_val = min(val_loss) if val_loss else float('inf')
        best_epoch = (val_loss.index(best_val) + 1) * checkpoint_epoch if val_loss else 0
    else:
        val_loss, train_loss = [], []
        best_val = float('inf')
        best_epoch = 0

    num_steps = (X_train.shape[0] + batch_size - 1) // batch_size
    t_start = _time.time()
    for epoch in range(start_epoch, num_epochs):
        losses = []
        rand_perm = torch.randperm(X_train.shape[0])
        for si, step in enumerate(range(0, X_train.shape[0], batch_size)):
            batch_idx = rand_perm[step:step + batch_size]
            inputs = X_train[batch_idx]
            labels = y_train[batch_idx]
            outputs, _ = model(inputs)
            labels = labels[:, :, :outputs.shape[-1]]
            mask = X_mask_train[batch_idx, :, :outputs.shape[-1]]
            loss = custom_loss(outputs, labels, mask)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _log(f'[Train] Epoch {epoch+1}/{num_epochs} | Step {si+1}/{num_steps} | Loss: {loss.item():.3e}')

        avg_loss = sum(losses) / len(losses)
        train_loss.append(avg_loss)

        # ETA calculation
        elapsed = _time.time() - t_start
        done = epoch - start_epoch + 1
        remaining = (num_epochs - epoch - 1) * elapsed / done
        eta_str = (datetime.now() + timedelta(seconds=remaining)).strftime('%y/%m/%d %H:%M:%S')
        _log(f'[Train] Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.3e} | ETA: {eta_str}')

        if (epoch + 1) % checkpoint_epoch == 0:
            val_losses = []
            rand_batch = torch.randperm(X_val.shape[0])[:validation_batch_size]
            for j in rand_batch:
                x_v = X_val[j].unsqueeze(0)
                y_v = y_val[j:j + 1]
                m_v = X_mask_val[j].unsqueeze(0)
                pred, _ = model(x_v)
                vl = custom_loss(pred, y_v[:, :, :pred.shape[-1]], m_v)
                val_losses.append(vl.item())
            avg_vl = sum(val_losses) / len(val_losses)
            val_loss.append(avg_vl)

            # Track best model
            if avg_vl < best_val:
                best_val = avg_vl
                best_epoch = epoch + 1

            _log(f'[Train] Epoch {epoch+1}/{num_epochs} | Train: {avg_loss:.3e} | Val: {avg_vl:.3e} | Best: {best_epoch} | ETA: {eta_str}', newline=True)
            torch.save(model.state_dict(),
                       os.path.normpath(os.path.join(checkpoint_dir, f'{title}_checkpoint_{epoch+1}.pth')))
            np.savez(os.path.normpath(os.path.join(checkpoint_dir, f'{title}_losses.npz')),
                     train_loss=train_loss, val_loss=val_loss,
                     best_epoch=best_epoch, checkpoint_epoch=checkpoint_epoch)

    print()  # final newline
    np.savez(os.path.normpath(os.path.join(checkpoint_dir, f'{title}_losses.npz')),
             train_loss=train_loss, val_loss=val_loss,
             best_epoch=best_epoch, checkpoint_epoch=checkpoint_epoch)

    # Keep only the best checkpoint
    _cleanup_checkpoints(checkpoint_dir, title, best_epoch)
    print(f"[Train] Best model: epoch {best_epoch} (val_loss={best_val:.3e})")
    return model


def trainDN(Xn_train, Xc_train, X_mask_train,
            Xn_val, Xc_val, X_mask_val,
            num_epochs, batch_size, validation_batch_size,
            checkpoint_dir=None, title=None, checkpoint_epoch=None,
            existing_checkpoint=False, device_allocate=None):
    """Train DenoisingDNN model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = device_allocate or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    time_win_size = Xc_train.shape[-1]
    X_mask_train = X_mask_train[:, :, :time_win_size]

    model = DenoisingDNN(device_allocate)

    start_epoch = 0
    if existing_checkpoint:
        model.load_state_dict(torch.load(existing_checkpoint))
        match = re.search(r'checkpoint_(\d+)', existing_checkpoint)
        if match:
            start_epoch = int(match.group(1))

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    Xn_train = torch.from_numpy(Xn_train).float().to(device)
    Xc_train = torch.from_numpy(Xc_train).float().to(device)
    X_mask_train = torch.from_numpy(X_mask_train).float().to(device)
    Xn_val = torch.from_numpy(Xn_val).float().to(device)
    Xc_val = torch.from_numpy(Xc_val).float().to(device)
    X_mask_val = torch.from_numpy(X_mask_val).float().to(device)

    if existing_checkpoint:
        lossdata = np.load(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses.npz')))
        val_loss = lossdata["val_loss"].tolist()
        train_loss = lossdata["train_loss"].tolist()
        best_val = min(val_loss) if val_loss else float('inf')
        best_epoch = (val_loss.index(best_val) + 1) * checkpoint_epoch if val_loss else 0
    else:
        val_loss, train_loss = [], []
        best_val = float('inf')
        best_epoch = 0

    num_steps = (Xn_train.shape[0] + batch_size - 1) // batch_size
    t_start = _time.time()
    for epoch in range(start_epoch, num_epochs):
        losses = []
        rand_perm = torch.randperm(Xn_train.shape[0])
        for si, step in enumerate(range(0, Xn_train.shape[0], batch_size)):
            batch_idx = rand_perm[step:step + batch_size]
            inputs = Xn_train[batch_idx]
            labels = Xc_train[batch_idx]
            outputs = model(inputs)
            labels = labels[:, :, :outputs.shape[-1]]
            mask = X_mask_train[batch_idx, :, :outputs.shape[-1]]
            loss = denoising_loss(outputs, labels, mask)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _log(f'[TrainDN] Epoch {epoch+1}/{num_epochs} | Step {si+1}/{num_steps} | Loss: {loss.item():.3e}')

        avg_loss = sum(losses) / len(losses)
        train_loss.append(avg_loss)

        elapsed = _time.time() - t_start
        done = epoch - start_epoch + 1
        remaining = (num_epochs - epoch - 1) * elapsed / done
        eta_str = (datetime.now() + timedelta(seconds=remaining)).strftime('%y/%m/%d %H:%M:%S')
        _log(f'[TrainDN] Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.3e} | ETA: {eta_str}')

        if (epoch + 1) % checkpoint_epoch == 0:
            val_losses = []
            rand_batch = torch.randperm(Xn_val.shape[0])[:validation_batch_size]
            for j in rand_batch:
                x_v = Xn_val[j].unsqueeze(0)
                y_v = Xc_val[j].unsqueeze(0)
                m_v = X_mask_val[j].unsqueeze(0)
                pred = model(x_v)
                vl = denoising_loss(pred, y_v[:, :, :pred.shape[-1]], m_v)
                val_losses.append(vl.item())
            avg_vl = sum(val_losses) / len(val_losses)
            val_loss.append(avg_vl)

            # Track best model
            if avg_vl < best_val:
                best_val = avg_vl
                best_epoch = epoch + 1

            _log(f'[TrainDN] Epoch {epoch+1}/{num_epochs} | Train: {avg_loss:.3e} | Val: {avg_vl:.3e} | Best: {best_epoch} | ETA: {eta_str}', newline=True)
            torch.save(model.state_dict(),
                       os.path.normpath(os.path.join(checkpoint_dir, f'{title}_checkpoint_{epoch+1}.pth')))
            np.savez(os.path.normpath(os.path.join(checkpoint_dir, f'{title}_losses.npz')),
                     train_loss=train_loss, val_loss=val_loss,
                     best_epoch=best_epoch, checkpoint_epoch=checkpoint_epoch)

    print()  # final newline
    np.savez(os.path.normpath(os.path.join(checkpoint_dir, f'{title}_losses.npz')),
             train_loss=train_loss, val_loss=val_loss,
             best_epoch=best_epoch, checkpoint_epoch=checkpoint_epoch)

    # Keep only the best checkpoint
    _cleanup_checkpoints(checkpoint_dir, title, best_epoch)
    print(f"[TrainDN] Best model: epoch {best_epoch} (val_loss={best_val:.3e})")
    return model
