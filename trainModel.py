
import uuid
import logging
import argparse
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import torch.distributed as dist
import numpy as np


class Trainer():
    def __init__(
        self,
        name,
        args: argparse.Namespace,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        train_sampler=None,
        scheduler=None,
        ema=None,
        tb_logger: torch.utils.tensorboard.writer.SummaryWriter = None,
        class_map: dict = None,
    ):
        super().__init__(name=name)
        self.trainer_id = "_".join(["supervised", str(uuid.uuid4())])
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.class_map = class_map

        try:
            self.rank = dist.get_rank()
        except ValueError:
            self.rank = 0
        except RuntimeError:
            self.rank = 0

        

    

    def _train_step(self, batch: Tuple):

        # Unpack batch
        img, targets = batch

        # Send inputs to device
        inputs = img.to(self.args.device)
        targets = targets.long().to(self.args.device)

        # Compute logits for labeled and unlabeled images
        logits = self.model(inputs)

        # Loss
        loss = self.criterion(logits, targets)
        
        return loss

    def _train_epoch(self, epoch: int):

        # Reset meters
        self.model.train()
        self.meters.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()

        # Set progress bar and unpack batches
        train_loader = self.train_loader
        p_bar = tqdm(range(len(train_loader)))

        for batch_idx, batch in enumerate(train_loader):

            # Zero the optimizer
            self.optimizer.zero_grad()

            # Train one batch and backpropagate
            loss = self._train_step(batch)
            loss.backward()

            # Step optimizer and update parameters for EMA
            self.optimizer.step()
            if self.ema:
                self.ema.update()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                )
            )
            p_bar.update()

            # Tensorboard batch writing
            loss_dict = {"train_loss": loss}
            batch_step = (epoch * len(train_loader)) + batch_idx
            if self.rank == 0:
                self.tb_logger.log_scalar_dict(
                    main_tag="step_loss", scalar_dict=loss_dict, step=batch_step
                )

            if batch_idx % 200 == 0:
                avg_metrics, mc_metrics = self.train_metrics.compute()
                print(avg_metrics)
                print(mc_metrics)

        # Step LR scheduler
        if self.scheduler:
            self.scheduler.step()

        # Compute epoch metrics and loss
        avg_metrics, mc_metrics = self.train_metrics.compute()
        loss = self.meters["train_loss"].avg

        # Set the epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = tag_scalar_dict = {"train_loss": loss}
        if self.rank == 0:
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
            )

            # Epoch Average Metric Logging
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_train_metrics", scalar_dict=avg_metrics, step=epoch_step
            )

            # Epoch Multiclass Metric Logging
            self.tb_logger.log_tensor_dict(
                main_tag="epoch_train_metrics",
                tensor_dict=mc_metrics,
                step=epoch_step,
                class_map=self.class_map,
            )

            # Logger Logging
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {loss:.6f}")
            self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
            self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        # return loss

    @torch.no_grad()
    def _val_step(self, batch: Tuple):

        # Unpack batch and send to device
        img, targets = batch
        img = img.float().to(self.args.device)
        targets = targets.long().to(self.args.device)

        # Forward pass through model
        logits = self.model(img)

        # Calculate validation loss
        loss = self.criterion(logits, targets)

        # Update running meters
        self.meters.update("validation_loss", loss.item(), logits.size()[0])

        # Update metrics
        self.val_metrics.update(preds=logits, targets=targets)

        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int):

        # Reset meters
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)))
        for batch_idx, batch in enumerate(self.val_loader):

            # Validate one batch
            loss = self._val_step(batch)

            # Update the progress bar
            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.val_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                )
            )
            p_bar.update()

        # Compute epoch metrics
        avg_metrics, mc_metrics = self.val_metrics.compute()
        loss = self.meters["validation_loss"].avg

        # Epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = {"validation_loss": loss}

        if self.rank == 0:
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
            )

            # Epoch Average Validation Metric Logging
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_val_metrics", scalar_dict=avg_metrics, step=epoch_step
            )

            # Epoch Multiclass Validation Metric Logging
            self.tb_logger.log_tensor_dict(
                main_tag="epoch_val_metrics",
                tensor_dict=mc_metrics,
                step=epoch_step,
                class_map=self.class_map,
            )

            # Logger Logging
            self.logger.info(f"Epoch {epoch + 1} - Validation Loss: {loss:.6f}")
            self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
            self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        return loss

    def train(self):
        if self.rank == 0:
            self.logger.info(
                f"Training {self.trainer_id} for {self.args.model.epochs} epochs."
            )
        for epoch in range(self.args.model.epochs):

            # Update the epoch in the DistributedSampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Train and validate one epoch
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(train_loss),
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.state_dict(),
            }

            self.checkpoint(epoch=epoch, logs=logs)