# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path
import signal
import shutil
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm

from utils import builder, configurator, io, misc, ops, pipeline, recorder


def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/zoomnet/cod_zoomnet.py", type=str)
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs.json", type=str)
    parser.add_argument("--model-name", default = "ZoomNet", type=str)
    parser.add_argument("--batch-size", default = 4, type=int)
    parser.add_argument("--load-from", type=str) # trained weight
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)

    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.info is not None:
        config.experiment_tag = args.info
    if args.load_from is not None:
        config.load_from = args.load_from

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    tr_paths = {}
    for tr_dataset in config.datasets.train.path:
        if tr_dataset not in datasets_info:
            raise KeyError(f"{tr_dataset} not in {args.datasets_info}!!!")
        tr_paths[tr_dataset] = datasets_info[tr_dataset]
    config.datasets.train.path = tr_paths

    val_paths = {}
    for val_dataset in config.datasets.val.path:
        if val_dataset not in datasets_info:
            raise KeyError(f"{val_dataset} not in {args.datasets_info}!!!")
        val_paths[val_dataset] = datasets_info[val_dataset]
    config.datasets.val.path = val_paths

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            raise KeyError(f"{te_dataset} not in {args.datasets_info}!!!")
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    if args.resume_from is not None:
        config.resume_from = args.resume_from
        resume_proj_root = os.sep.join(args.resume_from.split("/")[:-3])
        if resume_proj_root.startswith("./"):
            resume_proj_root = resume_proj_root[2:]
        config.output_dir = os.path.join(config.proj_root, resume_proj_root)
        config.exp_name = args.resume_from.split("/")[-3]
    else:
        config.output_dir = os.path.join(config.proj_root, "output")
    config.path = misc.construct_path(output_dir=config.output_dir, exp_name=config.exp_name)
    return config


@recorder.TimeRecoder.decorator(start_msg="Test")
def test_once(
    model, data_loader, save_path, tta_setting, clip_range=None, show_bar=False, desc="[TE]", to_minmax=False
):
    model.eval()
    model.is_training = False
    cal_total_seg_metrics = recorder.CalTotalMetric()

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        batch_images = misc.to_device(batch["data"], device=model.device)
        if tta_setting.enable:
            logits = pipeline.test_aug(
                model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
            )
        else:
            logits = model(data=batch_images)
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

        for i, pred in enumerate(probs):
            mask_path = batch["info"]["mask_path"][i]
            mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape

            # here, sometimes, we can resize the prediciton to the shape of the mask's shape
            pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")

            if clip_range is not None:
                pred = ops.clip_to_normalize(pred, clip_range=clip_range)

            if save_path:  # 这里的save_path包含了数据集名字
                ops.save_array_as_image(
                    data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path, to_minmax=to_minmax
                )
            else:
                pass

            pred = (pred * 255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


@torch.no_grad()
def testing(model, cfg):
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):
        pred_save_path = os.path.join(cfg.path.save, data_name)
        cfg.te_logger.record(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
        )
        cfg.te_logger.record(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")
        cfg.excel_logger(row_data=seg_results, dataset_name=data_name, method_name=cfg.exp_name)

@torch.no_grad()
def validation(model, cfg, current_epoch, val_loader):
    model.eval()
    # model.is_training = True
    total_loss = 0.0
    total_samples = 0

    for data_name, data_path, loader in val_loader:
        cfg.val_logger.record(f"Validation on dataset {data_name} at epoch {current_epoch}")

        # Loop over validation data
        for batch in loader:
            batch_data = misc.to_device(data=batch["data"], device=model.device)
            
            # Forward pass
            outputs, loss, _ = model.train_forward(
                data=batch_data,
                curr=dict(
                    iter_percentage=0.0,  # Not needed during validation
                    epoch_percentage=0.0,
                ),
            )
            batch_size = batch_data["mask"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Compute average loss over the validation set
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        cfg.val_logger.record(f"[Epoch {current_epoch}] Val Loss on {data_name}: {avg_loss:.4f}")

        # Optionally, log the loss to WandB or other loggers
        if cfg.log_interval.get('wandb', 0) > 0:
            cfg.wandb_logger.record_val_curve("validate/loss", avg_loss, current_epoch+1)
    return avg_loss


def training(model, cfg) -> pipeline.ModelEma:
    tr_loader = pipeline.get_tr_loader(cfg)
    val_loader = list(pipeline.get_val_loader(cfg)) 
    cfg.epoch_length = len(tr_loader)

    metrics_calculator = recorder.MetricsCalculator()
    best_val_loss = None
    best_epoch = None

    if not cfg.train.epoch_based:
        cfg.train.num_epochs = (cfg.train.num_iters + cfg.epoch_length) // cfg.epoch_length
    else:
        cfg.train.num_iters = cfg.train.num_epochs * cfg.epoch_length

    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=cfg.train.num_iters,
        epoch_length=cfg.epoch_length,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)

    # cfg.tr_logger.record(f"Scheduler:\n{scheduler}")
    cfg.tr_logger.record(f"Optimizer:\n{optimizer}")
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)

    start_epoch = 0
    if cfg.resume_from:
        params_in_checkpoint = io.load_specific_params(
            load_path=cfg.resume_from, names=["model", "optimizer", "scaler", "start_epoch"]
        )
        model.load_state_dict(params_in_checkpoint["model"])
        optimizer.load_state_dict(state_dict=params_in_checkpoint["optimizer"])
        scaler.load_state_dict(state_dict=params_in_checkpoint["scaler"])
        start_epoch = params_in_checkpoint["start_epoch"]

    model_ema = None
    if cfg.train.ema.enable:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = pipeline.ModelEma(
            model.module if hasattr(model, "module") else model,
            decay=cfg.train.ema.decay,
            device="cpu" if cfg.train.ema.force_cpu else None,
            has_set=False,
        )
        if cfg.resume_from:
            params_in_checkpoint = io.load_specific_params(load_path=cfg.resume_from, names=["model_ema"])
            model_ema.module.load_state_dict(state_dict=params_in_checkpoint["model_ema"])

    time_logger = recorder.TimeRecoder()
    loss_recorder = recorder.AvgMeter()

    curr_iter = 0
    try: 
        for curr_epoch in range(start_epoch, cfg.train.num_epochs):
            cfg.tr_logger.record(f"Exp_Name: {cfg.exp_name}")
            time_logger.start(msg="An Epoch Start...")

            loss_recorder.reset()
            metrics_calculator.reset() 
            model.train()
            model.is_training = True

            # an epoch starts
            for batch_idx, batch in enumerate(tr_loader):
                scheduler.step(curr_idx=curr_iter)  # update learning rate

                batch_data = misc.to_device(data=batch["data"], device=model.device)
                with torch.amp.autocast('cuda',enabled=cfg.train.use_amp):
                    probs, loss, loss_str = model(
                        data=batch_data,
                        curr=dict(
                            iter_percentage=curr_iter / cfg.train.num_iters,
                            epoch_percentage=curr_epoch / cfg.train.num_epochs,
                        ),
                    )
                    loss = loss / cfg.train.grad_acc_step

                scaler.scale(loss).backward()

                if cfg.train.grad_clip.enable:
                    scaler.unscale_(optimizer)
                    ops.clip_grad(
                        chain(*[group["params"] for group in optimizer.param_groups]),
                        mode=cfg.train.grad_clip.mode,
                        clip_cfg=cfg.train.grad_clip.cfg,
                    )

                # Accumulates scaled gradients.
                if (curr_iter + 1) % cfg.train.grad_acc_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=cfg.train.optimizer.set_to_none)

                    if model_ema is not None:
                        model_ema.update(model)

                item_loss = loss.item()
                data_shape = batch_data["mask"].shape
                loss_recorder.update(value=item_loss, num=data_shape[0])

                # Update metrics calculator
                outputs = probs.get('sal', probs.get('prob_map'))
                if outputs is None:
                    raise KeyError("Model outputs do not contain keys. Adjust the key accordingly.")
                targets = batch_data['mask']
                metrics_calculator.update(outputs=outputs, targets=targets)

                # log iteration loss in terminal
                if cfg.log_interval.txt > 0 and (
                        curr_iter % cfg.log_interval.txt == 0
                        or (curr_iter + 1) % cfg.epoch_length == 0
                        or (curr_iter + 1) == cfg.train.num_iters
                ):
                    msg = " | ".join(
                        [
                            f"I:{curr_iter}:{cfg.train.num_iters} {batch_idx}/{cfg.epoch_length} {curr_epoch}/{cfg.train.num_epochs}",
                            f"Lr:{optimizer.lr_string()}",
                            f"M:{loss_recorder.avg:.5f}/C:{item_loss:.5f}",
                            f"{list(data_shape)}",
                            f"{loss_str}",
                        ]
                    )
                    cfg.tr_logger.record(msg)

                curr_iter += 1
                if curr_iter >= cfg.train.num_iters:
                    break
            # an epoch ends

            # Compute and log epoch metrics
            epoch_metrics = metrics_calculator.compute_metrics()
            epoch_metrics['avg_loss'] =  loss_recorder.avg
            epoch_metrics['lr_0'] = optimizer.lr_groups()[0]
            epoch_metrics['lr_1'] = optimizer.lr_groups()[1]
            # TODO: log lr 
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in epoch_metrics.items()])
            cfg.tr_logger.record(f"Epoch {curr_epoch} Metrics: {metrics_str}")
            if cfg.log_interval.wandb > 0:
                cfg.wandb_logger.record_epoch_metrics(epoch_metrics, curr_epoch+1)

            # After each epoch, or every N epochs, perform validation
            if curr_epoch == 0 or (curr_epoch + 1) % cfg.val.interval == 0:
                val_loss = validation(model=model, cfg=cfg, current_epoch=curr_epoch, val_loader=val_loader)
                # save best weight
                if best_val_loss is None or val_loss < best_val_loss: 
                    best_path = os.path.join(cfg.path.pth, f'best.pth')
                    io.save_weight(model=model, save_path=best_path)
                    best_val_loss = val_loss
                    best_epoch = curr_epoch

            # Save checkpoint every M epochs
            if (curr_epoch + 1) % cfg.log_interval.save_checkpoint == 0:
                checkpoint_path = os.path.join(cfg.path.pth, f'state_epoch_{curr_epoch + 1}.pth')
                io.save_weight(model=model, save_path=checkpoint_path)
                if model_ema is not None:
                    ema_checkpoint_path = os.path.join(cfg.path.pth, f'state_epoch_{curr_epoch + 1}_ema.pth')
                    io.save_weight(model=model_ema.module, save_path=ema_checkpoint_path)
                

            # log sample images
            if curr_epoch % 2 == 0:  
                # plot some batches of the training phase, save in path
                # path = os.path.join(cfg.path.pth_log, f"train_epoch_{curr_epoch}.png")
                # recorder.plot_results(
                #     dict(**probs, **batch_data),
                #     save_path= path,
                # )

                if cfg.log_interval.wandb > 0 :
                    cfg.wandb_logger.record_images(dict(**probs, **batch_data), curr_epoch+1)

                # log predictions in wandb
                # if cfg.log_interval.wandb > 0:
                #     cfg.wandb_logger.record_prediction_table(
                #         epoch=curr_epoch+1, 
                #         images=batch_data["image1.0"],   # Input image
                #         masks=batch_data["mask"],        # Ground truth mask
                #         sals=probs['sal'])              # Model saliency map

            if curr_epoch == 0 and model_ema is not None:
                model_ema.set(model=model)  # using a better initial model state

            # save all params for (curr_epoch+1)th epoch
            io.save_params(
                exp_name=cfg.exp_name,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scaler=scaler,
                next_epoch=curr_epoch + 1,
                total_epoch=cfg.train.num_epochs,
                save_num_models=cfg.train.save_num_models,
                full_net_path=cfg.path.final_full_net,
                state_net_path=cfg.path.final_state_net,
            )
            time_logger.now(pre_msg="An Epoch End...")

            if curr_iter >= cfg.train.num_iters:
                break
            
            # only save the last weight
            io.save_weight(model=model, save_path=cfg.path.final_state_net)
            # cfg.tr_logger.record(f"Best Performance on: Epoch {best_epoch} with Val Loss {best_val_loss}")
            return model_ema
    except KeyboardInterrupt:
        cfg.tr_logger.record("Training interrupted. Saving model weights...")
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        if best_epoch is not None and best_val_loss is not None:
            cfg.tr_logger.record(f"Best Performance on: Epoch {best_epoch} with Val Loss {best_val_loss}")
        cfg.tr_logger.record(f"Model weights saved. Killing process {os.getpid()}...")
        os.kill(os.getpid(), signal.SIGTERM) 


def main():
    cfg = parse_config()

    if not cfg.resume_from:
        misc.pre_mkdir(path_config=cfg.path)
        with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
            f.write(cfg.pretty_text)
        shutil.copy(__file__, cfg.path.trainer_copy)

    cfg.tr_logger = recorder.TxtLogger(cfg.path.tr_log)
    cfg.te_logger = recorder.TxtLogger(cfg.path.te_log)
    cfg.val_logger = recorder.TxtLogger(cfg.path.val_log)
    # TODO: Excel -> CSV(More flexible and simple)
    cfg.excel_logger = recorder.MetricExcelRecorder(
        xlsx_path=cfg.path.excel, dataset_names=sorted([x for x in cfg.datasets.test.path.keys()])
    )
    if cfg.log_interval.wandb > 0:
        cfg.wandb_logger = recorder.WandbRecorder(project_name="ZoomNet", config=cfg)

    if cfg.base_seed >= 0:
        cfg.tr_logger.record(f"{cfg.proj_root} with base_seed {cfg.base_seed}")
    else:
        cfg.tr_logger.record(f"{cfg.proj_root} without fixed seed")

    if cfg.train.ms.enable:
        deterministic = True
    else:
        deterministic = cfg.deterministic
    misc.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=deterministic)

    model, model_code = builder.build_obj_from_registry(
        registry_name="MODELS", obj_name=cfg.model_name, return_code=True
    )
    cfg.tr_logger.record(model_code)
    # cfg.tr_logger.record(model)

    model.device = "cuda:0"
    model.to(model.device)

    if cfg.load_from:
        model_ema = io.load_weight(model=model, load_path=cfg.load_from)
    else:
        model_ema = training(model=model, cfg=cfg)

    if cfg.has_test:
        if model_ema is not None:
            testing(model=model_ema.module, cfg=cfg)
            if cfg.train.ema.keep_original_test:
                cfg.tr_logger.record(f"The results from original model will overwrite the model_ema's.")
                testing(model=model, cfg=cfg)
        else:
            testing(model=model, cfg=cfg)

    cfg.tr_logger.record("End training...")


if __name__ == "__main__":
    main()
