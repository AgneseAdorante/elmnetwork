import json
import os
import random
import time
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.sharding as jshard
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.datasets.cifar10_dvs.cifar10_dvs_dataset import CIFAR10DVSPreprocessedDataset
from src.datasets.dvs_gesture.dvs_dataset import (
    PreprocessedNpzDataset,
    RandomTranslateEventFrames,
)
from src.datasets.shd.shd_train_utils import shd_loss, shd_make_step
from src.models.elm_network import ELMNetwork
from src.training.reg_schedule import scaled_reg_config
from src.training.regularizer import ELMNetworkRegularizer
from src.training.train_utils import (
    calculate_model_cost_and_weight_stats,
    copy_model_to_cpu,
)


@hydra.main(
    version_base="1.3",
    config_path=".",
    config_name="curr_experiment_config",
)
def main(cfg: DictConfig):
    print("---------- Experiment started! ----------")

    # random delay for async start
    if not cfg.setup.debug_run:
        sleep_time = random.randint(1, 300)
        print(f"Sleeping: {sleep_time}s")
        time.sleep(sleep_time)
    else:
        print(f"Sleeping: 0s")

    ########## Logging ##########
    print("---------- Logging setup started: ----------")

    # artifacts directory
    experiment_base_dir = Path(os.getcwd())
    artifacts_dir = experiment_base_dir / "artifacts"
    os.makedirs(artifacts_dir)

    # get hydra config
    main_config = OmegaConf.to_container(cfg, resolve=True)
    with open(str(artifacts_dir / "main_config.json"), "w") as f:
        json.dump(main_config, f, indent=4)

    # wandb related setup
    if cfg.setup.wandb_logging:

        # login to wandb
        os.environ["WANDB_INIT_TIMEOUT"] = "300"
        entity = str(cfg.setup.wandb_entity_name)
        project_name = str(cfg.setup.wandb_project_name)
        with open(str(cfg.setup.wandb_api_key_path), "r") as file:
            api_key = file.read().strip()
        wandb.login(key=api_key)

        # create new run
        wandb.init(
            entity=entity,
            project=project_name,
            group=cfg.setup.wandb_group_name,
            config=main_config,
            save_code=False,
        )
        time.sleep(10)

    print("Experiment Base Directory:")
    print(experiment_base_dir)

    # Save the config properly
    print("Experiment Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ########## Seeding ##########
    print("---------- Setup started: ----------")

    curr_actual_seed = cfg.setup.seed
    print(f"Current Actual Seed: {curr_actual_seed}")

    # set memory prealloc jax
    parallel_executions = int(os.environ.get("CSTM_PRLL_EXEC", "1"))
    curr_gpu_mem_fraction = 0.95 * (1 / parallel_executions)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(curr_gpu_mem_fraction)
    print(f"GPU Memory Fraction: {curr_gpu_mem_fraction}")

    # check backend working
    jax.config.update("jax_default_matmul_precision", cfg.setup.matmul_precision)
    print(f"Jax matmul precision {cfg.setup.matmul_precision}.")
    print("Jax CUDA available: ", jax.default_backend() == "gpu")
    print("Jaxlib binding works: ", bool(jax.numpy.ones(()) == 1))
    print("Jax Num Devices: ", len(jax.devices()))

    # general seeding
    random.seed(curr_actual_seed)
    np.random.seed(curr_actual_seed)

    # train seeding
    key = jrandom.PRNGKey(curr_actual_seed)
    key, init_key, train_key = jrandom.split(key, 3)

    # evaluation seeding
    const_eval_seed = 12345
    eval_key = jrandom.PRNGKey(const_eval_seed)
    eval_key, stats_key, viz_key, valid_key, test_key = jrandom.split(eval_key, 5)

    ########## Data ##########
    print("---------- Data configuration started: ----------")

    # dataset selection: dvs_gesture | cifar10_dvs
    dataset_name = cfg.training.dataset_name
    direction_selective = (
        "ema" if cfg.training.direction_selective else False
    )
    shuffle_inputs = OmegaConf.select(cfg, "training.shuffle_inputs", default=False)
    permutation_seed = OmegaConf.select(cfg, "training.permutation_seed", default=42)

    # translation augmentation is dropped for the scrambled condition, where
    # shifting permuted pixels is not a shift of the image
    max_shift = OmegaConf.select(cfg, "training.translate_max_shift", default=0)
    train_transform = (
        RandomTranslateEventFrames(max_shift=max_shift) if max_shift > 0 else None
    )

    dataset_path = Path(cfg.setup.datasets_base_folder) / cfg.training.dataset_folder

    if dataset_name == "dvs_gesture":
        input_size = cfg.training.input_size
        train = PreprocessedNpzDataset(
            dataset_path,
            split="train",
            direction_selective=direction_selective,
            transform=train_transform,
            val_ratio=cfg.training.valid_fraction,
            seed=42,
            shuffle_inputs=shuffle_inputs,
            permutation_seed=permutation_seed,
            input_size=input_size,
        )
        val = PreprocessedNpzDataset(
            dataset_path,
            split="val",
            direction_selective=direction_selective,
            val_ratio=cfg.training.valid_fraction,
            seed=42,
            shuffle_inputs=shuffle_inputs,
            permutation_seed=permutation_seed,
            input_size=input_size,
        )
        test = PreprocessedNpzDataset(
            dataset_path,
            split="test",
            direction_selective=direction_selective,
            shuffle_inputs=shuffle_inputs,
            permutation_seed=permutation_seed,
            input_size=input_size,
        )
        data_num_classes = 11
    elif dataset_name == "cifar10_dvs":
        common = dict(
            val_ratio=cfg.training.valid_fraction,
            test_ratio=cfg.training.test_fraction,
            seed=42,
            spatial_bin_size=cfg.training.spatial_bin_size,
            direction_selective=direction_selective,
        )
        train = CIFAR10DVSPreprocessedDataset(
            dataset_path, split="train", transform=train_transform, **common
        )
        val = CIFAR10DVSPreprocessedDataset(dataset_path, split="val", **common)
        test = CIFAR10DVSPreprocessedDataset(dataset_path, split="test", **common)
        data_num_classes = 10
    else:
        raise ValueError(f"Unknown dataset_name {dataset_name!r}")

    train_dataset = DataLoader(
        train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # NOTE: eval sets are shuffled with a fixed generator. Samples are ordered by
    # class on disk, so an unshuffled loader truncated by debug_run would only
    # ever see the first class. The seed keeps the order identical across runs.
    valid_dataset = DataLoader(
        val,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(const_eval_seed),
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_dataset = DataLoader(
        test,
        batch_size=cfg.training.eval_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(const_eval_seed),
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    batches_per_epoch = len(train_dataset) if not cfg.setup.debug_run else 20
    valid_batches_per_epoch = len(valid_dataset) if not cfg.setup.debug_run else 20
    test_batches_per_epoch = len(test_dataset) if not cfg.setup.debug_run else 20

    example_batch = next(iter(test_dataset))
    data_num_time_bins = example_batch[0].shape[1]
    data_num_input_channel = example_batch[0].shape[2]

    ########## Model ##########
    print("---------- Model configuration started: ----------")

    # get the model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config["num_input"] = data_num_input_channel
    model_config["num_output"] = data_num_classes

    # model initialization
    model = ELMNetwork(
        **model_config,
        key=init_key,
    )

    # serialize network config
    with open(str(artifacts_dir / "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # trainable parameter specification
    train_params_filter = model.get_train_params_filter()

    # model visualization
    print("Model Architecture:")
    summary_str = str(model)
    print(summary_str)
    with open(str(artifacts_dir / "model_summary.txt"), "w") as f:
        f.write(summary_str)

    # model statistics
    print("Model Stats:")
    model_stats = calculate_model_cost_and_weight_stats(
        model,
        train_params_filter,
        data_shape=(1, data_num_input_channel),
        comp_key=stats_key,
        data_dtype=jnp.bool,
    )
    print(json.dumps(model_stats, indent=4))
    with open(str(artifacts_dir / "model_stats.json"), "w") as f:
        json.dump(model_stats, f, indent=4)
    model_cost = {
        "bytes accessed": model_stats["cost_analysis"]["bytes accessed"],
        "flops": model_stats["cost_analysis"]["flops"],
        "transcendentals": model_stats["cost_analysis"]["transcendentals"],
        "total_params": model_stats["total_params"],
        "total_trainable_params": model_stats["total_trainable_params"],
    }
    if cfg.setup.wandb_logging:
        wandb.log({"model_cost": model_cost})

    ########## Scheduler and Optimizer##########
    print("---------- Scheduler and Optimizer configuration started: ----------")

    # instantiate scheduler
    if cfg.training.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.training.learning_rate,
            warmup_steps=cfg.training.warmup_steps if not cfg.setup.debug_run else 5,
            decay_steps=batches_per_epoch * cfg.training.num_epochs,
            end_value=0.0,
        )
    else:
        raise ValueError("Unsupported scheduler")

    # instantiate optimizer
    if cfg.training.optimizer == "adam":
        optimizer = optax.adam(learning_rate=scheduler)
    elif cfg.training.optimizer == "adamax":
        optimizer = optax.adamax(learning_rate=scheduler)
    else:
        raise ValueError("Unsupported optimizer")

    # apply gradient clipping
    grad_clip = optax.clip_by_global_norm(cfg.training.clip_norm)
    gradient_transform = optax.chain(
        grad_clip,
        optimizer,
    )

    # optimizer state
    opt_state = gradient_transform.init(eqx.filter(model, train_params_filter))

    # setup regularizer
    reg_config = OmegaConf.select(cfg, "training.reg_config", default=None)
    if reg_config is not None:
        reg_config = OmegaConf.to_container(reg_config, resolve=True)
    reg_schedule_cfg = OmegaConf.select(cfg, "training.reg_schedule", default=None)
    if reg_schedule_cfg is not None:
        reg_schedule_cfg = OmegaConf.to_container(reg_schedule_cfg, resolve=True)

    # rebuilt per epoch when a ramp is configured, see the epoch loop below
    epoch_reg_config, reg_mult = scaled_reg_config(reg_config, 0.0, reg_schedule_cfg)
    regularizer = (
        None if epoch_reg_config is None else ELMNetworkRegularizer(epoch_reg_config)
    )

    ########## MULTI GPU SHARING ##########
    print("---------- Multi GPU sharding initialization: ----------")

    num_devices = len(jax.devices())
    mesh = jax.make_mesh((num_devices,), ("batch",))
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    # sharding model and opt state
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    ########## EVAL FUNCTION ##########
    print("---------- Eval function defined: ----------")

    def evaluate_model(
        model,
        eval_key,
        eval_iter,
        batch_size: int,
        num_classes: int,
        model_data_sharding=None,
        debug_log: bool = False,
        max_batches: int = None,
    ):
        eval_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        if model_data_sharding is not None:
            model = eqx.filter_shard(model, model_data_sharding[0])
        total = len(eval_iter) if max_batches is None else max_batches
        for step, data in tqdm(
            enumerate(eval_iter, 0), total=total, disable=not debug_log
        ):
            if max_batches is not None and step >= max_batches:
                break
            inputs, labels = data
            eval_key, inference_key = jrandom.split(eval_key, 2)

            # Conversion to JNP
            inputs = jnp.array(inputs, dtype=jnp.bool)
            labels = jnp.array(labels, dtype=jnp.int32)

            # sharding data
            if model_data_sharding is not None:
                inputs, labels = eqx.filter_shard(
                    (inputs, labels), model_data_sharding[1]
                )

            # Only compute loss
            loss, (logits, _) = shd_loss(
                model,
                inputs,
                labels,
                inference_key,
                batch_size,
                num_classes,
                inference=True,
                init_carry=None,
                return_logits=True,
                return_carry=False,
                label_smoothing=0.0,  # eval
                regularizer=None,  # eval
            )
            eval_loss += loss.item()
            num_batches += 1

            # Calculate accuracy
            predicted = jnp.argmax(logits, 1)
            correct_predictions += jnp.sum(predicted == labels)
            total_predictions += labels.shape[0]

        # divide by the batches actually seen, which max_batches may cap
        eval_loss = eval_loss / max(num_batches, 1)
        eval_accuracy = correct_predictions / max(total_predictions, 1)

        return eval_loss, eval_accuracy

    ########## Training START ##########
    print("---------- Training started: ----------")

    best_model_file_name = "best_model_state_dict.eqx"
    best_model_state_dict = copy_model_to_cpu(model)
    eqx.tree_serialise_leaves(
        str(artifacts_dir / best_model_file_name), best_model_state_dict
    )

    # set global variable
    train_step = 0
    train_start_time = time.time()
    best_valid_accuracy = 0.0
    best_model_epoch = 0
    best_model_train_accuracy = 0.0
    valid_accuracy = 0.0

    # termination by step, not by epoch
    for train_epoch in range(1, cfg.training.num_epochs + 1):
        # ------- START OF EPOCH -------

        # ramp scheduled regularizer strengths, fixed within an epoch so the
        # jitted step recompiles once per epoch rather than every step
        if reg_schedule_cfg is not None:
            reg_progress = (train_epoch - 1) / max(cfg.training.num_epochs - 1, 1)
            epoch_reg_config, reg_mult = scaled_reg_config(
                reg_config, reg_progress, reg_schedule_cfg
            )
            regularizer = (
                None
                if epoch_reg_config is None
                else ELMNetworkRegularizer(epoch_reg_config)
            )

        # running statistics
        train_total_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        # configure pbar
        pbar = tqdm(
            enumerate(train_dataset, 0),
            initial=train_step,
            total=batches_per_epoch * cfg.training.num_epochs,
            disable=not cfg.setup.debug_run,
        )
        for epoch_step, (inputs, labels) in pbar:
            # ------- START OF STEP -------

            # a torch DataLoader always yields the full dataset, so the epoch is
            # truncated here instead; keeps the step count consistent with the
            # decay_steps the LR schedule was built with
            if epoch_step >= batches_per_epoch:
                break

            # step tracking
            train_step += 1

            # convert data
            inputs = jnp.array(inputs, dtype=jnp.bool)
            labels = jnp.array(labels, dtype=jnp.int32)

            # sharding data
            inputs, labels = eqx.filter_shard((inputs, labels), data_sharding)

            # make step
            train_key, make_step_key = jrandom.split(train_key, 2)
            loss, logits, _, model, opt_state, _ = shd_make_step(
                model,
                inputs,
                labels,
                opt_state,
                make_step_key,
                batch_size=cfg.training.batch_size,
                num_classes=data_num_classes,
                train_params_filter=train_params_filter,
                gradient_transform=gradient_transform,
                init_carry=None,
                return_logits=True,
                return_grad_norms=False,
                return_carry=False,
                label_smoothing=cfg.training.label_smoothing,
                regularizer=regularizer,
            )

            # training went wrong
            if loss is None or jnp.isnan(loss):
                print("Loss is nan: exiting training.")
                break
            loss = loss.item()

            # update train loss
            train_total_loss += loss
            train_loss = train_total_loss / (epoch_step + 1)

            # update train accuracy
            predicted = jnp.argmax(logits, 1)
            train_correct_predictions += jnp.sum(predicted == labels)
            train_total_predictions += labels.shape[0]
            train_accuracy = train_correct_predictions / train_total_predictions

            # update progress bar
            pbar.set_description(
                (
                    f"Train Step: {train_step}, "
                    f"Train Loss: {train_loss:.5f}, "
                    f"Train Acc: {train_accuracy:.5f}"
                )
            )

            # ------- END OF STEP -------

        # training went wrong
        if loss is None or jnp.isnan(loss):
            break

        # ------- VALIDATION -------

        # evaluate model
        valid_loss, valid_accuracy = evaluate_model(
            model,
            valid_key,
            valid_dataset,
            cfg.training.batch_size,
            data_num_classes,
            model_data_sharding=(model_sharding, data_sharding),
            debug_log=cfg.setup.debug_run,
            max_batches=valid_batches_per_epoch,
        )

        # update best model
        if (
            not (valid_accuracy is None or jnp.isnan(valid_accuracy))
            and valid_accuracy > best_valid_accuracy
        ):
            best_valid_accuracy = valid_accuracy
            best_model_train_accuracy = train_accuracy
            best_model_epoch = train_epoch
            best_model_state_dict = copy_model_to_cpu(model)
            eqx.tree_serialise_leaves(
                str(artifacts_dir / best_model_file_name), best_model_state_dict
            )

        # logging
        train_time = time.time() - train_start_time
        train_log = {
            "train_epoch": int(train_epoch),
            "train_time": float(train_time),
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            f"train_accuracy": float(train_accuracy),
            f"valid_accuracy": float(valid_accuracy),
            "learning_rate": float(scheduler(train_step)),
            "best_model_epoch": int(best_model_epoch),
        }
        print(json.dumps(train_log, separators=(",", ":")))
        if cfg.setup.wandb_logging:
            wandb.log({"train_log": train_log})

        # ------- END OF EPOCH -------

    # ------- END OF TRAINING -------

    ########## Evaluation ##########
    print("---------- Evaluation started: ----------")

    # load the best model
    print(f"Loading best model from epoch: {best_model_epoch}...")
    model = eqx.tree_deserialise_leaves(
        str(artifacts_dir / best_model_file_name), model
    )

    # seeding & determinism
    random.seed(const_eval_seed)
    np.random.seed(const_eval_seed)
    eval_key = jrandom.PRNGKey(const_eval_seed)
    eval_key, stats_key, viz_key, valid_key, test_key = jrandom.split(eval_key, 5)

    ########## Quantitative Evaluation ##########
    print("---------- Quantitative evaluation started: ----------")

    # deterministic evaluation
    test_key, valid_test_key, test_test_key = jrandom.split(test_key, 3)

    # Calculate valid performance
    valid_loss, valid_accuracy = evaluate_model(
        model,
        valid_test_key,
        valid_dataset,
        cfg.training.batch_size,
        data_num_classes,
        model_data_sharding=(model_sharding, data_sharding),
        debug_log=cfg.setup.debug_run,
        max_batches=valid_batches_per_epoch,
    )

    # Calculate test performance
    test_loss, test_accuracy = evaluate_model(
        model,
        test_test_key,
        test_dataset,
        cfg.training.eval_batch_size,  # fixed size 8 will be shardable
        data_num_classes,
        model_data_sharding=(model_sharding, data_sharding),
        debug_log=cfg.setup.debug_run,
        max_batches=test_batches_per_epoch,
    )

    # Logging evaluation metrics
    eval_results = {
        "final_train_loss": float(train_loss),
        "final_valid_loss": float(valid_loss),
        "final_test_loss": float(test_loss),
        f"final_train_accuracy": float(best_model_train_accuracy),
        f"final_valid_accuracy": float(valid_accuracy),
        f"final_test_accuracy": float(test_accuracy),
    }
    print(json.dumps(eval_results, indent=4))
    with open(str(artifacts_dir / "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    if cfg.setup.wandb_logging:
        wandb.log({"eval_results": eval_results})

    ########## Cleanup ##########
    print("---------- Cleanup started: ----------")

    # save artifacts to wandb
    if cfg.setup.wandb_logging:
        files_to_upload = os.listdir(str(artifacts_dir))
        if not cfg.setup.wandb_upload_weights:
            files_to_upload.remove(best_model_file_name)
        for file_name in files_to_upload:
            file_path = str(artifacts_dir / file_name)
            wandb.save(file_path, base_path=str(experiment_base_dir), policy="now")
        wandb.finish()

    ########## FIN ##########
    print("---------- Experiment finished! ----------")


if __name__ == "__main__":
    main()
