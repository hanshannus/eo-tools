from eotorch.dataloaders import init_torch_dataloader
from eotorch.tasks import TaskTrainer, TaskMetrics, TaskLoss
from pytorch_lightning import Trainer
from pathlib import Path
from yamx import Yaml
from loguru import logger
from typing import Union


def get_dataloader(config):
    # dataset
    dataset = config["images"] & config["labels"]
    # sampler
    sampler = config["sampler"]
    # dataloader
    return init_torch_dataloader(
        dataset=dataset,
        sampler=sampler,
        **config["params"],
    )


def get_task(config, model):
    logger.info("Get loss.")
    cfg = config["loss"]
    loss = TaskLoss(cfg["class"])
    print(repr(cfg["preprocessing"]))
    loss.preprocessing = cfg["preprocessing"]

    logger.info("Get metrics")
    cfg = config["metrics"]
    metrics = TaskMetrics(cfg["class"])
    print(repr(cfg["preprocessing"]))
    metrics.preprocessing = cfg["preprocessing"]

    logger.info("Get optimizer")
    cfg = config["optimizer"]
    optimizer = cfg["class"](model.parameters(), **cfg["params"])

    logger.info("Get learning rate scheduler")
    cfg = config["scheduler"]
    scheduler = cfg["class"](optimizer, **cfg["params"])

    optimizer_config = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": config["monitor"],
        },
    }

    logger.info("Initialize trainer task.")
    return TaskTrainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer_config=optimizer_config,
        **config.get("params", {}),
    )


def train(
    config_path: Union[str, Path],
) -> Trainer:
    logger.info("Load configuration file.")
    config = Yaml.load(config_path)
    logger.info("Get dataloaders.")
    train_dataloader = get_dataloader(config["train_dataloader"])
    val_dataloader = get_dataloader(config["val_dataloader"])
    logger.info("Get model.")
    model = config["model"]
    logger.info("Get task.")
    task = get_task(config["task"], model)
    logger.info("Initialize trainer.")
    trainer = Trainer(**config["trainer"])
    logger.info("Start training.")
    trainer.fit(
        model=task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    logger.info("Training finished.")
    return trainer
