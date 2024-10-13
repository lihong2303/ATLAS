
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate
from definitions import (MMCLArguments,
                         TrainingSingleDatasetInfo,
                         TrainingArguments,
)

def build_config():
    cli_conf = OmegaConf.from_cli()

    if 'config' not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuation yaml file."
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    cli_conf.pop('config')
    config:MMCLArguments = OmegaConf.merge(conf,cli_conf)
    # assert ("max_steps" in config.training.lightning), "lightning config must specify 'max_steps'"
    if not os.path.exists(os.path.join(config.training.lightning.default_root_dir,config.training.cur_expt_name)):
        os.makedirs(os.path.join(config.training.lightning.default_root_dir,config.training.cur_expt_name))
    OmegaConf.save(config,os.path.join(config.training.lightning.default_root_dir,config.training.cur_expt_name,"train_params.yaml"))
    return config


def build_datamodule_kwargs(dm_config:TrainingSingleDatasetInfo):
    kwargs = {
        "data_dir":dm_config.data_dir,
        "batch_size":dm_config.batch_size,
        "test_batch_size":dm_config.test_batch_size,
        "num_workers":dm_config.num_workers,
        "allow_uneven_batches":dm_config.allow_uneven_batches,
    }
    kwargs.update(dm_config.datamodule_extra_kwargs)
    return kwargs
