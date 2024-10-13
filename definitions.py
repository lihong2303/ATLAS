

from typing import List,Optional,Any,Dict,Tuple,Union

from dataclasses import dataclass,field
from omegaconf import MISSING


def _default_split_key_mapping():
    return {x: x for x in ["train","validation","test"]}

@dataclass
class DatasetInfo:
    key:str = MISSING

@dataclass
class HFDatasetInfo(DatasetInfo):
    key: str = MISSING
    subset:Optional[str] = None
    remove_columns:Optional[List[str]] = None
    rename_columns:Optional[List[Any]] = None
    split_key_mapping:Optional[Dict[str,str]] = field(default_factory=_default_split_key_mapping)
    extra_kwargs:Dict[str,Any] = field(default_factory=dict)

@dataclass
class TorchVisionDatasetInfo(DatasetInfo):
    key:str = MISSING
    train_split:str = "train"
    val_split:str = "val"
    has_val:bool = True
    test_split:str = "test"

@dataclass
class LowShotConfig:
    key: bool = False
    num_low_shot: int = 2048

@dataclass
class TrainingSingleDatasetInfo:
    data_dir:str
    low_shot_config: LowShotConfig
    train:List[DatasetInfo] = field(default_factory=lambda:[HFDatasetInfo()])
    val:Optional[List[DatasetInfo]] = None
    batch_size:Optional[int] = None
    test_batch_size:Optional[int] = None
    num_workers:Optional[int] = None
    allow_uneven_batches:bool = False
    datamodule_extra_kwargs:Dict[str,Any] = field(default_factory=dict)
@dataclass
class TrainingDatasetsInfo:
    selected:List[str] = field(default_factory=lambda:["image","text","vl"])
    image:Optional[TrainingSingleDatasetInfo] = None
    text:Optional[TrainingSingleDatasetInfo] = None
    vl:Optional[TrainingSingleDatasetInfo] = None
    num_classes:int = MISSING

@dataclass
class TrainingArguments:
    target_model:str = None
    cur_dataset: str = None
    cur_expt_name: str = None
    adapter_weighted_method: str = None
    initialize_from_checkpoint: str = None
    resume_from_checkpoint: str = None
    continual_sequence:str = None
    lightning:Dict[str,Any] = field(default=dict)
    lightning_checkpoint:Optional[Dict[str,Any]] = None
    lightning_load_from_checkpoint:Optional[str] = None
    seed:int = -1
    learning_rate:float = 0.0002
    lr_end:float = 1e-7
    adam_eps:float = 1e-8
    adam_weight_decay:float = 0.01
    adam_betas: Tuple[float,float] = field(default_factory=lambda:(0.9,0.999))
    warmup_ratio:float = 0.1
    

@dataclass
class Adapter_infos:
    key:bool = False
    adapter_embed_dim:int = 256
@dataclass
class Vision_adapter_infos(Adapter_infos):
    key: bool = False
    adapter_embed_dim: int = 256
    
@dataclass
class Text_adapter_infos(Adapter_infos):
    key: bool = False
    adapter_embed_dim:int = 256
    
@dataclass
class Adapter:
    adapter_infos:Optional[Adapter_infos] = None
    vision_adapter_infos: Optional[Vision_adapter_infos] = None
    text_adapter_infos:Optional[Text_adapter_infos] = None

    
@dataclass
class CL_setting:
    key:str = None
    ewc_fisher_sample_percentage: float = 0.01
    ewc_loss_weight: float = 100.0
    save_task_parameters:bool = False

@dataclass
class ModelArguments:
    key: str = "CLIP"
    pretrained:bool = False
    VILT_ckpt_dir:Optional[str] = None
    VILT_tokenizer:Optional[str] = None
    num_classes:int = 3
    classifier_in_dim:int = 1024
    update_method:str = "cls-head"
    cl_setting:Optional[CL_setting] = None
    adapter: Optional[Adapter] = None

@dataclass
class MMCLArguments:
    datasets:TrainingDatasetsInfo = TrainingDatasetsInfo()
    training:TrainingArguments = TrainingArguments()
    model:ModelArguments = ModelArguments()