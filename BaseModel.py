# coding: utf-8
# Author：wangzixiang
# Date ：2025/6/21 0:29
import warnings
warnings.filterwarnings("ignore")

from asteroid import torch_utils
import json, yaml
from dns_loader import DNSDataset, WavHopDataset
from torch.utils.data import DataLoader
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch, time, os

class MyBaseSystem():
    def __init__(self, conf_path):
        if not os.path.exists(conf_path):
            print("conf path error!")
        with open(conf_path, "r") as f:
            conf = yaml.safe_load(f)
        self.conf = conf
        self.save_file = conf["preset"]["save_file"]
        self.checkpoint_dir = os.path.join(conf["preset"]["save_file"], "checkpoints/")
        
        # 初始化数据加载器
        if self.conf["preset"]["dataset"] == 'D':
            if self.conf["preset"].get("enframe", False):
                self.train_loader, self.val_loader = get_dns_data_frame_loader(
                    conf["train"]["batch_size"], 
                    conf["train"]["num_workers"], 
                    json_home=conf["preset"]["json_home"],
                    frame_dur=conf["preset"]["frame_dur"], 
                    hop_dur=conf["preset"]["hop_dur"],
                    data_home=conf["preset"]["data_home"]
                )
            else:
                self.train_loader, self.val_loader = get_dns_data_loader(
                    conf["train"]["batch_size"], 
                    conf["train"]["num_workers"], 
                    json_home=conf["preset"]["json_home"],
                    data_home=conf["preset"]["data_home"]
                )
        else:
            raise ValueError("Unsupported dataset type in config")
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.checkpoint_cb = None
        self.early_stop_cb = None
        self.init_checkpoints()
        self.system = None
        self.trainer = None

    def init_checkpoints(self):
        os.makedirs(self.save_file, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_cb = ModelCheckpoint(
            dirpath=self.checkpoint_dir, 
            monitor="val_loss", 
            mode="min", 
            save_top_k=5, 
            verbose=True
        )

    def init_optims(self, model, criterion, optimizer=None):
        if optimizer is None:
            self.optimizer = make_optimizer(model.parameters(), **self.conf["optim"])
        else:
            self.optimizer = optimizer
            
        if self.conf["train"].get("half_lr", False):
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer, 
                factor=self.conf["scheduler"]["factor"],
                patience=self.conf["scheduler"]["patience"], 
                verbose=self.conf["scheduler"]["verbose"]
            )
        
        if self.conf["train"].get("early_stop", False):
            self.early_stop_cb = EarlyStopping(
                monitor="val_loss", 
                patience=self.conf["train"].get("early_stop_patience", 20),
                verbose=True
            )

        self.criterion = criterion

    def init_system_and_trainer(self, SystemClass, model, devices=None):
        """初始化训练系统和训练器
        
        参数:
            SystemClass: 系统类
            model: 要训练的模型
            devices: 使用的设备列表 (如 [0,1]) 或设备数量 (如 2)
        """
        # 准备回调列表
        callbacks = [self.checkpoint_cb]
        if hasattr(self, 'early_stop_cb') and self.early_stop_cb:
            callbacks.append(self.early_stop_cb)
        
        # 设置训练设备
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if devices is None:
            devices = "auto"  # 自动检测可用设备
            
        self.system = SystemClass(
            model=model,
            loss_func=self.criterion,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            scheduler=self.scheduler,
            config=self.conf
        )
        
        self.trainer = pl.Trainer(
            max_epochs=self.conf["train"]["epochs"],
            callbacks=callbacks,
            default_root_dir=self.save_file,
            devices=devices,  # 使用 devices 参数
            accelerator=accelerator,
            strategy="auto",  # 自动选择并行策略
            limit_train_batches=1.0,  # 替代 train_percent_check
            gradient_clip_val=5.0,
            enable_progress_bar=True,
            logger=True
        )

    def fit(self):
        self.trainer.fit(self.system)
        best_k = {k: v.item() for k, v in self.checkpoint_cb.best_k_models.items()}
        with open(os.path.join(self.save_file, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

def get_dns_data_loader(batch_size, num_workers, json_home, data_home):
    train_json_file = os.path.join(json_home, "train_file_info.json")
    val_json_file = os.path.join(json_home, "val_file_info.json")
    train_set = DNSDataset(train_json_file, data_home=data_home)
    val_set = DNSDataset(val_json_file, data_home=data_home)
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True  # 加速数据加载
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    return train_loader, val_loader

def get_dns_data_frame_loader(batch_size, num_workers, json_home, data_home, frame_dur, hop_dur):
    train_json_file = os.path.join(json_home, "train_file_info.json")
    val_json_file = os.path.join(json_home, "test_file_info.json")
    
    train_set = WavHopDataset(
        train_json_file, 
        frame_dur=frame_dur, 
        hop_dur=hop_dur, 
        data_home=data_home
    )
    val_set = WavHopDataset(
        val_json_file, 
        frame_dur=frame_dur, 
        hop_dur=hop_dur, 
        data_home=data_home
    )
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    return train_loader, val_loader

def load_best_param(log_file, model, device="cuda", test=True):
    if not os.path.exists(log_file):
        print("Log path error")
        return model
    
    best_models_file = os.path.join(log_file, "best_k_models.json")
    if not os.path.exists(best_models_file):
        print(f"Best models file not found at {best_models_file}")
        return model
    
    with open(best_models_file, "r") as f:
        best_k = json.load(f)
    
    if not best_k:
        print("No best models found in JSON")
        return model
    
    best_model_path = min(best_k, key=best_k.get)
    if not os.path.exists(best_model_path):
        # 尝试修复路径
        ckpt_name = os.path.basename(best_model_path)
        best_model_path = os.path.join(log_file, "checkpoints", ckpt_name)
        
        if not os.path.exists(best_model_path):
            print(f"Model checkpoint not found at {best_model_path}")
            return model
    
    try:
        map_location = torch.device(device) if device else "cpu"
        ckpt = torch.load(best_model_path, map_location=map_location)
        model = torch_utils.load_state_dict_in(ckpt["state_dict"], model)
        
        if test:
            model.eval()
        print(f"Successfully loaded parameters from {best_model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return model

def model_test_timer(model, input_size=(8, 16000 * 3), device="cuda", repetitions=10):
    test_inp = torch.randn(input_size)
    
    if "cuda" in device and torch.cuda.is_available():
        model = model.to(device)
        test_inp = test_inp.to(device)
    
    # 预热
    with torch.no_grad():
        _ = model(test_inp)
    
    start = time.time()
    for _ in range(repetitions):
        with torch.no_grad():
            output = model(test_inp)
    
    avg_time = (time.time() - start) / repetitions
    print(f"Output size: {output.size()}")
    print(f"Average inference time over {repetitions} runs: {avg_time:.4f} seconds")

def load_conf(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)