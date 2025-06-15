from datasets.replica import ReplicaDataset
from datasets.scannet import ScanNetDataset

def load_dataset(config):
    if config["Dataset"]["type"] == "replica":
        return ReplicaDataset(config)
    elif config["Dataset"]["type"] == "scannet":
        return ScanNetDataset(config)
    else:
        raise ValueError("Unknown dataset type")
