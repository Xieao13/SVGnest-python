from torch.utils.data import Dataset
import numpy as np
import torch
import json
import logging
import sys


logger = logging.getLogger(__name__)

class PlacementOnlyDataset(Dataset):
    """专用于放置顺序的数据集"""

    def __init__(self, jsonl_file):
        self.data = []
        self.load_jsonl_data(jsonl_file)

    def load_jsonl_data(self, jsonl_file):
        """加载JSONL格式的数据文件"""
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
        logger.info(f"从 {jsonl_file} 加载了 {len(self.data)} 个数据样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bin_data = torch.tensor(item['bin'], dtype=torch.float32)
        parts = torch.tensor(item['parts'], dtype=torch.float32)
        placement_order = torch.tensor(item['placement_order'], dtype=torch.long)
        valid_length = len(item['parts'])

        return {
            'bin': bin_data,
            'parts': parts,
            'placement_order': placement_order,
            'efficiency': item['efficiency'],
            'valid_length': valid_length
        }


def PlacementOnly_collate_fn(batch):
    """放置顺序的批次处理函数"""
    max_len = max(item['valid_length'] for item in batch)
    
    bins = []
    padded_parts = []
    padded_placement_orders = []
    efficiencies = []
    valid_lengths = []

    for item in batch:
        if len(item['placement_order']) != item['valid_length']:
            logger.debug(f"data with invalid: placement_order: {item['placement_order']}, valid_length: {item['valid_length']}")
            continue
        bins.append(item['bin'])
        
        parts = item['parts']
        current_len = item['valid_length']

        padding_len = max_len - current_len
        padded = torch.cat([parts, torch.zeros((padding_len, 2), dtype=torch.float32)], dim=0)
        padded_parts.append(padded)
        
        placement_order = item['placement_order']
        
        padded_order = torch.cat([placement_order, torch.full((padding_len,), -1, dtype=torch.long)])
        padded_placement_orders.append(padded_order)
        
        efficiencies.append(item['efficiency'])
        valid_lengths.append(current_len)

    return {
        'bins': torch.stack(bins),
        'parts': torch.stack(padded_parts),
        'placement_orders': torch.stack(padded_placement_orders),
        'efficiencies': torch.tensor(efficiencies, dtype=torch.float32),
        'valid_lengths': torch.tensor(valid_lengths, dtype=torch.long)
    }

if __name__ == "__main__":
    dataset = PlacementOnlyDataset('./data/placement-0529-ga-20epoch-norotation/train.jsonl')
    # print(len(dataset))
    # print(dataset[0])
    batch = PlacementOnly_collate_fn([dataset[0], dataset[1]])
    print(batch)
