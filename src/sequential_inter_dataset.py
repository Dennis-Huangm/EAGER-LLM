import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class SequentialInterDataset(Dataset):
    """
    Dataset for sequential recommendation from .inter format files
    Format: user_id item_id_1 item_id_2 ... item_id_n
    """
    
    def __init__(self, data_file, mode="test", max_his_len=20, 
                 his_sep=",", sample_num=-1, index_file=None):
        super().__init__()
        
        self.data_file = data_file
        self.mode = mode
        self.max_his_len = max_his_len
        self.his_sep = his_sep
        self.sample_num = sample_num
        self.index_file = index_file
        
        # Load index mapping if provided
        self.item_to_sid = None
        if self.index_file:
            self._load_index_mapping()
        
        # Load data
        self._load_data()
        
        # Process data for test mode
        if self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError("Only test mode is supported")
    
    def _load_index_mapping(self):
        """Load item_id to sid mapping from index.json file"""
        print(f"Loading index mapping from {self.index_file}...")
        with open(self.index_file, 'r') as f:
            index_data = json.load(f)
        
        # Convert to sid format: join the four tokens
        self.item_to_sid = {}
        for item_id, tokens in index_data.items():
            # Join the four tokens: <|a_xxx|><|b_xxx|><|c_xxx|><|d_xxx|>
            sid = ''.join(tokens)
            self.item_to_sid[item_id] = sid
        
        print(f"Loaded {len(self.item_to_sid)} item mappings")
    
    def _load_data(self):
        """Load interaction data from .inter file"""
        self.user_inters = []
        
        print(f"Loading data from {self.data_file}...")
        
        # First pass: count total lines (only if we need to sample)
        total_lines = None
        if self.sample_num > 0:
            print("Counting total lines for sampling...")
            with open(self.data_file, 'r') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            print(f"Total users: {total_lines}")
            
            # Calculate sampling ratio
            sample_ratio = min(1.0, self.sample_num / total_lines)
            print(f"Sampling ratio: {sample_ratio:.4f}")
        
        # Second pass: load data (with optional sampling)
        with open(self.data_file, 'r') as f:
            # Skip header
            header = f.readline()
            
            # Read data
            loaded_count = 0
            for line_idx, line in enumerate(f):
                # Early sampling: skip lines based on ratio
                if self.sample_num > 0 and sample_ratio < 1.0:
                    if np.random.random() > sample_ratio:
                        continue
                    
                    # Stop if we have enough samples
                    if len(self.user_inters) >= self.sample_num:
                        break
                
                if line_idx % 1000000 == 0:
                    print(f"Processed {line_idx} lines, loaded {len(self.user_inters)} users...")
                
                parts = line.strip().split()
                if len(parts) < 3:  # At least user_id + 2 items
                    continue
                
                user_id = parts[0]
                item_ids = parts[1:]
                
                # Convert item IDs to strings
                item_ids = [str(item_id) for item_id in item_ids]
                
                self.user_inters.append({
                    'user_id': user_id,
                    'items': item_ids
                })
        
        print(f"Loaded {len(self.user_inters)} users")
    
    def _process_test_data(self):
        """Process data for testing - predict the last item"""
        inter_data = []
        
        for user_inter in self.user_inters:
            items = user_inter['items']
            
            if len(items) < 2:
                continue
            
            one_data = dict()
            
            # Convert target item to sid if mapping exists
            target_item = items[-1]
            if self.item_to_sid:
                target_item = self.item_to_sid.get(target_item, target_item)
            one_data["item"] = target_item
            
            # History items (all except last)
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            
            # Convert history items to sid format if mapping exists
            if self.item_to_sid:
                history = [self.item_to_sid.get(item_id, item_id) for item_id in history]
            
            # Join history with separator and add trailing comma to prompt next item generation
            one_data["inters"] = self.his_sep.join(history) + self.his_sep
            inter_data.append(one_data)
        
        # Sampling is already done in _load_data, no need to sample again
        print(f"Processed {len(inter_data)} test instances")
        
        return inter_data
    
    def __len__(self):
        return len(self.inter_data)
    
    def __getitem__(self, index):
        d = self.inter_data[index]
        
        # Input: history SID sequence
        input_text = d["inters"]
        
        # Output: target SID
        output_text = d["item"]
        
        return dict(input_ids=input_text, labels=output_text)
    
    def get_all_items(self):
        """Get all unique items in the dataset (in SID format if mapping exists)"""
        if hasattr(self, 'all_items_cache'):
            return self.all_items_cache
        
        all_items = set()
        for user_inter in self.user_inters:
            all_items.update(user_inter['items'])
        
        # Convert to SID format if mapping exists
        if self.item_to_sid:
            all_items = {self.item_to_sid.get(item_id, item_id) for item_id in all_items}
        
        self.all_items_cache = all_items
        return all_items
    
    def get_all_items_without_dup(self):
        """Get all unique items (already no duplicates in set)"""
        return self.get_all_items()
    
    def get_prefix_allowed_tokens_fn(self, tokenizer):
        """
        For this simple version without token indices, we return None
        to skip prefix constraint (allows all tokens during generation)
        """
        return None
