import os
import json
import pickle
import hashlib
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
    
    def _get_cache_path(self, suffix=""):
        """Generate cache file path based on data file and parameters"""
        cache_key = f"{self.data_file}_{self.sample_num}_{self.max_his_len}_{self.his_sep}_{self.index_file}{suffix}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_dir = os.path.dirname(self.data_file) or "."
        base_name = os.path.basename(self.data_file)
        return os.path.join(cache_dir, f".{base_name}.{cache_hash}.cache")
    
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
        """Load interaction data from .inter file with caching"""
        cache_path = self._get_cache_path("_raw")
        
        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                cache_mtime = os.path.getmtime(cache_path)
                data_mtime = os.path.getmtime(self.data_file)
                if cache_mtime > data_mtime:
                    print(f"Loading raw data from cache: {cache_path}")
                    with open(cache_path, 'rb') as f:
                        self.user_inters = pickle.load(f)
                    print(f"Loaded {len(self.user_inters)} users from cache")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}, falling back to raw file")
        
        self.user_inters = []
        print(f"Loading data from {self.data_file}...")
        
        # Use larger buffer for better I/O
        buffer_size = 64 * 1024 * 1024  # 64MB
        
        # First pass: count total lines (only if we need to sample)
        total_lines = None
        sample_ratio = 1.0
        if self.sample_num > 0:
            print("Counting total lines for sampling...")
            with open(self.data_file, 'r', buffering=buffer_size) as f:
                total_lines = sum(1 for _ in f) - 1
            print(f"Total users: {total_lines}")
            sample_ratio = min(1.0, self.sample_num / total_lines)
            print(f"Sampling ratio: {sample_ratio:.4f}")
        
        # Second pass: load data
        with open(self.data_file, 'r', buffering=buffer_size) as f:
            header = f.readline()
            
            for line_idx, line in enumerate(f):
                if self.sample_num > 0 and sample_ratio < 1.0:
                    if np.random.random() > sample_ratio:
                        continue
                    if len(self.user_inters) >= self.sample_num:
                        break
                
                if line_idx % 1000000 == 0:
                    print(f"Processed {line_idx} lines, loaded {len(self.user_inters)} users...")
                
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                self.user_inters.append({
                    'user_id': parts[0],
                    'items': parts[1:]
                })
        
        print(f"Loaded {len(self.user_inters)} users")
        
        # Save to cache
        try:
            print(f"Saving raw data cache to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.user_inters, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _process_test_data(self):
        """Process data for testing - predict the last item, with caching"""
        cache_path = self._get_cache_path("_processed")
        
        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                cache_mtime = os.path.getmtime(cache_path)
                data_mtime = os.path.getmtime(self.data_file)
                index_mtime = os.path.getmtime(self.index_file) if self.index_file else 0
                if cache_mtime > max(data_mtime, index_mtime):
                    print(f"Loading processed data from cache: {cache_path}")
                    with open(cache_path, 'rb') as f:
                        inter_data = pickle.load(f)
                    print(f"Loaded {len(inter_data)} test instances from cache")
                    return inter_data
            except Exception as e:
                print(f"Cache load failed: {e}, reprocessing...")
        
        print("Processing test data...")
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
        
        print(f"Processed {len(inter_data)} test instances")
        
        # Save to cache
        try:
            print(f"Saving processed data cache to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(inter_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Failed to save cache: {e}")
        
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
