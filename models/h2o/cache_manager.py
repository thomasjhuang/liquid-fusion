from collections import defaultdict
import threading
from models.h2o.h2o_llama import LlamaAttention_heavy_hitter
import logging

logger = logging.getLogger(__name__)

class KVCacheManager:
    def __init__(self):
        self._stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_tokens': 0
        })
        self.lock = threading.Lock()
    
    def record_cache_event(self, layer_id: str, event_type: str):
        with self.lock:
            self._stats[layer_id][event_type] += 1
            
    def clean_cache(self, model):
        """Clean cache for all attention layers"""
        for name, module in model.named_modules():
            if isinstance(module, (LlamaAttention_heavy_hitter)):
                module._clean_cache()
                
    def get_stats(self):
        with self.lock:
            return dict(self._stats)
