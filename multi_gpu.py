import torch.nn as nn

class MultiGPUWrapper:
    """
    DataParallelでWrapしてもmoduleのアトリビュートにアクセスできるWrapper
    DataParallelへのアトリビュートアクセスが優先され，なければmoduleのものにアクセスする
    dirなど中身を見たいときにはthis.moduleもしくはthis.data_parallelを用いる
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        self.module = module
        self.data_parallel = nn.DataParallel(self.module, device_ids, output_device, dim)
        
    def __getattr__(self, attr):
        if hasattr(self.data_parallel, attr):
            return getattr(self.data_parallel, attr)
        else:
            return getattr(self.module, attr)
    
    def __call__(self, *args, **kwargs):
        return self.data_parallel(*args, **kwargs)