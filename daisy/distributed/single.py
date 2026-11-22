from contextlib import nullcontext

from daisy.distributed.backend import DistributedBackend


class SingleNodeBackend(DistributedBackend):
    def __init__(self, args):
        super().__init__(args)
        self.rank = 0

    def transform_model(self, model):
        return model

    def get_context_for_microstep_forward(self, *args, **kwargs):
        return nullcontext()

    def get_adjusted_args_for_process(self, args):
        return args

    def is_master_process(self) -> bool:
        return True

    def get_raw_model(self, model):
        return model

    def get_world_size(self):
        return 1

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [parameter_name]
