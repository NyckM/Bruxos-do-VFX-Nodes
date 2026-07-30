import torch
from peft import LoraConfig, inject_adapter_in_model


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        trainable_modules = filter(
            lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        return trainable_param_names

    def add_lora_to_model(
        self, model, target_modules, lora_rank, lora_alpha=None, adapter_name="default"
    ):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules
        )
        print(f"Lora rank {lora_config.lora_alpha}")
        model = inject_adapter_in_model(
            lora_config, model, adapter_name=adapter_name)
        return model
