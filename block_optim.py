import torch
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
import time
import math
import warnings
import gc
import re
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.integrations import is_deepspeed_zero3_enabled
import logging

if is_deepspeed_zero3_enabled():
    from deepspeed.runtime.zero.utils import apply_to_tensors_only
    from deepspeed.utils import z3_leaf_parameter
    from deepspeed.utils import logger

logger.setLevel(logging.WARNING) # surpress the tedious info log from deepspeed when switching trainable blocks

def print_rank_0(s, force=True):
    if not torch.distributed.is_initialized():
        print(s)
    elif torch.distributed.get_rank() == 0 and force:
        print(s)

def merge_block(prefix_list, merge_size, mode="random"):
    """merge the blocks of parameters into a larger block with a specified size."""
    import more_itertools
    assert merge_size <= len(prefix_list)
    indices = torch.randperm(len(prefix_list)) if mode == "random" else torch.arange(len(prefix_list))
    merged_block_list = [[prefix_list[i][0] for i in chunk] for chunk in more_itertools.chunked(indices, merge_size)]

    return merged_block_list

class BlockOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on a specified block list."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        block_prefix_list: List[str] = None,
        switch_block_every: int = 50,
        start_block: Optional[int] = None,
        switch_mode: str = "descending",
        active_modules: List[str] = [],
        include_embedding=False,
        include_lm_head=False,
        verbose: Optional[int] = 1,
        lora_mode = "all",
        n_layer_per_block = 1,
        lr_warmup_step_afterswitch = None,
        sgd_mode = "disabled",
        sgd_lr_scaling = 1.,
        sgd_use_sign = False
    ):
        """
        Args:
            base_optimizer: The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list: The list of blocks of parameters to be updated.
            switch_block_every: The number of optimization steps before switching to the next block.
            start_block (int): The index of the block to start with.
            switch_mode: Options: ["ascending", "descending", "random"] The block update order.
            active_modules (List[str]): The list of modules that are always active during optimization.
            verbose: The verbosity level for printing information during optimization.
            lora_mode: Options: ["all", "partial", "adapter_only"]. Invalid when there is no LoRA module.
                "all" means all the LoRA modules will be trained, while "partial" means only the LoRA modules after the active block will be trained.
                "adapter_only" means only the LoRA modules will be trained.
            lr_warmup_step_afterswitch: The number of steps for the learning rate warmup after switching to a new block.
            sgd_mode: Options: ["disabled", "all", "partial"]. When grad is ready, perform sgd update and drop the grad. This will not incur much memory overhead but will induce additional computation overhead.
                "all" means for each backward pass, all the parameters will perform one sgd step.
                "partial" means only the params of the later layers of the active block will perform sgd.
                "disabled" means sgd update is disabled.
            sgd_lr_scaling: The scaling factor for the learning rate during SGD update.
            sgd_use_sign: Whether to use sign SGD update.
        """
        self.switch_mode = switch_mode
        
        if block_prefix_list is None:
            block_prefix_list = self.infer_param_groups([n for n, _ in named_parameters_list], include_embedding, include_lm_head, n_layer_per_block)

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)
        
        self.verbose = verbose            
        self.switch_block_every = switch_block_every
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.block_num = len(block_prefix_list)
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults.pop("weight_decay")
        self.ds_zero3_enabled = is_deepspeed_zero3_enabled()
        self.lr_warmup_step_afterswitch = lr_warmup_step_afterswitch
        self.lora_mode = lora_mode
        self.sgd_mode = sgd_mode

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer

        if start_block is not None:
            self.current_block_idx = start_block
        elif self.switch_mode == "descending":
            self.current_block_idx = self.block_num - 1
        elif self.switch_mode == "ascending":
            self.current_block_idx = 0
        
        if self.switch_mode == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
            print_rank_0("next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()

        if any("lora" in n for n, _ in named_parameters_list):
            print_rank_0("LoRA mode detected. Both Block parameters and LoRA modules will be trained. LoRA modules' training mode: ", lora_mode)
            
        fp32_params = []
        for n, p in named_parameters_list:
            if p.dtype == torch.float32:
                fp32_params.append(n)
        if len(fp32_params) > 0:
            warnings.warn(f"We expect model to be loaded in fp16/bf16 precision, while detect fp32"
                f"weight for the following parameters: {fp32_params} \n"
                "This will cause additional memory usage and lose the benefit of mixed precision training.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)

        if self.sgd_mode != "disabled":
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.sgd_hook(n, lr_scaling=sgd_lr_scaling, use_sign=sgd_use_sign))
            if sgd_use_sign:
                print_rank_0("SignSGD is enabled")

        self.switch_trainable_params()
    
    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p
    
    def infer_param_groups(self, param_names, include_embedding, include_lm_head, n_layer_per_block):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        
        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)
        
        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)
        
        if n_layer_per_block > 1:
            block_prefix_list = merge_block(block_prefix_list, n_layer_per_block, self.switch_mode)
        
        return block_prefix_list

    def sgd_hook(self, n, lr_scaling=1., use_sign=False):
        """hook for performing sgd update on the fly"""
        # TODO: deal with the case where different parameters have different lr

        def sgd_update(p):
            # no sgd for the active block params
            if any(p_name in n for p_name in self.active_param_prefixs):
                return
            
            if use_sign:
                p.data.add_(p.grad.data.sign(), alpha=-self.param_groups[0]["lr"] * lr_scaling)
            else:
                p.data.add_(p.grad.data, alpha=-self.param_groups[0]["lr"] * lr_scaling)
            p.grad = None
        
        def sgd_update_zero3(p):
            # no sgd for the active block params
            if any(p_name in n for p_name in self.active_param_prefixs):
                return

            # based on the implementation https://github.com/OpenLMLab/LOMO/blob/main/lomo_optim/lomo.py
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
            param_fp32 = p.ds_tensor.to(torch.float32)
            grad_fp32 = p.grad.to(torch.float32)
            one_dim_grad_fp32 = grad_fp32.view(-1)

            partition_size = p.ds_tensor.numel()
            start = partition_size * torch.distributed.get_rank()
            end = min(start + partition_size, grad_fp32.numel())
            partitioned_grad_fp32 = one_dim_grad_fp32.narrow(0, start, end - start)

            partitioned_p = param_fp32.narrow(0, 0, end - start)
            if use_sign:
                partitioned_p.add_(partitioned_grad_fp32.sign(), alpha=-self.param_groups[0]["lr"] * lr_scaling)
            else:
                partitioned_p.add_(partitioned_grad_fp32, alpha=-self.param_groups[0]["lr"] * lr_scaling)
            p.ds_tensor[: end - start] = partitioned_p

        return sgd_update_zero3 if self.ds_zero3_enabled else sgd_update


    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        """Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer"""

        # warmup correction of the scheduled lr
        cur_block_step = (self.global_step + 1) % self.switch_block_every
        if (self.lr_warmup_step_afterswitch is None) or (self.global_step < self.lr_warmup_step_afterswitch):
            correction_ratio = 1.
        else:
            correction_ratio = min(1., (cur_block_step + 1) / self.lr_warmup_step_afterswitch)
        for group in self.base_optimizer.param_groups:
            scheduled_lr = self.param_groups[0]["lr"]
            group["lr"] = scheduled_lr * correction_ratio

    def step(self, *args, **kwargs) -> None:
        if self.ds_zero3_enabled:
            self.step_ds_zero3(*args, **kwargs)
        else:
            self.step_single_gpu(*args, **kwargs)

        torch.cuda.empty_cache()

        if (self.global_step + 1) % self.switch_block_every == 0:
            self.switch_trainable_params()

    def step_single_gpu(self, *args, **kwargs) -> None:
        self.record_mark = True

        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()

        self.global_step += 1

    def step_ds_zero3(self, *args, **kwargs) -> None:
        """
        Basic flow: 
        1. DeepSpeedZeroOptimizer_Stage3._optimizer_step()
          * convert wrapped optim (the BlockOptimizer)'s param into hp
          * call wrapped optim's step(), i.e. this function
        2. DeepSpeedZeroOptimizer_Stage3._reassign_or_swap_out_partitioned_parameters()
          * copy hp param to lp
          * repartition the params across different GPUs
          
        In other words, deepspeed has handled the mixed-precision training, so only ordinary step is needed
        """

        self.record_mark = True

        # Since ds ZeRO-3 update the parameter in group-wise manner, 
        # we need to update the referenced of base optimizer before every step
        for i in range(len(self.param_groups)):
            self.base_optimizer.param_groups[i]["params"] = self.param_groups[i]["params"]

        self._update_lr()
        self.base_optimizer.step(*args, **kwargs)

        # ds ZeRO-3 will call step function once for each partitioned group
        self.global_step += 1/len(self.param_groups)

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def _reset_ds_optimizer(self, trainable_param_groups):
        ds_optimizer = self.ds_optimizer
        
        # reset the bookkeeping of ds optimizer
        ds_optimizer.fp16_groups = []
        ds_optimizer.fp16_partitioned_groups = []
        ds_optimizer.fp16_partitioned_groups_flat = []
        ds_optimizer.fp16_partitioned_groups_flat_numel = []
        ds_optimizer.fp16_partitioned_groups_flat_id = []
        ds_optimizer.groups_padding = []
        ds_optimizer.fp32_partitioned_groups_flat = []
        
        # setup the fp16 groups and partition it
        ds_optimizer._create_fp16_partitions_with_defragmentation(trainable_param_groups)
        
        # register necessary hooks for synchronizing gradients
        self._create_reduce_and_remove_grad_hooks(trainable_param_groups)

        # create fp32 flat partition, initialize ipg buffer and grad partition buffer
        ds_optimizer._setup_for_real_optimizer()
        
        # invalidate the trace cache, since the module processing order has been changed
        ds_optimizer.parameter_offload.get_param_coordinator(training=True)._invalidate_trace()
        
        torch.cuda.empty_cache()

    def _create_reduce_and_remove_grad_hooks(self, trainable_param_groups):
        assert hasattr(self, "ds_optimizer"), "The optimizer doesn't have reference to its parent deepspeed optimizer yet. Set optimizer.ds_optimizer = optimizer after deepspeed.initiallize(..., optimizer=optimizer, ...)."
        ds_optimizer = self.ds_optimizer
        
        ds_optimizer.grad_accs = []
        ds_optimizer.leaf_parameters = defaultdict(list)
        for i, param_group in enumerate(ds_optimizer.fp16_groups):
            for param in param_group:
                if param.requires_grad:

                    # The hook must be created in un-partitioned parameter
                    param.all_gather()

                    def wrapper(param):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        # @instrument_w_nvtx
                        def reduce_partition_and_remove_grads(*notneeded):
                            ds_optimizer.reduce_ready_partitions_and_remove_grads(param)

                        ds_optimizer._grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
                        ds_optimizer.grad_accs.append(grad_acc)

                    #print(f"param grad fn {param.expand_as(param).grad_fn}")
                    if z3_leaf_parameter(param):
                        ds_optimizer.leaf_parameters[param.ds_z3_leaf_module].append(param)
                    else:
                        wrapper(param)

                    # Partition the parameter after creating the hook
                    param.partition()

        # We delay reduce-scatter for all gradients in the leaf modules until the backward pass of the leaf module is done
        for leaf_module, leaf_parameters in ds_optimizer.leaf_parameters.items():

            def wrapper_pre_hook(params):

                def forward_pre_hook(module, input):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_module_inputs_remaining = 0

                    # @instrument_w_nvtx
                    def reduce_leaf_module_grads(grad):
                        module._leaf_module_inputs_remaining -= 1
                        # Make sure everything is done in the leaf module
                        if module._leaf_module_inputs_remaining == 0:
                            for param in params:
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                ds_optimizer.reduce_ready_partitions_and_remove_grads(param)

                    def set_module_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_module_inputs_remaining += 1
                            tensor.register_hook(reduce_leaf_module_grads)
                        return tensor

                    output = apply_to_tensors_only(set_module_bwd_hook, input)

                    return output

                return forward_pre_hook

            def wrapper_post_hook():

                def forward_post_hook(module, input, output):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_output_required_grad_num = 0

                    def increment_rg_count_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_output_required_grad_num += 1
                        return tensor

                    apply_to_tensors_only(increment_rg_count_bwd_hook, output)

                    if module._leaf_module_inputs_remaining == 0 and module._leaf_output_required_grad_num > 0:
                        raise RuntimeError(
                            "A module cannot be set as a leaf module when it does not have any input tensors that require gradients and has output tensors that require gradients. This is because the gradient reduction hook will not be called in this case."
                        )

                return forward_post_hook

            ds_optimizer._leaf_module_hooks.append(leaf_module.register_forward_pre_hook(wrapper_pre_hook(leaf_parameters)))
            ds_optimizer._leaf_module_hooks.append(leaf_module.register_forward_hook(wrapper_post_hook()))


    def switch_trainable_params(self) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.

        Args:
            verbose (Optional[int], optional): The verbosity level for printing information. Defaults to None.
        """

        self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
        
        if self.verbose >= 1:
            print_rank_0(f"Parameters with the following prefix will be trainable: {self.active_param_prefixs}")
        
        param_names = [n for n, _ in self.named_parameters_list]
        if self.lora_mode == "all":
            self.active_param_prefixs += [n for n in param_names if "lora" in n]
        elif self.lora_mode == "partial":
            # we assume that the named_parameters_list (roughly) represents the reverse order of backpropogation
            trainable_param_names = [n for n in param_names if any(pref in n for pref in self.active_param_prefixs)]
            least_idx = min(param_names.index(n) for n in trainable_param_names)
            self.active_param_prefixs += [n for n in param_names[least_idx:] if "lora" in n]
        elif self.lora_mode == "adapter_only":
            self.active_param_prefixs = [n for n in param_names if "lora" in n and any(pref in n for pref in self.active_param_prefixs)]

        if self.ds_zero3_enabled:
            self._switch_trainable_params_zero3()
            
        else:
            self._switch_trainable_params_single_gpu()
        
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})

        self._update_active_block_idx()
        gc.collect()

    def _cal_adam_lr_scale_mean(self):
        """calculate avg_mean / avgsq_mean"""
        opt_state = self.base_optimizer.state
        active_params = list(opt_state.keys())
        opt_step = opt_state[active_params[0]]["step"]
        beta_1, beta_2 = self.base_optimizer.defaults['betas']

        avg_mean = torch.cat([opt_state[p]["exp_avg"].abs().flatten() for p in active_params]).mean() / (1 - beta_1 ** opt_step)
        avgsq_mean = torch.cat([torch.sqrt(opt_state[p]["exp_avg_sq"]).flatten() for p in active_params]).mean() / torch.sqrt(1 - beta_2 ** opt_step)
        scale = avg_mean / avgsq_mean

        return scale

    def _switch_trainable_params_zero3(self) -> None:
        assert not hasattr(self, "param_idx2lp") and not hasattr(self, "param_idx2hp")        
        
        # filter the trainable params
        trainable_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]
        
        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                param.requires_grad_(True)
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    trainable_param_groups[0]['params'].append(param)
                else:
                    trainable_param_groups[1]['params'].append(param)
                    
                if self.verbose >= 2:
                    print_rank_0(name)

        # remove the empty param groups
        trainable_param_groups[:] = [pg for pg in trainable_param_groups if len(pg["params"]) != 0]

        self.param_groups = trainable_param_groups
        self.base_optimizer.param_groups = trainable_param_groups
        
        # During the initialization, the ds_optimizer is not set yet
        if hasattr(self, "ds_optimizer"):
    
            # remove the grad sync hooks for the previous block
            for hook in self.ds_optimizer._grad_acc_hooks:
                hook.remove()
            for hook in self.ds_optimizer._leaf_module_hooks:
                hook.remove()
            self.ds_optimizer._grad_acc_hooks.clear()
            self.ds_optimizer._leaf_module_hooks.clear()
            
            # reset the bookkeeping of ds optimizer
            self._reset_ds_optimizer(trainable_param_groups)
            
    def _switch_trainable_params_single_gpu(self) -> None:
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]

        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                param.requires_grad_(True)
                param_hp = param.clone().float().detach().to(param.device)
                param_hp.requires_grad = True
                
                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)
                
                if self.verbose >= 2:
                    print_rank_0(name)
        
        self._update_traianble_params_for_sgd_hook()
            
        self.base_optimizer.param_groups = active_param_groups

    def _update_traianble_params_for_sgd_hook(self):
        param_names = [n for n, _ in self.named_parameters_list]

        if self.sgd_mode == "disabled":
            return
        elif self.sgd_mode == "all":
            pname_for_sgd = param_names
        elif self.sgd_mode == "partial":
            trainable_param_names = [n for n in param_names if any(pref in n for pref in self.active_param_prefixs)]
            least_idx = min(param_names.index(n) for n in trainable_param_names)
            pname_for_sgd = param_names[least_idx:]
        
        if self.verbose == 2:
            print_rank_0(f"SGD hook enabled. The following parameters will perform sgd update: {pname_for_sgd}")

        for n, p in self.named_parameters_list:
            if n in pname_for_sgd:
                p.requires_grad_(True)

    def _update_active_block_idx(self):
        # Update the trainable block
        if self.switch_mode == "random":
            # self.current_block_idx = random.randint(0, self.block_num - 1)
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                print_rank_0("Next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % self.block_num
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % self.block_num
        elif self.switch_mode == "fixed":
            pass
