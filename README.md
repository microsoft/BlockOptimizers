# BlockOptimizers
This repository implements
* BAdam algorithm from "[BAdam: A Memory Efficient Full Parameter Optimization Method for Large Language Models](https://arxiv.org/abs/2404.02827)"
* BREAD algorithmic framework from "[Accelerating Block Coordinate Descent for LLM Finetuning via Landscape Correction](https://openreview.net/forum?id=zs6bRl05g8)"

The implementation is built upon the code base https://github.com/Ledzy/BAdam. The code supports model-parallel training offered by Deepspeed ZeRO-3.

# Usage
Both implementations of BAdam and BREAD are based on on the class `BlockOptimizer`, which wraps the original optimizer (e.g. `torch.optim.AdamW`) to change its behavior, such as using block-wise update, injecting on-the-fly SGD hook, and dynamically adjusting trainable parameters. 

To start with, one shall copy the `block_optim.py` into local directory and add the following modification on the original code.

```python
from block_optim import BlockOptimizer

optimizer = BlockOptimizer(
    base_optimizer=original_optimizer, # can be any torch.Optimizer
    named_parameters_list=list(model.named_parameters()), 
    switch_block_every=100, # optimize each block sub-problem by 100 steps
    switch_mode="random", # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
)
```

The above code snippet implements BAdam. To further accelerate the convergence, BREAD correct those inactive blocks by memory-efficient update. We have implemented two classes of landscape correction.


### Low-rank Based Correction
```python
# NOTE: make sure that the model has been transformed into peft model with LoRA adapeters, as shown below
from peft import LoraConfig, TaskType, get_peft_model
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(original_model, peft_config)

optimizer = BlockOptimizer(
    ... # same arguments as BAdam
    lora_mode="all" # options: ["all", "partial"]
)
```

### On-the-fly SGD Based Correction
```python
optimizer = BlockOptimizer(
    ... # same arguments as BAdam
    sgd_mode="all", # options: ["all", "partial", "disabled"]
    sgd_lr_scaling=5., # the learning rate scaling of SGD update. SGD usually uses larger lr than Adam.
    sgd_use_sign=False # if True, use signSGD update. SignSGD update usually converges faster than SGD when the model is not well trained, e.g. under the scenario of training from scratch.
)
```

We remark that any memory efficient optimization algorithms/techniques can be incoporated into the BREAD framework to accelerate BAdam. We leave the integration of other optimizers as a future work.