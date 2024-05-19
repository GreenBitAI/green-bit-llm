import torch

from .bnb_optimizer import Optimizer2State

try:
    from galore_torch.galore_projector import GaLoreProjector
except ModuleNotFoundError as e:
    raise Exception("Error: GaLoreProjector is not available. Make sure 'galore-torch' has been installed on you system.")

try:
    from bitorch_engine.layers.qlinear.nbit import MPQWeightParameter
    from bitorch_engine.utils.quant_operators import gptq_style_unpacking
    from bitorch_engine.layers.qlinear.nbit.cuda.utils import pack_fp_weight
except ModuleNotFoundError as e:
    raise Exception(f"Error occurred while importing Bitorch Engine module '{str(e)}'.")


class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=.0, amsgrad=False, optim_bits=8,
                 args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False,
                 dtype: torch.dtype = torch.float16):
        self.dtype = dtype
        super().__init__( "adam", params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping,
                          block_wise, is_paged=is_paged )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        #if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0

                if isinstance(p, MPQWeightParameter):
                    grad = p.privileged_grad.to(self.dtype).to(p.grad.device)
                else:
                    grad = p.grad.to(self.dtype)

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])

                    projector = state["projector"]
                    grad = projector.project(grad, state["step"])

                saved_data = None
                if "rank" in group or isinstance(p, MPQWeightParameter):
                    # suboptimal implementation
                    # In the implementation mentioned, the author sets the variable p (representing model parameters) to zero,
                    # meaning p does not change during the update step. Instead, only the gradient states are updated,
                    # and actual weight modifications are calculated manually later in the code.
                    saved_data = p.data.clone()
                    p.data = torch.zeros_like(grad)

                    if 'weight_decay' in group and group['weight_decay'] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group['weight_decay_saved'] = group['weight_decay']
                        group['weight_decay'] = 0

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex, grad)

                self.prefetch_state(p)

                self.update_step(group, p, gindex, pindex, grad)

                torch.cuda.synchronize()

                if 'weight_decay_saved' in group:
                    group['weight_decay'] = group['weight_decay_saved']
                    del group['weight_decay_saved']

                w_unpacked = None
                # GaLore Projection Back
                if "rank" in group:
                    # now the p.data is actually: -norm_grad*lr
                    norm_grad = projector.project_back(p.data)

                    if isinstance(p, MPQWeightParameter):
                        # unpack qweight
                        p.data = saved_data
                        w_unpacked = gptq_style_unpacking(p).to(self.dtype).to(saved_data.device)
                        w_unpacked.add_(norm_grad)
                        if group["weight_decay"] > 0.0:
                            w_unpacked.add_(w_unpacked, alpha=-group['lr'] * group['weight_decay'])
                        # pack fp weight back to Q-weight and update qweight data
                        p.data = pack_fp_weight(w_unpacked, p)
                    else:
                        p.data = saved_data.add_(norm_grad)
                        if group["weight_decay"] > 0.0:
                            p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                elif isinstance(p, MPQWeightParameter):
                    # now the p.data is actually: -norm_grad*lr
                    norm_grad = p.data.clone()
                    # unpack qweight
                    p.data = saved_data
                    w_unpacked = gptq_style_unpacking(p).to(self.dtype).to(saved_data.device)
                    w_unpacked.add_(norm_grad)
                    if group["weight_decay"] > 0.0:
                        w_unpacked.add_(w_unpacked, alpha=-group['lr'] * group['weight_decay'])
                    # pack fp weight back to Q-weight and update qweight data
                    p.data = pack_fp_weight(w_unpacked, p)

                # pack fp weight back to qweight
                if w_unpacked is not None:
                    del w_unpacked
                if saved_data is not None:
                    del saved_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss
