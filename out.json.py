import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(input_ptr, other_ptr, output_ptr, n_elements, rounding_mode, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform division
    result = input_vals / other_vals
    
    # Apply rounding if specified
    if rounding_mode == 0:  # None
        pass
    elif rounding_mode == 1:  # floor
        result = tl.floor(result)
    elif rounding_mode == 2:  # trunc
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 3:  # round
        result = tl.round(result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Map rounding_mode to integer codes
    mode_code = 0
    if rounding_mode == 'floor':
        mode_code = 1
    elif rounding_mode == 'trunc':
        mode_code = 2
    elif rounding_mode == 'round':
        mode_code = 3
    
    if out is None:
        out = torch.empty_like(input)
    
    _kernel[grid](
        input.contiguous().data_ptr(),
        other.contiguous().data_ptr(),
        out.data_ptr(),
        n_elements,
        mode_code,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
