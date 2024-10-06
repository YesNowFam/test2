from torch.utils.cpp_extension import load
import torch
extension = load(name="fused_lrelu", sources=["fused_lrelu.cpp", "fused_lrelu.cu"])
NULL = torch.empty([0])

def lrelu_cuda(x, b, y, activate, alpha, clamp):
    if activate or clamp >= 0 or b.numel() != 0:
        return extension.fused_lrelu(x, b, y, activate, alpha, clamp)
    return x

class LeakyReluForward(torch.autograd.Function):
    @staticmethod
    def forward(x, b, activate, alpha, clamp):
        x = x.contiguous()
        b = b.contiguous()
        return lrelu_cuda(x, b, NULL, activate, alpha, clamp)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, activate, alpha, clamp = inputs
        ctx.save_for_backward(output)
        ctx.activate = activate
        ctx.alpha = alpha
        ctx.clamp = clamp
    
    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        y = ctx.saved_tensors
        x_grad = grad
        b_grad = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.activate or ctx.clamp >= 0:
                x_grad = LeakyReluBackward.apply(grad, y, ctx.activate, ctx.alpha, ctx.clamp)

        if ctx.needs_input_grad[1]:
            b_grad = x_grad.sum([0, 2, 3])
        
        return x_grad, b_grad, None, None, None
    
class LeakyReluBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad, y, activate, alpha, clamp):
        return lrelu_cuda(grad, NULL, y, activate, alpha, clamp)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, y, activate, alpha, clamp = inputs
        ctx.save_for_backward(y)
        ctx.activate = activate
        ctx.alpha = alpha
        ctx.clamp = clamp

    @staticmethod
    def backward(ctx, grad2):
        grad2 = grad2.contiguous()
        y = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            x_grad2 = LeakyReluBackward.apply(grad2, y, ctx.activate, ctx.alpha, ctx.clamp)

        return x_grad2, None, None, None, None
    
def fused_lrelu(x, b, activate=False, alpha=0.2, clamp=-1):
    return LeakyReluForward.apply(x, b, activate, alpha, clamp)


# Unit test
if __name__ == "__main__":
    x = torch.randn(1,3,1,1).to(0).requires_grad_()
    b = torch.ones(3).to(0).requires_grad_()
    print(x)
    print(fused_lrelu(x, b, activate=True, clamp=2))

