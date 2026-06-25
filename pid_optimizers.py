"""
PID-style optimizers as torch.optim.Optimizer subclasses.

The model (ResNet-18) and autograd run on the GPU via PyTorch; only the
parameter-update math below is hand-written, so every P / I / D term stays
inspectable and editable. Both optimizers reduce to their baseline by
construction:

    SGD_PID  -> plain SGD          when Kp=1, Ki=0, Kd=0
    Adam_PID -> Adam              when Kp=0, Ki=1, Kd=0

References:
    An et al.,  "A PID Controller Approach for Stochastic Optimization of
                Deep Networks", CVPR 2018.
    Dai et al., "PID controller-based adaptive gradient optimizer for deep
                neural networks", IET Control Theory & Appl., 2023.
"""

import torch
from torch.optim.optimizer import Optimizer


class SGD_PID(Optimizer):
    r"""
    PID controller over SGD, following An et al. (CVPR 2018).

    The gradient g_t is treated as the control error e(t). The update combines:

        P : present gradient     Kp * g_t
        I : accumulated history  Ki * I_buf,   I_buf = m*I_buf + g_t
        D : change of gradient   Kd * D_buf,   D_buf = m*D_buf + (1-m)*(g_t - g_{t-1})

        theta_{t+1} = theta_t - lr * (Kp*g_t + Ki*I_buf + Kd*D_buf)

    Effective-learning-rate caveat: with momentum m and Ki>0 the integral
    buffer saturates near 1/(1-m), so the *effective* step on the I term is
    ~ lr*Ki/(1-m). A large Ki here behaves like a much larger learning rate,
    which is the usual reason SGD_PID destabilises relative to plain SGD.
    Tune Ki (or lower lr) accordingly.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 Kp=1.0, Ki=1.0, Kd=0.3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        Kp=Kp, Ki=Ki, Kd=Kd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['momentum']
            wd = group['weight_decay']
            Kp, Ki, Kd = group['Kp'], group['Ki'], group['Kd']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)            # L2 weight decay (new tensor)

                state = self.state[p]
                if len(state) == 0:
                    state['I_buf'] = torch.zeros_like(p)
                    state['D_buf'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                I_buf = state['I_buf']
                D_buf = state['D_buf']
                prev_grad = state['prev_grad']

                # I term: accumulate the history of gradients
                I_buf.mul_(m).add_(g)
                # D term: smoothed change of gradient
                D_buf.mul_(m).add_(g - prev_grad, alpha=(1.0 - m))
                # store current grad for the next derivative
                prev_grad.copy_(g)

                # PID combination (g.mul(Kp) is a fresh tensor; grad is untouched)
                update = g.mul(Kp).add_(I_buf, alpha=Ki).add_(D_buf, alpha=Kd)
                p.add_(update, alpha=-lr)

        return loss


class Adam_PID(Optimizer):
    r"""
    Adaptive-PID over Adam, following Dai et al. (IET CTA 2023, Eq. 7-8).

    Adam is re-expressed as an adaptive Integral controller; this adds explicit
    Proportional and Derivative terms:

        m_t   = b1*m + (1-b1)*g          v_t  = b2*v + (1-b2)*g^2
        mhat  = m_t/(1-b1^t)             vhat = v_t/(1-b2^t)
        denom = sqrt(vhat) + eps
        D_t   = g_t - g_{t-1}
        Dhat  = lr/(1-gamma^t) * D_t     # own bias correction, gamma=0.9

        theta_{t+1} = theta_t
                      - (lr/denom)*Kp*g_t      # P  (adaptive scaling)
                      - (lr/denom)*Ki*mhat     # I  (= Adam when Kp=0,Ki=1,Kd=0)
                      - Kd*Dhat                # D  (OUTSIDE the 1/sqrt(vhat) scaling)

    The D term lives outside the adaptive normalisation, matching the faithful
    Dai et al. formulation. Reduces exactly to Adam at Kp=0, Ki=1, Kd=0.

    Note on step 1: prev_grad starts at 0, so D_1 = g_1 and the bias factor
    lr/(1-gamma^1) = 10*lr give a sizeable derivative kick on the first step.
    This is faithful to Eq. 7 as written. Set d_warmup=True to zero the D term
    on step 1 if you want to suppress that transient when comparing curves.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, Kp=0.5, Ki=1.0, Kd=0.3, gamma=0.9,
                 d_warmup=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        Kp=Kp, Ki=Ki, Kd=Kd, gamma=gamma, d_warmup=d_warmup)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            Kp, Ki, Kd = group['Kp'], group['Ki'], group['Kd']
            gamma = group['gamma']
            d_warmup = group['d_warmup']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)            # L2 (Dai et al. used none; pass 0 to match)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                m, v, prev_grad = state['m'], state['v'], state['prev_grad']

                # Adam moments
                m.mul_(b1).add_(g, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(g, g, value=(1.0 - b2))
                mhat = m / (1.0 - b1 ** t)
                denom = (v / (1.0 - b2 ** t)).sqrt_().add_(eps)

                # Derivative (outside the adaptive scaling), own bias correction
                if t == 1 and d_warmup:
                    D = torch.zeros_like(g)
                else:
                    D = g - prev_grad
                Dhat_scale = lr / (1.0 - gamma ** t)
                prev_grad.copy_(g)

                # P + I share the adaptive (lr/denom) scaling
                pi = g.mul(Kp).add_(mhat, alpha=Ki)      # Kp*g + Ki*mhat
                p.addcdiv_(pi, denom, value=-lr)         # theta -= lr * pi / denom
                # D term, separate
                p.add_(D, alpha=-Kd * Dhat_scale)        # theta -= Kd * Dhat

        return loss


class SGD_PID_Classic(Optimizer):
    r"""
    Original An et al. (2018) form of PID-over-SGD, ported verbatim from the
    NumPy core.py used in Phase 1:

        V_t = alpha*V_{t-1} - lr*g_t                       # momentum buffer (P+I fused)
        D_t = alpha*D_{t-1} + (1-alpha)*(g_t - g_{t-1})    # smoothed derivative
        theta_{t+1} = theta_t + V_t + Kd*D_t

    Key difference from SGD_PID (the separated three-gain form): here lr lives
    INSIDE the momentum buffer V, and the derivative term is NOT scaled by lr.
    This is the faithful tensorboy/PIDOptimizer parameterization. The two
    classes are reparameterizations of the same update family, so the SAME
    numeric Kd means a different magnitude in each -- do not compare gains
    across forms directly; compare each form at its own best.
    """

    def __init__(self, params, lr=0.1, alpha=0.9, Kd=0.5, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, alpha=alpha, Kd=Kd, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            Kd = group['Kd']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['V'] = torch.zeros_like(p)
                    state['D'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                V, D, prev_grad = state['V'], state['D'], state['prev_grad']
                V.mul_(alpha).add_(g, alpha=-lr)                    # V = alpha*V - lr*g
                D.mul_(alpha).add_(g - prev_grad, alpha=(1.0 - alpha))
                prev_grad.copy_(g)
                p.add_(V).add_(D, alpha=Kd)                        # theta += V + Kd*D

        return loss
