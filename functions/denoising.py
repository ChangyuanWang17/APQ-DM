import torch
from torch import optim
import torch.nn.functional as F
from functions.losses import loss_registry
from utils.quant_util import QConv2d

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def cal_entropy(attn):
    return -1 * torch.sum((attn * torch.log(attn)), dim=-1).mean()

def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # print(len(seq)) # 100
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            # print(t, t.shape)  #800, 64
            # print(xt.device, t.device)
            et = model(xt, t)
            # print(et.device)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # .index_select(0, t)有问题,t.long()解决
    a = (1-b).cumprod(dim=0).index_select(0, t.long())
    a = a.view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    # output = model(x0, t.float())
    # print(output.requires_grad)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3)), output
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0), output
    
def generalized_steps_loss(x, seq, model, b, optimizer, t_mode, **kwargs):
    model.eval()
    # model.train()
    # if t_mode == "diff":
    #     seq = torch.tensor([kwargs["timestep_select"]])
    #     seq_next = torch.tensor([0])
    # else:
    #     seq_next = [-1] + list(seq[:-1])
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    count_1 = 0
    # print(len(seq)) # 100
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        # print(t.requires_grad)  # False
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda').detach()
        # xt = xs[0].to('cuda').detach()
        # xt.requires_grad = False
        # print(xt.requires_grad)
        # print(t, t.shape)  #800, 64
        # print(xt.device, t.device)
        # et = model(xt, t)
        e = torch.randn_like(xt)
        # print(xt[0][1][0][1])
        # total_loss, et = noise_estimation_loss(model, xt, t, e, b)
        for ii, jj in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * ii).to(x.device)
            total_loss, et = noise_estimation_loss(model, xt, t, e, b)
            # print(total_loss)
        # et = model(xt, t.float())
        dm_loss_t = 0
        for k, layer in enumerate(model.modules()):
            # print(type(layer))
            if type(layer) in [QConv2d]:
                # print(type(layer))
                # alpha = layer.sw
                # alpha = layer.alpha_activ[count_1]
                alpha = F.softmax(layer.alpha_activ, dim=1)
                # print(alpha[0].grad)
                _ ,group_n, dim = alpha.shape
                # dm_loss_t = 0
                # for k in range(group_n):
                #     dm_loss_t += cal_entropy(alpha[k])
                dm_loss_t += cal_entropy(alpha[count_1]) / (group_n * dim)
        # print(total_loss)
        total_loss = total_loss + 1.0 * dm_loss_t
        # print(alpha)
        # total_loss = dm_loss_t
        # print(total_loss.requires_grad)
        # print(et.device)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to('cpu'))
        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.to('cpu'))

        # print(f"loss: {total_loss}, {dm_loss_t}, {count_1}")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        count_1 += 1

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')
            # print(t)
            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
