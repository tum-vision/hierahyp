"""
Hyperbolic functions for semantic segmentation. Largely inspired from the 
Hyperbolic Image Segmentation (Atigh et al., 2022) author implementation:
https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation

Please consider citing this work:
@article{ghadimiatigh2022hyperbolic,
  title={Hyperbolic Image Segmentation},
  author={GhadimiAtigh, Mina and Schoep, Julian and Acar, Erman and van Noord, Nanne and Mettes, Pascal},
  journal={arXiv preprint arXiv:2203.05898},
  year={2022}
}
"""



import torch
import torch.nn
import torch.nn.functional
import torch.nn.functional as F

import time 

PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0


def torch_sqnorm(u, keepdim=True, dim=-1):
    # performs sq norm over last dim
    return torch.sum(u * u, dim=dim, keepdim=keepdim)


def cross_correlate_torch(inputs, filters):
    return F.conv2d(inputs, filters, stride=1, padding=(filters.shape[-1] - 1) // 2)


def hyp_mlr_torch(inputs, c, P_mlr, A_mlr): 
    xx = torch_sqnorm(inputs)     
    pp = torch_sqnorm(-P_mlr, keepdim=False, dim=1)  
    assert not torch.isnan(pp).any()
    # 1x1 conv.
    # | -p * x |^2 p has shape ncls, D, need in-out shape for filter: D,ncls
    P_kernel = torch.transpose(-P_mlr, -1, 0)[None, None, :, :]

    inputs = inputs.permute((0,3,1,2))
    P_kernel = P_kernel.permute((3,2,0,1))

    px = cross_correlate_torch(inputs, P_kernel)
    px = px.permute((0,2,3,1))
    inputs = inputs.permute((0,2,3,1))
    P_kernel = P_kernel.permute((2,3,1,0))

    # c^2 * | X|^2 * |-P|^2
    sqsq = torch.mul(c * xx.cuda(), c * pp[None, None, None, :].cuda())  # sh B,H,W,ch

    # Weight operations
    A_norm = torch.norm(A_mlr, dim=1)  # [ncls,1]
    normed_A = torch.nn.functional.normalize(A_mlr, p=2, dim=1) 

    normed_A_transposed = torch.transpose(normed_A, dim0=-2, dim1=-1) # Transpose the last two dimensions
    A_kernel= normed_A_transposed.unsqueeze(0).unsqueeze(0) # Add two singleton dimensions at the beginning 

    # rewrite mob add as alpha * p + beta * x
    # where alpha = A/D
    A = 1 + torch.add(2 * c * px.cuda(), c * xx.cuda())  # sh B,H,W,ch
    
    assert not torch.isnan(A).any()

    B = 1 - c * pp.cuda()  # sh ch ## beta = B/D
    assert not torch.isnan(B).any()

    D = 1 + torch.add(2 * c * px.cuda(), sqsq.cuda())  # sh B,H,W,ch
    D = torch.maximum(D, torch.tensor(EPS))
    assert not torch.isnan(D).any()

    # calculate mobadd norm indepently from mob add
    # if mob_add = alpha * p + beta * x, then
    #  |mob_add|^2 = theta**2 * |p|^2 + gamma^2 * |x|^2 + 2*theta*gamma*|px|
    # theta = A/D, gamma = B/D
    alpha = A / D  # B,H,W,ch
    assert not torch.isnan(alpha).any()
    beta = B[None, None, None, :].cuda() / D.cuda()  # B,H,W,ch
    assert not torch.isnan(beta).any()

    # calculate mobius addition norm independently
    mobaddnorm = (
            (alpha ** 2 * pp[None, None, None, :].cuda())
            + (beta ** 2 * xx.cuda())
            + (2 * alpha * beta * px.cuda())
    )

    # now in order to project the mobius addition onto the hyperbolic disc
    # we need to divide vectors whos l2norm : |x| (not |x|^2) are higher than max norm
    maxnorm = (1.0 - PROJ_EPS) / c**0.5

    # we can do this also after the dot with a as its a scalar division
    #print('torch.greater(torch.sqrt(mobaddnorm), maxnorm) = ', torch.greater(torch.sqrt(mobaddnorm), maxnorm))
    project_normalized = torch.where(
        torch.greater(torch.sqrt(mobaddnorm), maxnorm), 
        maxnorm / torch.maximum(torch.sqrt(mobaddnorm), torch.tensor(EPS)),  
        torch.ones_like(mobaddnorm)
    ) 
    assert not torch.isnan(project_normalized).any()

    mobaddnormprojected = torch.where(
        torch.less(torch.sqrt(mobaddnorm), maxnorm),  
        mobaddnorm,  
        torch.ones_like(mobaddnorm) * maxnorm ** 2,
    )
    assert not torch.isnan(mobaddnormprojected).any()

    inputs = inputs.permute((0,3,1,2))
    A_kernel = A_kernel.permute((3,2,0,1))
    xdota = cross_correlate_torch(inputs, A_kernel)
    xdota = xdota.permute((0,2,3,1))
    inputs = inputs.permute((0,2,3,1))
    A_kernel = A_kernel.permute((2,3,1,0))
 
    xdota = beta * xdota

    pdota = (
            alpha.cuda() * torch.sum(-P_mlr.cuda() * normed_A.cuda(), dim=1)[None, None, None, :].cuda()
    )  # ncls
    mobdota = xdota + pdota  # sh B,H,W,ch
    mobdota *= project_normalized  

    lamb_px = 2.0 / torch.maximum(1 - c * mobaddnormprojected, torch.tensor(EPS))
    assert not torch.isnan(lamb_px).any()

    sineterm = c**0.5 * mobdota * lamb_px
    return 2.0 / c**0.5 * A_norm.cuda() * torch.asinh(sineterm).cuda()        
