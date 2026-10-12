import numpy as np
rng=np.random.default_rng(0)

# (a) second-order softmax bias under heteroscedastic Gaussian logit noise
n=64; s=rng.normal(0,2.0,n); p=np.exp(s)/np.exp(s).sum()
sig2=rng.uniform(0.0,0.3,n)  # heteroscedastic variances
N=400000
eps=rng.normal(0,1,(N,n))*np.sqrt(sig2)
ph=np.exp(s+eps); ph/=ph.sum(1,keepdims=True)
mc=ph.mean(0)-p
pred=(p/2)*(sig2*(1-2*p)-np.sum(p*sig2*(1-2*p)))
print("(a) heteroscedastic: corr(MC,pred)=%.4f  max|MC-pred|/max|MC|=%.3f"%(np.corrcoef(mc,pred)[0,1], np.abs(mc-pred).max()/np.abs(mc).max()))
# homoscedastic flattening check
sig=0.3; eps=rng.normal(0,sig,(N,n)); ph=np.exp(s+eps); ph/=ph.sum(1,keepdims=True)
mc=ph.mean(0)-p; pred=sig**2*p*((p**2).sum()-p)
print("(a) homoscedastic: corr=%.4f ratio=%.3f ; entropy FP=%.4f noisy-mean=%.4f"%(np.corrcoef(mc,pred)[0,1], np.abs(mc-pred).max()/np.abs(mc).max(), -(p*np.log(p)).sum(), -(ph.mean(0)*np.log(ph.mean(0))).sum()))

# (b) block-scaled FP4: NVFP4 (B=16, E4M3 scale) vs MXFP4 (B=32, UE8M0 round-up / round-down)
E2M1=np.array([0,0.5,1,1.5,2,3,4,6.0])
def q_grid(x,grid):
    idx=np.abs(x[...,None]-grid).argmin(-1); return grid[idx]
def block_mse(B,scale_mode,N=200000):
    X=rng.normal(0,1,(N,B)); M=np.abs(X).max(1); s=M/6.0
    if scale_mode=='e4m3':
        # 3 mantissa bits -> relative rounding error roughly uniform in +-2^-4 (ignore exponent range)
        s_q=s*(1+rng.uniform(-2**-4,2**-4,N))
    elif scale_mode=='pow2_up':
        s_q=2.0**np.ceil(np.log2(s))
    elif scale_mode=='pow2_down':
        s_q=2.0**np.floor(np.log2(s))
    else: s_q=s
    Y=X/s_q[:,None]; Yq=np.sign(Y)*q_grid(np.minimum(np.abs(Y),6.0),E2M1)
    return ((X-Yq*s_q[:,None])**2).mean()
nv=block_mse(16,'e4m3'); mxu=block_mse(32,'pow2_up'); mxd=block_mse(32,'pow2_down'); ex16=block_mse(16,'exact')
print("(b) MSE NVFP4-like=%.4f  MXFP4 round-up=%.4f  MXFP4 round-down=%.4f  exact-scale B16=%.4f ; RMS ratio MXup/NV=%.3f  MXdown/NV=%.3f"%(nv,mxu,mxd,ex16,np.sqrt(mxu/nv),np.sqrt(mxd/nv)))
# shrinkage bias of E2M1 RTN on Gaussian scaled to max 6 (B=16 exact scale)
X=rng.normal(0,1,(200000,16)); M=np.abs(X).max(1); s=M/6.0; Y=X/s[:,None]; Yq=np.sign(Y)*q_grid(np.abs(Y),E2M1)
print("(b) E2M1 shrinkage: E[|Q(x)|-|x|]/E|x| = %.4f ; E[Q(x)x]/E[x^2]=%.4f"%(((np.abs(Yq)-np.abs(Y)).mean()/np.abs(Y).mean()), (Yq*Y).mean()/(Y**2).mean()))

# (c) sigma-delta vs naive requantization of a slow SSM channel
def run(a,T,Delta,mode,br=None):
    h=0.0; hq=0.0; r=0.0; errs=[]
    for t in range(T):
        u=rng.normal(0,1e-3)   # small inputs -> slow drift
        h=a*h+u
        if mode=='rtn':
            hq=Delta*np.round((a*hq+u)/Delta)
        elif mode=='sr':
            v=(a*hq+u)/Delta; fl=np.floor(v); hq=Delta*(fl+(rng.random()<v-fl))
        elif mode=='sd':
            v=a*hq+u+r; hq=Delta*np.round(v/Delta); r=v-hq
            if br is not None:  # quantize residual to br bits over [-Delta/2,Delta/2]
                d2=Delta/2**br; r=d2*np.round(r/d2)
        errs.append(hq-h)
    return np.array(errs)
a=1-1e-3; T=20000; Delta=0.02
for mode,br in [('rtn',None),('sr',None),('sd',None),('sd',4)]:
    e=run(a,T,Delta,mode,br); print("(c) a=%.3f mode=%s br=%s : max|err|/Delta=%.2f  final|err|/Delta=%.2f"%(a,mode,br,np.abs(e).max()/Delta,abs(e[-1])/Delta))

# (d) merge SNR: requantize W+D with |d|<<Delta, uniform offsets
Delta=1.0; n=2_000_000; W=rng.uniform(-50,50,n); d=rng.normal(0,0.05,n)
Dt=np.round((W+d)/Delta)*Delta-np.round(W/Delta)*Delta
print("(d) E[Dt]/E[d] slope=%.3f  Var(Dt)=%.4f  pred Delta*E|d|-E[d^2]=%.4f  SNR=||d||^2/||Dt-d||^2=%.4f pred=%.4f"%(np.polyfit(d,Dt,1)[0],Dt.var(),Delta*np.abs(d).mean()-(d**2).mean(),(d**2).sum()/((Dt-d)**2).sum(),(d**2).mean()/(Delta*np.abs(d).mean()-(d**2).mean())))
