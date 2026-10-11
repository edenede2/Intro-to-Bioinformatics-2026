"""Row-vector coordinates: transformed points = points @ rotation + translation."""
import numpy as np
from scipy.spatial.distance import cdist


def points(x,minimum=3):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=3 or not minimum<=len(x)<=200 or not np.isfinite(x).all():
        raise ValueError(f'Use {minimum}–200 finite 3D points.')
    return x


def quaternion_rotation(axis,angle):
    axis=np.asarray(axis,float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)==0: raise ValueError('Rotation axis must be a nonzero finite vector.')
    axis/=np.linalg.norm(axis); w=np.cos(angle/2); x,y,z=axis*np.sin(angle/2)
    column=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return column.T,np.array([w,x,y,z])


def kabsch(source,target):
    p,q=points(source),points(target)
    if p.shape!=q.shape: raise ValueError('Corresponding point sets must have the same shape.')
    pc,qc=p-p.mean(axis=0),q-q.mean(axis=0)
    if np.linalg.matrix_rank(pc)<2 or np.linalg.matrix_rank(qc)<2: raise ValueError('At least three non-collinear points are required for a stable rotation.')
    u,s,vt=np.linalg.svd(pc.T@qc); correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
    r=u@correction@vt; t=q.mean(axis=0)-p.mean(axis=0)@r; fitted=p@r+t
    residual=np.linalg.norm(fitted-q,axis=1)
    return {'rotation':r,'translation':t,'fitted':fitted,'rmsd':float(np.sqrt(np.mean(residual**2))),'residual':residual,'singular_values':s}


def reconstruct(distances):
    d=np.asarray(distances,float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not 3<=len(d)<=200 or not np.isfinite(d).all() or (d<0).any() or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):
        raise ValueError('Distance matrix must be finite, symmetric, nonnegative, square, and have a zero diagonal (3–200 points).')
    j=np.eye(len(d))-np.ones_like(d)/len(d); gram=-0.5*j@(d*d)@j
    values,vectors=np.linalg.eigh(gram); order=np.argsort(values)[::-1]; values=values[order]; vectors=vectors[:,order]
    coords=vectors[:,:3]*np.sqrt(np.maximum(values[:3],0))
    return coords,values,float(np.sqrt(np.mean((cdist(coords,coords)-d)**2)))


def parse_pdb(text):
    """Read the first model, C-alpha atoms, blank/A alternate locations; deduplicate residues."""
    coords=[]; residues=[]; seen=set()
    for line in text.splitlines():
        if line.startswith('ENDMDL'): break
        if not line.startswith('ATOM  ') or line[12:16].strip()!='CA' or line[16:17] not in (' ','A'): continue
        key=(line[21:22],line[22:27])
        if key in seen: continue
        try: xyz=[float(line[30:38]),float(line[38:46]),float(line[46:54])]
        except ValueError as exc: raise ValueError('Malformed PDB C-alpha coordinates.') from exc
        seen.add(key); coords.append(xyz); residues.append(f'{key[0]}:{key[1].strip()} {line[17:20]}')
    return points(coords),residues


def structure_align(p,q,iterations=8,gap=-3):
    """Local educational method: ordered distance-DP correspondences and Kabsch.
    Starts with equally spaced correspondences; no guarantee of a global optimum.
    """
    p,q=points(p),points(q); k=min(len(p),len(q))
    ip=np.linspace(0,len(p)-1,k).astype(int); iq=np.linspace(0,len(q)-1,k).astype(int)
    fit=kabsch(p[ip],q[iq]); history=[]
    for _ in range(iterations):
        moved=p@fit['rotation']+fit['translation']; distances=cdist(moved,q)
        n,m=distances.shape; d=np.zeros((n+1,m+1)); ptr=np.zeros_like(d,int)
        d[:,0]=np.arange(n+1)*gap; d[0,:]=np.arange(m+1)*gap; ptr[1:,0]=1; ptr[0,1:]=2
        for i in range(1,n+1):
            for j in range(1,m+1):
                vals=[d[i-1,j-1]+3-distances[i-1,j-1],d[i-1,j]+gap,d[i,j-1]+gap]; ptr[i,j]=np.argmax(vals); d[i,j]=max(vals)
        ip=[]; iq=[]; i,j=n,m
        while i or j:
            state=ptr[i,j]
            if state==0: ip.append(i-1); iq.append(j-1)
            if state!=2: i-=1
            if state!=1: j-=1
        ip,iq=ip[::-1],iq[::-1]
        if len(ip)<3: raise ValueError('Fewer than three correspondences; try a stronger gap penalty.')
        fit=kabsch(p[ip],q[iq]); history.append({'iteration':len(history)+1,'pairs':len(ip),'rmsd':fit['rmsd'],'alignment_score':float(d[n,m])})
    return fit,ip,iq,history
