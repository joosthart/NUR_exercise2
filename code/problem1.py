import os

import numpy as np
import matplotlib.pyplot as plt

def LCG(x, a=np.uint64(1664525), c=np.uint64(1013904223)):
    return (a*x + c) % 2**64
        
def XOR_shift(x, a1=np.uint64(13), a2=np.uint64(11), a3=np.uint64(3)):
    x = np.uint64(x)
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3)
    return x

def RNG(x):
    while True:
        x = LCG(np.uint64(x))
        x = XOR_shift(x)
        # x += XOR_shift(x)
        x = LCG(x, 
                np.uint64(6364136223846793005), 
                np.uint64(1442695040888963407)
            )
        yield x / (2**(64)-1)
        
def pearson_correlation(x, y):
    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)
    
    sumxy = sum([(i-xmean)*(j-ymean) for i,j in zip(x, y)])
    sumxx = sum([(i-xmean)**2 for i in x])
    sumyy = sum([(i-ymean)**2 for i in y])
    
    return sumxy * (sumxx*sumyy)**(-0.5)

def hernquist(r, Mdm=1e12, a=80):
    return Mdm/(2*np.pi) * a/(r*(r+a)**3) 

def dhernquistdr(r, Mdm=1e12, a=80):
    return -Mdm*a/(2*np.pi)*((a+4*r)/(r**2*(r+a)**4))

def bracketing(f, a, b, w=1.618):
    if f(b) > f(a):
        a, b = b, a
    
    c = b + (b-a)*w
    
    if f(c) > f(b):
        return [a, b, c]
    
    d = parabola_min(f,a,b,c)
    
    if f(d)<f(c):
        return [b, d, c]
    elif f(d)>f(b):
        return [a,b,d]
    elif abs(d-b) > 100*abs(c-b):
        d = c+(c-b)*w
        return [b,c,d]
    else:
        return[b,c,d]

def golden_section(f, xmin, xmax, target_acc=1e-6, maxit=1e4):
    
    w = 0.38197 # 2-phi
    i = 0
    a,b,c = bracketing(f,xmin, xmax)
    
    while i < maxit:
        if abs(c-b) > abs(b-a):
            x1, x2 = b, c
        else:
            x1, x2 = a, b

        d = b + (x2 -x1)*w

        if abs(c-a) < target_acc:
            if f(d) < f(b):
                return d
            else:
                return d

        if f(d) < f(b):
            if x1 == b and x2 == c:
                a, b = b, d
            elif x1 == a and x2 == b:
                c, b = b, d
        else:
            if x1 == b and x2 == c:
                c = d
            elif x1 == a and x2 == b:
                a = d
        i+=1

    return d

def central_difference(f, x, h=None):
    if not h:
        h = sum([abs(x[i+1]-x[i]) for i in range(len(x)-1)])/len(x)
    dfdx = (f(x+h)-f(x-h))/(2*h)
    return dfdx

def ridders(f,x,m,d,target_err):
    h = [0.1]
    dfdx_hat = [central_difference(f,x,h[-1])]
    while len(dfdx_hat) < m:
        h.append(h[-1]/d)
        dfdx_hat.append(central_difference(f,x,h[-1]))
        
    for k in range(len(dfdx_hat)):
        D = np.zeros((np.shape(dfdx_hat)[0], np.shape(dfdx_hat)[0]))
        D[:,0] = dfdx_hat
        prev = 0
        for i in range(len(dfdx_hat)-1): #columns
            for j in range(len(dfdx_hat)-i-1):
                D[i, j+1] = (d**(2*(j+1))*D[i+1,j] - D[i,j])/(d**(2*(j+1))-1)
            if abs(D[i,0]-prev)>target_err:
                prev=D[i,0]
            else:
                print('target error reached')
                return D[i,0]
        return D[i,0]

def bisection(f, a, b, acc=1e-6):
    c = 0.5*(a+b)
    fc = f(c)
    if abs(fc) <= acc:
        return c
    fa = f(a)
    fb = f(b)
    if fa*fc < 0:
        return bisection(f,a,c, acc)
    elif fb*fc < 0:
        return bisection(f,c,b, acc)
    else:
        raise RuntimeError("No root found in interval ({}, {})".format(a,b))

def hernquist_potential(x, y, Mdm=1e12, a=80, G=1):
    return -G*Mdm / ( ((x-1.3)**2 + 2*(y-4.2)**2)**(0.5) +a)

def grad_hernquist_potential(x,y, Mdm=1e12, a=80, G=1):
    pot = hernquist_potential(x,y,Mdm,a)
    denominator = ((x-1.3)**2 + 2*(y-4.2)**2)**(1/2)
    factor=1/(G*Mdm)
    dpotdx = factor * (x-1.3)/denominator * pot**2
    dpotdy = factor * 2*(y-4.2)/denominator * pot**2
    return np.array([dpotdx, dpotdy]).T

def quasi_newton(f, gradf, startx, starty, target_acc=1e-6, maxit=1000):
    
    pos = []
    xi = np.array([startx, starty]).T
    Hi = np.diag((1,1))
    i = 0
    while i < maxit:
        pos.append(xi)
        
        
        gradf = grad_hernquist_potential(xi[0], xi[1])
        
        ni = -Hi.dot(gradf)
        func_l = lambda l: hernquist_potential(x=xi[0]+l*ni[0], y=xi[1]+l*ni[1])
        l = golden_section(func_l, 0, 10000 ,target_acc=1e-6)
        
        x_i1 = xi + l*ni
        delta_i1 = l*ni
        
        D_i1 = grad_hernquist_potential(x_i1[0], x_i1[1]) - gradf  
        
        
        if all(abs(d)<target_acc for d in D_i1):
            print('target acc reached')
            return pos, x_i1
        
        HD = Hi.dot(D_i1)
        
        u = delta_i1/(delta_i1 @ D_i1) - (HD)/(D_i1 @ (HD))
        
        Hi = Hi \
               + np.outer(delta_i1, delta_i1)/(delta_i1 @ D_i1) \
               - np.outer(HD, HD)/(D_i1 @ HD) \
               + (D_i1 @ HD)*np.outer(u,u)
        
        xi = x_i1
        i += 1
    return pos ,x_i1

if __name__ == '__main__':

    # setting constants
    seed = 42
    N = int(1e6)
    plot_dir = './plots/'
    output_dir = './output/'

    print('random seed: ', seed)

    #1a
    
    rng = RNG(seed)
    rand_n = [next(rng) for _ in range(N)]

    #1a1
    plt.figure(figsize=(5,5))
    plt.plot(rand_n[1:1000+1], rand_n[0:1000], '.')
    plt.axis(
        xmin = min(rand_n[1:1000+1]),
        xmax = max(rand_n[1:1000+1]),
        ymin = min(rand_n[0:1000]),
        ymax = max(rand_n[0:1000])
    )
    plt.xlabel(r'random value $i$')
    plt.ylabel(r'random value $i+1$')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '1a_sequential_random_numbers.png'))
    plt.clf()

    #1a2
    plt.figure(figsize=(5,5))
    plt.hist(rand_n, 20, rwidth=0.9, alpha=0.8)
    plt.axis(ymin=49500, ymax=50500, xmin=0, xmax=1)
    plt.hlines(
        50000, 
        0, 
        1, 
        ls='-', 
        color='C1', 
        zorder=1, 
        label='$\lambda$'
    )
    plt.hlines(
        50000-np.sqrt(50000)
        , 
        0, 
        1, 
        ls='--', 
        color='C1', 
        zorder=1,
        label = r'$1\sigma$'
    )
    plt.hlines(
        50000+np.sqrt(50000)
        , 
        0, 
        1, 
        ls='--', 
        color='C1', 
        zorder=1
    )
    plt.xlabel('random value')
    plt.ylabel('number of occurances')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,'1a_histogram_random_numbers.png'))
    plt.clf()

    #1a3
    correlation1 = pearson_correlation(rand_n[:int(1e5)-1],rand_n[1:int(1e5)])
    correlation2 = pearson_correlation(rand_n[:int(1e5)-1],rand_n[2:int(1e5)+1])
    with open(
             os.path.join(output_dir, '1a_pearson_correlation1.txt'), 'w'
         ) as f:
        f.write('{:0.5f}'.format(correlation1))
    with open(
             os.path.join(output_dir, '1a_pearson_correlation2.txt'), 'w'
         ) as f:
        f.write('{:0.5f}'.format(correlation2))
    
    #1b
    # Sample in spherical coordinates
    theta = np.arccos([1 - 2*next(rng) for _ in range(N)])
    phi = np.array([2*np.pi*next(rng) for _ in range(N)])
    r = np.array(rand_n)**(1/3)

    # Transform Spherical to Carthesian coordinates
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    # Calculate m(<r)
    cumsum_m = []
    steps = 100
    xrange = np.linspace(0,1,steps)
    for i in xrange:
        cumsum_m.append(np.sum(r<i)/N)
    
    # Plot results
    plt.figure(figsize=(5,5))
    plt.loglog(xrange, cumsum_m, label='random sample', ls='--')
    plt.loglog(xrange, xrange**3, label='theoretical', ls='-.')
    plt.legend()
    plt.xlabel('radius')
    plt.ylabel('enclosed mass fraction')
    plt.axis(xmin=1e-2, ymin=1e-6, xmax=1, ymax=1)
    plt.savefig(os.path.join(plot_dir, '1b_enclosed_mass_fraction.png'))
    plt.clf()

    #1c
    n = int(1e3)
    a = 80 #kpc

    # Use samples from previous problem
    theta = theta[:n]
    phi = phi[:n]
    u = np.array(rand_n)[:n]
    r = a*u**0.5/(1-u**0.5)

    # Transform Spherical to Carthesian coordinates
    x = r*np.sin(theta)*np.cos(phi) * 1e-3
    y = r*np.sin(theta)*np.sin(phi) * 1e-3
    z = r*np.cos(theta) * 1e-3

    plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z,s=1)
    ax.set_xlabel(r'$x$ (Mpc)')
    ax.set_ylabel(r'$y$ (Mpc)')
    ax.set_zlabel(r'$z$ (Mpc)')
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_zticks([-10, -5, 0, 5, 10])
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    plt.savefig(os.path.join(plot_dir, '1c_3d_scatter_hernquist.png'))
    plt.clf()


    plt.figure(figsize=(5,2.5))
    plt.plot(theta, phi, '.')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    plt.axis(xmin=0, xmax=np.pi, ymin=0, ymax=2*np.pi)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '1c_3d_scatter_phi_theta.png'))
    plt.clf()

    #1d
    m=10

    # estimate d(rho)/dr using ridders' method
    drhodr_hat = ridders(hernquist, 1.2*80, m, 2, 1e-12)
    drhodr = dhernquistdr(1.2*80)
    with open(
             os.path.join(output_dir, '1d_derivative_hernquist_hat.txt'), 'w'
         ) as f:
        f.write('{:.9f}'.format(drhodr_hat))
    with open(
             os.path.join(output_dir, '1d_derivative_hernquist.txt'), 'w'
         ) as f:
        f.write('{:.9f}'.format(drhodr))
    
    #1e
    rho_crit = 150

    # Calculate R200 and R500 using bisection algorithm
    R200 = bisection(lambda x: hernquist(x) - 200*rho_crit, 80, 100)
    R500 = bisection(lambda x: hernquist(x) - 500*rho_crit, 60, 65)
    with open(os.path.join(output_dir, '1e_R200.txt'), 'w') as f:
        f.write('{:.4f}'.format(R200))
    with open(os.path.join(output_dir, '1e_R500.txt'), 'w') as f:
        f.write('{:.4f}'.format(R500))

    #1f
    
    # Search center using Quasi Newton method
    pos, (x_hat, y_hat) = quasi_newton(
        hernquist_potential, 
        grad_hernquist_potential, 
        -1000, 
        -200
    )
    # Calculate distance to centre for every iteration
    distance = [((xi[0]-1.3)**2 + (xi[1]-4.2)**2)**0.5 for xi in pos]

    plt.figure(figsize=(5,2.5))
    plt.semilogy(distance)
    plt.axis(xmin=0, xmax=len(distance)-1)
    plt.xlabel('iteration')
    plt.ylabel('distance to minimum (kpc)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '1f_3d_distance_to_minimum.png'))
    plt.clf()
