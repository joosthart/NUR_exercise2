import os

import numpy as np
import matplotlib.pyplot as plt

def LCG(x, a=np.uint64(1664525), c=np.uint64(1013904223)):
    """Linear Congruential Generator

    Args:
        x (int): Seed
        a (uint64, optional): Miltiplier. Defaults to np.uint64(1664525).
        c (uint64, optional): Increment. Defaults to np.uint64(1013904223).

    Returns:
        int: Next generated pseudo-random-number
    """
    return (a*x + c) % 2**64
        
def XOR_shift(x, a1=np.uint64(13), a2=np.uint64(11), a3=np.uint64(3)):
    """XOR-shift random number generator

    Args:
        x (int): state
        a1 (unit64, optional): First right bitshift. Defaults to np.uint64(13).
        a2 (unit64, optional): First left bitshift. Defaults to np.uint64(11).
        a3 (uint64, optional): second right bitshift. Defaults to np.uint64(3).

    Returns:
        int: Next generated pseudo-random-number
    """
    x = np.uint64(x)
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3)
    return x

def RNG(x):
    """Pseudo-random-number generator using a combination of linear congruential 
    generatorts and XOR-shift generators. The structure of the generator is as 
    follows:
        LCG1 -> (XOR1 ^ XOR2) -> LCG2.
    
    

    Args:
        x ([type]): [description]

    Yields:
        [type]: [description]
    """
    while True:
        x = LCG(np.uint64(x))
        x = XOR_shift(x) ^ XOR_shift(x, 
                                     np.uint64(15), 
                                     np.uint64(13), 
                                     np.uint64(9)
                               )
        x = LCG(
            x, 
            np.uint64(6364136223846793005), 
            np.uint64(1442695040888963407)
        )
        yield x / (2**(64)-1)
        
def pearson_correlation(x, y):
    """ Pearson correlation for lists x an y.

    Args:
        x (list): list with floats
        y (list): list with floats

    Returns:
        float: correlation between lists x and y
    """
    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)
    
    sumxy = sum([(i-xmean)*(j-ymean) for i,j in zip(x, y)])
    sumxx = sum([(i-xmean)**2 for i in x])
    sumyy = sum([(i-ymean)**2 for i in y])
    
    return sumxy * (sumxx*sumyy)**(-0.5)

def hernquist(r, Mdm=1e12, a=80):
    """ Hernquist profile for dark matter halos.

    Args:
        r (float): Radius in kpc
        Mdm (float, optional): Total mass in solar mass. Defaults to 1e12.
        a (float, optional): scale length in kpc. Defaults to 80.

    Returns:
        float: density at radius r
    """
    return Mdm/(2*np.pi) * a/(r*(r+a)**3) 

def dhernquistdr(r, Mdm=1e12, a=80):
    """Derivative of Hernquist profile.

    r (float): Radius in kpc
        Mdm (float, optional): Total mass in solar mass. Defaults to 1e12.
        a (float, optional): scale length in kpc. Defaults to 80.

    Returns:
        float: d(rho)/dr | r
    """
    return -Mdm*a/(2*np.pi)*((a+4*r)/(r**2*(r+a)**4))

def parabola_min(f, x1, x2, x3):
    """Find the minimum a parabole going through points x1, x2 and x3.

    Args:
        f (callable): function to calculate yi values corresponding to xi
        x1 (float): x1 coordinate
        x2 (float): x2 coordinate
        x3 (float): x3 coordinate

    Returns:
        float: x coordinate of minimum
    """
    f1,f2,f3 = f(x1), f(x2), f(x3)
    numerator = (x2-x1)**2*(f2-f3) - (x2-x3)**2*(f2 - f1)
    denominator = (x2-x1)*(f2-f3) - (x2-x3)*(f2 - f1)
    
    return x2 - 0.5 * numerator / denominator

def bracketing(f, a, b, w=1.618):
    """ Bracketing a minimum, using parabolic interpolation.

    Args:
        f (callable): Function for which to find root
        a (float): boundry of bracket
        b (float): boundry of bracket
        w (float, optional): splitting fraction of bracket. Defaults to 1.618.

    Returns:
        list/float: list of float containig bracket
    """

    # ensure that a < b
    if f(b) > f(a):
        a, b = b, a
    
    # make a guess for c
    c = b + (b-a)*w
    
    # if on the right hand side of b, retrun bracket [a,b,c]
    if f(c) > f(b):
        return [a, b, c]
    
    # find the minimum of the parabola throuh [a,b,c]
    d = parabola_min(f,a,b,c)
    
    # find out the order of the new bracket and return smallest bracket
    if f(d)<f(c):
        return [b, d, c]
    elif f(d)>f(b):
        return [a,b,d]
    # if d is to far from b, take section step
    elif abs(d-b) > 100*abs(c-b):
        d = c+(c-b)*w
        return [b,c,d]
    else:
        return[b,c,d]

def golden_section(f, xmin, xmax, target_acc=1e-6, maxit=1e4):
    """Finding the mimimum of a function, f, in the range [xmin, xmax] using the 
    Golden section algorithm.

    Args:
        f (callable): Function fo which to find minimum
        xmin (float): left boundry of bracket
        xmax (float): right boundry of bracket
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.
        maxit (int, optional): Maximum number of iterations. Defaults to 1e4.

    Returns:
        float: x-value of the obtained minimum
    """
    w = 0.38197 # 2-phi
    i = 0
    # Bracket the minimum using bracketing algorithm
    a,b,c = bracketing(f,xmin, xmax)
    
    while i < maxit:
        # Identify larger interval
        if abs(c-b) > abs(b-a):
            x1, x2 = b, c
        else:
            x1, x2 = a, b

        # Choose new point in a self similar way
        d = b + (x2 -x1)*w

        # abort if target acc reached and return best value
        if abs(c-a) < target_acc:
            if f(d) < f(b):
                return d
            else:
                return b

        # Tighten the bracket
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

    # if maxit reached, return last d
    return d

def central_difference(f, x, h=None):
    """Calculate estimate for derivative of function f for list of x-coordinates x.

    Args:
        f (callable): Function for which to calculate derivative
        x (list): list of x values for which to calculate central difference.
        h (float, optional): Stap size of central difference. Defaults to None.

    Returns:
        list: list of y values of derivatives
    """
    # if h is not specified take mean difference between values in x
    if not h:
        h = sum([abs(x[i+1]-x[i]) for i in range(len(x)-1)])/len(x)
    # calculate central difference
    dfdx = (f(x+h)-f(x-h))/(2*h)
    return dfdx

def ridders(f,x,m,d,target_err):
    """Estimate the derivative of a function, f, for coordinates x. Order of the 
    Ridders' method can be specified before hand. Function will be terminated if 
    a target accuracy is reached.

    Args:
        f (callable): Function for which to calculate derivative
        x (float): x value for which to calculate derivative
        m (int): Order of approximation
        d (float): Factor by which to decrease interval in central difference
        target_acc (float, optional): Target accuracy.

    Returns:
        float: estimate of derivative at x
    """
    # initial step sice for central difference
    h = [0.1]
    # first estimate 
    dfdx_hat = [central_difference(f,x,h[-1])]
    # first order estimation for m values of h (hi = h0/(i*d)) 
    while len(dfdx_hat) < m:
        h.append(h[-1]/d)
        dfdx_hat.append(central_difference(f,x,h[-1]))
    
    for k in range(len(dfdx_hat)):
        D = np.zeros((np.shape(dfdx_hat)[0], np.shape(dfdx_hat)[0]))
        D[:,0] = dfdx_hat
        prev = 0

        # combine pairs of results 
        for i in range(len(dfdx_hat)-1): #columns
            for j in range(len(dfdx_hat)-i-1):
                D[i, j+1] = (d**(2*(j+1))*D[i+1,j] - D[i,j])/(d**(2*(j+1))-1)
            # only continue when improvement over previous best approximation 
            # estimate is smaller than the target error.
            if abs(D[i,0]-prev)>target_err:
                prev=D[i,0]
            else:
                # abort id target accuracy is reached
                return D[i,0]
        return D[i,0]

def bisection(f, a, b, target_err=1e-6):
    """Recursive implementation of bisection root finding algorithm. Keeps 
    splitting the interval in half, and keep interval which encloses root.

    Args:
        f (callable): Function for which to calculate derivative
        a (float): boundry of bracket
        b (float): boundry of bracket
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.

    Raises:
        RuntimeError: no root inside bracket [a,b]

    Returns:
        float: x value of root of
    """
    # split interval in half
    c = 0.5*(a+b)
    fc = f(c)
    # stop when f(c) is closer to 0 than target accuracy
    if abs(fc) <= target_err:
        return c
    fa = f(a)
    fb = f(b)
    # find bracket enclosing root and recursivelly call bisection with the 
    # smaller interval
    if fa*fc < 0:
        return bisection(f,a,c, target_err)
    elif fb*fc < 0:
        return bisection(f,c,b, target_err)
    # abort when nu root is inbetween [a,c] or [b,c]
    else:
        raise RuntimeError("No root found in interval ({}, {})".format(a,b))

def hernquist_potential(x, y, x_center=1.3, y_center=4.2, Mdm=1e12, a=80, G=1):
    """ 2D Hernquist potential with center at (x_center, y_center).

    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc
        x_center (float): x coordinate of center
        y_center (float): y coordinate of center
        Mdm (float, optional): Total mass in solar mass. Defaults to 1e12.
        a (float, optional): scale length in kpc. Defaults to 80.
        G (flaot, optional): Gravitational constant. Defaults to 1.

    Returns:
        float: potential at (x,y)
    """
   
    return -G*Mdm / ( ((x-x_center)**2 + 2*(y-y_center)**2)**(0.5) +a)

def grad_hernquist_potential(
        x, y, x_center=1.3, y_center=4.2, Mdm=1e12, a=80, G=1
    ):
    """ Gradient of 2D Hernquist potential with center at (x_center, y_center).

    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc
        x_center (float): x coordinate of center
        y_center (float): y coordinate of center
        Mdm (float, optional): Total mass in solar mass. Defaults to 1e12.
        a (float, optional): scale length in kpc. Defaults to 80.
        G (flaot, optional): Gravitational constant. Defaults to 1.

    Returns:
        list: gradient of potential at (x,y)
    """
    pot = hernquist_potential(x,y,x_center,y_center,Mdm,a)
    denominator = ((x-x_center)**2 + 2*(y-y_center)**2)**(1/2)
    factor=1/(G*Mdm)
    dpotdx = factor * (x-x_center)/denominator * pot**2
    dpotdy = factor * 2*(y-y_center)/denominator * pot**2
    return np.array([dpotdx, dpotdy]).T

def quasi_newton(f, gradf, startx, starty, target_acc=1e-6, maxit=1000):
    """Quasi-Newton methode for finding minimum of 2D function. Function makes 
    use of BFGS methode for updating hessian.

    Args:
        f (callabl): Fucntion which has as input an x and y coordinate
        gradf (callable): Function that calculates the gradient at x and y
        startx (float): Starting x point of minimum search
        starty (float): Starting y point of minimum search
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.
        maxit (int, optional): Maximum number of iterations. Defaults to 1e4.

    Returns:
        list, list: list containig all position during search and list with 
            final x,y coordinate
    """
    pos = []
    xi = np.array([startx, starty]).T
    # initial hessian
    Hi = np.diag((1,1))
    i = 0
    while i < maxit:
        pos.append(xi)
        
        # Calculate gradiant at (x,y)
        gradf = grad_hernquist_potential(xi[0], xi[1])
        
        # obtain direction for the next step
        ni = -Hi.dot(gradf)
        # minimize x_i+1 = x_i + l_i*n_i using golden section algorithm.=
        func_l = lambda l: hernquist_potential(x=xi[0]+l*ni[0], y=xi[1]+l*ni[1])
        l = golden_section(func_l, 0, 10000 ,target_acc=1e-6)
        
        x_i1 = xi + l*ni
        delta_i1 = l*ni
        
        D_i1 = grad_hernquist_potential(x_i1[0], x_i1[1]) - gradf  
        
        # Check for convergence
        if all(abs(d)<target_acc for d in D_i1):
            return pos, x_i1
        
        # Update hessian according to BFGS method
        HD = Hi.dot(D_i1)
        
        u = delta_i1/(delta_i1 @ D_i1) - (HD)/(D_i1 @ (HD))
        
        Hi = Hi \
               + np.outer(delta_i1, delta_i1)/(delta_i1 @ D_i1) \
               - np.outer(HD, HD)/(D_i1 @ HD) \
               + (D_i1 @ HD)*np.outer(u,u)
        
        # take the step
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
        label=r'$\lambda$'
    )
    plt.hlines(
        50000-np.sqrt(50000), 
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
    a = 80 #kpc

    # Sample r in spherical coordinates for Hernquist profile
    r = a*np.array(rand_n)**0.5/(1-np.array(rand_n)**0.5)

    # Calculate m(<r)/M
    cumsum_m = []
    steps = 1000
    r_range = np.logspace(0,np.log10(max(r)),steps)
    for i in r_range:
        cumsum_m.append(np.sum(r<i)/len(r))
    
    cumsum_theoratical = r_range**2/(a+r_range)**2
    
    # Plot results
    plt.figure(figsize=(5,5))
    plt.loglog(r_range, cumsum_m, label='random sample', ls='--')
    plt.loglog(r_range, cumsum_theoratical, label='theoretical', ls='-.')
    plt.legend(loc=4)
    plt.xlabel('radius (kpc)')
    plt.ylabel('enclosed mass fraction')
    plt.axis(xmin=1, ymin=1e-4, xmax=max(r), ymax=1+0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '1b_enclosed_mass_fraction.png'))
    plt.clf()

    #1c
    n = int(1e3)
    a = 80 #kpc

    # Use samples from previous problem
    theta = np.arccos([1 - 2*next(rng) for _ in range(n)])
    phi = np.array([2*np.pi*next(rng) for _ in range(n)])
    r = r[:n]

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