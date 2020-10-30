import os
import statistics

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

def rejection_sampling(p, p_max, xrange, n, rng):
    """Rejection sampling of function p usning random number generator function rng. The maximum value of p is used to normalise the function.

    Args:
        p (callable): Distribution to sample
        p_max (float): Normalisation factor of p
        xrange (list): xmin an xmax value to sample for
        n (int): Number of points to sample
        rng (generator): Random number generator object

    Returns:
        list: x coordinates of accepted points
    """
    accepted_points = []
    while len(accepted_points) < n:
        x = next(rng) * (xrange[1]-xrange[0]) + xrange[0]
        y = next(rng)
        if p(x)/p_max > y: # accept x
            accepted_points.append(x)
    return accepted_points

def quicksort_with_indexing(l, index_l):
    """Implementation of quicksort algorithm. Hence, this implementation is not 
    entirely correct. It fails for list with duplicates. For the problems at 
    hand, this does not raise any problems. However this has to be fixed.

    This implementation is slightly different from quicksort(), since it keeps 
    track of the permutations on l.

    Pivot is choosen as median of first, median and last element.

    This implementation is recursive

    Args:
        l (list): list to sort
        index_l (list): [0,1,..,len(l)-2, len(l)-1]

    Returns:
        (list,list): sorted list, permutations list
    """
    # if len list is 2, return ellements in appropriate order
    if len(l)==2:
        pivot = l[-1]
        if l[0] > l[-1]:
            l[0], l[-1] = l[-1], l[0]
            # keep track of permutations
            index_l[0], index_l[-1] = index_l[-1], index_l[0]
        return l, index_l
    # if len list is smaller than 2, return list
    elif len(l) < 2:
        return l, index_l
    # Choose a pivot and put it in the right order
    else:
        pivot = statistics.median([l[0], l[len(l)>>1], l[-1]])
        if l[0] == pivot:
            l[0], l[len(l)>>1] = l[len(l)>>1], l[0]
            # keep track of permutations
            index_l[0], index_l[len(l)>>1] = index_l[len(l)>>1], index_l[0]
        elif l[-1] == pivot:
            l[-1], l[len(l)>>1] = l[len(l)>>1], l[-1]
            # keep track of permutations
            index_l[-1], index_l[len(l)>>1] = index_l[len(l)>>1], index_l[-1]
        idx_pivot = len(l)>>1
        if l[0] > l[-1]:
            l[0], l[-1] = l[-1], l[0]
            # keep track of permutations
            index_l[0], index_l[-1] = index_l[-1], index_l[0]
    
    i = 1
    j = len(l)-1
    while j >= i:
        # increase i until l[i] is larger than pivot
        while not l[i] >= pivot:
            i+=1
        # decrease j until l[j] is smaller than pivot
        while not l[j] <= pivot:
            j-=1
        # Indeces hace crossed, brak while loop
        if j < i:
            break

        # swap elements i and j
        l[i], l[j] = l[j], l[i]
        # keep track of permutations
        index_l[i], index_l[j] = index_l[j], index_l[i]
        
        # if the pivot is swaped, the index of the pivot changes and has to be 
        # updated.
        if i == idx_pivot:
            idx_pivot = j
            i += 1
        elif j == idx_pivot:
            idx_pivot = i
            j -= 1
                    
    # split list and permutation list below and above pivot 
    l_lower = l[:idx_pivot]
    l_upper = l[idx_pivot+1:]
    index_lower = index_l[:idx_pivot]
    index_upper = index_l[idx_pivot+1:]
    
    # recursively call quicksort for lower and upper list and permutaion list
    l_lower, index_lower = quicksort_with_indexing(l_lower, index_lower)
    l_upper, index_upper = quicksort_with_indexing(l_upper, index_upper)
    
    return l_lower + [pivot] + l_upper, \
           index_lower + [index_l[idx_pivot]] + index_upper  

def quicksort(l):
    """Implementation of quicksort algorithm. Hence, this implementation is not 
    entirely correct. It fails for list with duplicates. For the problems at 
    hand, this does not raise any problems. However this has to be fixed.

    Pivot is choosen as median of first, median and last element.

    This implementation is recursive

    Args:
        l (list): list to sort

    Returns:
        list: sorted list
    """
    # if len list is 2, return ellements in appropriate order
    if len(l)==2:
        pivot = l[-1]
        if l[0] > l[-1]:
            l[0], l[-1] = l[-1], l[0]
        return l
    # if len list is smaller than 2, return list
    elif len(l) < 2:
        return l
    # Choose a pivot and put it in the right order
    else:
        pivot = statistics.median([l[0], l[len(l)>>1], l[-1]])
        if l[0] == pivot:
            l[0], l[len(l)>>1] = l[len(l)>>1], l[0]
        elif l[-1] == pivot:
            l[-1], l[len(l)>>1] = l[len(l)>>1], l[-1]
        idx_pivot = len(l)>>1
        if l[0] > l[-1]:
            l[0], l[-1] = l[-1], l[0]
    
    i = 1
    j = len(l)-1
    while j >= i:
        # increase i until l[i] is larger than pivot
        while not l[i] >= pivot:
            i+=1
        # decrease j until l[j] is smaller than pivot
        while not l[j] <= pivot:
            j-=1
        # Indeces hace crossed, brak while loop
        if j < i:
            break

        # swap elements i and j
        l[i], l[j] = l[j], l[i]
        
        # if the pivot is swaped, the index of the pivot changes and has to be 
        # updated.
        if i == idx_pivot:
            idx_pivot = j
            i += 1
        elif j == idx_pivot:
            idx_pivot = i
            j -= 1
                    
    # split list below and above pivot
    l_lower = l[:idx_pivot]
    l_upper = l[idx_pivot+1:]
    
    # recursively call quicksort for lower and upper list
    l_lower = quicksort(l_lower)
    l_upper = quicksort(l_upper)
    
    return l_lower + [pivot] + l_upper

def n(x, A=256/(5*np.pi**(1.5)), a=2.4, b=0.25, c=1.6, N=100):
    """Number density profile"""
    return A*N*(x/b)**(a-3)*np.exp(-(x/b)**c)

def pdx(x, A=256/(5*np.pi**(1.5)), a=2.4, b=0.25, c=1.6):
    """Number density profile"""
    return 4*np.pi*x**2*A*(x/b)**(a-3)*np.exp(-(x/b)**c)

def poisson(k,l):
    """Poisson probability calculator. The caculation of e^-(l)l^k/k! is 
    performed in a loop. Thus, k! and l^k do not have to be calculated directly. 
    This prevents overflows up to higher k and l compared to direct caculations.
    Args:
        k (int): Integer for which to calculate probability
        l (float): Positive value for the mean of the distribution
    Returns:
        float: Poisson probability for P(k,l)
    Raises:
        ValueError: l should be larger than 0.
        ValueError: k should be an integer.
    """

    # Check l and k allowed values
    if l<0:
        raise ValueError("l should be larger than 0.")
    elif (
            not isinstance(k, int) and 
            not isinstance(k, np.int32) and 
            not isinstance(k, np.int64)
         ):
        raise ValueError("k should be an integer not {}.".format(type(k)))
    elif k < 0:
        ValueError("k should be larger than 0.")
    if k == 0:
        # l^k=1 and k! = 1
        return np.exp(-l)
    else:
        p = 1
        # Calculate l^k/k!*e^-l using a product
        e = np.exp(-l/k) # calculate ones
        for f in range(1, k+1):
            p*=l/f*e
        return p

if __name__=='__main__':

    # Setting constants
    seed = 42
    plot_dir = './plots/'
    output_dir = './output/'

    print('random seed: ', seed)

    #2a
    minus_Ndx = lambda x : -4*np.pi*x**2*n(x)

    # find minimum using golden section method
    x_max_Ndx = golden_section(minus_Ndx, 1e-4, 1)
    max_Ndx = -1*minus_Ndx(x_max_Ndx)
    with open(os.path.join(output_dir, '2a_x_max_Ndx.txt'), 'w') as f:
        f.write('{:0.5f}'.format(x_max_Ndx))
    with open(os.path.join(output_dir, '2a_max_Ndx.txt'), 'w') as f:
        f.write('{:0.2f}'.format(max_Ndx))
    
    #2b
    rng = RNG(seed+1)

    # sample p(x)dx using rejection sampling
    sample = rejection_sampling(pdx, pdx(x_max_Ndx), [1e-4,5], 10000, rng)

    x_sample = np.logspace(-4,np.log10(5),20)
    x_pdx = np.logspace(-4,np.log10(5),200)
    
    # plotting results
    plt.figure(figsize=(5,5))
    plt.hist(sample,x_sample, density=True, label='sample')
    plt.plot(x_pdx, pdx(x_pdx), label='analytical')
    plt.xscale('log')
    plt.xlabel(r'$r/r_\mathrm{vir}$')
    plt.ylabel('density')
    plt.axis(xmin=1e-4, xmax=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '2b_rejection_sampling.png'))
    plt.clf()

    #2c
    random_n = [next(rng) for _ in range(len(sample))]

    # Sort a randoml list of the same length as the sample and save the 
    # permutations. This yields a list containing randomly shuffled indexes.
    _, index_array = quicksort_with_indexing(
        random_n, list(range(len(random_n)))
    )
    # Use shuffling indices to select 100 random samples
    sample_selection = np.array(sample)[index_array[:100]].tolist()

    # Sort random sample using quicksort
    sorted_sample_selection = np.array(quicksort(sample_selection))

    # Calculate cummulative sum at every radius r
    bins = np.logspace(-4,np.log10(5),500)
    cumsum_sorted_sample_selection = []
    for i in bins:
        cumsum_sorted_sample_selection.append(
            len(sorted_sample_selection[sorted_sample_selection<i])
        )
    
    # plotting
    plt.figure(figsize=(5,5))
    plt.semilogx(bins, cumsum_sorted_sample_selection)
    plt.xlabel(r'$r/r_\mathrm{vir}$')
    plt.ylabel('number of galaxies with radius')
    plt.axis(
        xmin=1e-4,
        xmax=5, 
        ymin=0, 
        ymax=cumsum_sorted_sample_selection[-1]+0.5
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '2c_galaxies_within_radius.png'))
    plt.clf()

    #2d
    # Use plt.hist to obtain counts in bins
    histogram = plt.hist(sample, x_sample, density=False)

    # obtain largest radial bin
    radial_bin_edges = (
        histogram[1][np.argmax(histogram[0])],
        histogram[1][np.argmax(histogram[0])+1]
    )

    sample_selection = [
        i and j for i, j in zip(
            sample>radial_bin_edges[0],
            sample<radial_bin_edges[1]
        )
    ]
    sample_radial_bin =  np.array(sample)[sample_selection]

    # sort sample in radial bin to efficiently obtain percentiles
    sorted_sample_radial_bin = quicksort(sample_radial_bin.tolist())
    
    N = len(sample_radial_bin)
    median = sorted_sample_radial_bin[N>>1]
    percentile_16 = sorted_sample_radial_bin[int(0.16*N)]
    percentile_84 = sorted_sample_radial_bin[int(0.84*N)]
    
    # write output to file
    with open(os.path.join(output_dir, '2d_16th_percentile.txt'), 'w') as f:
        f.write('{:.2f}'.format(percentile_16))
    with open(os.path.join(output_dir, '2d_median.txt'), 'w') as f:
        f.write('{:.2f}'.format(median))
    with open(os.path.join(output_dir, '2d_84th_percentile.txt'), 'w') as f:
        f.write('{:.2f}'.format(percentile_84))
    
    # devide galaxies in halos containing 100 galaxies
    sample_100_bins = [sample[i:i+100] for i in range(0,len(sample), 100)]

    # Calculate number of 
    numbers_in_bin_in_halo = []
    for s in sample_100_bins:
        sample_selection = [
            i and j for i, j in zip(
                s>radial_bin_edges[0],
                s<radial_bin_edges[1]
            )
        ]
        numbers_in_bin_in_halo.append(sum(sample_selection))
    
    # calculate poissonian mean and 1-sigma
    poisson_mean = sum(numbers_in_bin_in_halo)/len(numbers_in_bin_in_halo)
    poisson_1sig = poisson_mean**0.5

    # plot bar plot with number of galaxies in each halo
    plt.figure(figsize=(5,5))
    plt.bar(
        list(range(len(numbers_in_bin_in_halo))), 
        numbers_in_bin_in_halo, 
        width=1, 
        zorder=0,
        alpha=0.8
    )
    plt.hlines(
        poisson_mean, 
        0, 
        100, 
        ls='-',
        color='C1',
        label=r'$\lambda$'
    )
    plt.hlines(
        poisson_mean - poisson_1sig, 
        0, 
        100, 
        ls='--', 
        color='C1', 
        label=r'$\lambda \pm 1\sigma$')
    plt.hlines(
        poisson_mean + poisson_1sig, 
        0, 
        100, 
        ls='--', 
        color='C1')
    plt.axis(
        xmin=0, 
        xmax=99, 
        ymin=poisson_mean - 3*poisson_1sig,
        ymax=poisson_mean + 3*poisson_1sig
    )
    plt.xlabel('Halo number')
    plt.ylabel('Number of satellite galaxies')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '2d_bar_plot_counts_per_halo.png'))
    plt.clf()

    # Plot density of number counts in bins
    plt.hist(
        numbers_in_bin_in_halo, 
        bins=9,
        density=True, 
        label='number of galaxies'
    )
    plt.plot(
        np.arange(20,60,1), 
        [poisson(x, poisson_mean) for x in np.arange(20,60,1)],
        label = r'$P_{36}(x)$'
    )
    plt.axis(
        xmin=poisson_mean - 2*poisson_1sig,
        xmax=poisson_mean + 2*poisson_1sig
    )
    plt.xlabel(r'number of galaxies ($x$)')
    plt.ylabel('number denisty')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '2d_number_denisty_counts_per_halo.png'))
    plt.clf()