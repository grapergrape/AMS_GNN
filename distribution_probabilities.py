
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import geom

def uniform_gamma(x):
    """
    This function assumes that the shortest path is equally likely to go through any edge. 
    """
    return 1.0

def exponential_gamma(x, y):
    """
    This function assumes that the likelihood of an edge being in the shortest path decreases exponentially as the edge weight increases for both vertices.
    Given that veins become more curved in the presence of an aneurysm and assuming that this increased curvature increases the edge weights,
    we can expect that paths along heavily curved veins will have larger costs. 

    Best choice for our problem.
    """
    scale_parameter_x = 1.0  # Adjust this parameter based on your data for vertex x
    scale_parameter_y = 1.0  # Adjust this parameter based on your data for vertex y
    return expon.pdf(x, scale=scale_parameter_x) * expon.pdf(y, scale=scale_parameter_y)



def gaussian_gamma(x):
    """
    This function assumes that the edge weights of the shortest path follow a Normal distribution.
    """
    mean = 0.0  # Adjust this parameter based on your data
    std_dev = 1.0  # Adjust this parameter based on your data
    return norm.pdf(x, loc=mean, scale=std_dev)
def geometric_gamma(x):
    """
    This function assumes that the likelihood of an edge being in the shortest path decreases geometrically as the edge weight increases. 
    Given that veins become more curved in the presence of an aneurysm and assuming that this increased curvature increases the edge weights,
    we can expect that paths along heavily curved veins will have larger costs. But only takes descrete weights as in integer. 
    """
    p = 0.2  # Adjust this probability parameter based on your data
    return geom.pmf(x, p)
