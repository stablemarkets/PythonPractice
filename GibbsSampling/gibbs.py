### import required modules
from scipy.stats import norm
from scipy.stats import gamma
from matplotlib import pyplot as plt
import numpy as np

### simulate data
### simulate from Normal mean 100, st. dev 30.
y = norm.rvs(loc=100, scale=30, size=10000)

### conditional posterior of mu
def cond_post_mu(y, prior_mu_mean, prior_mu_phi, prec):
	phi = 1/prec
	n = len(y)
	sum_y = sum(y)

	# compute conditional posterior parameters
	post_mean = (1/(1/prior_mu_phi + n/phi))*(prior_mu_mean/prior_mu_phi + sum_y/phi)
	post_phi = (1/prior_mu_phi + n/phi)**(-1)

	# draw from conditional posterior of mu
	draw  = norm.rvs(loc = post_mean, scale = np.sqrt(post_phi),	size=1 )

	return(draw)

### conditional posterior of phi
def cond_post_prec(y, prior_prec_a, prior_prec_b, mu):
	n = len(y)
	sse_y = sum((y-mu)**2)

	post_a = prior_prec_a + n/2
	post_b = prior_prec_b + sse_y/2

	draw = gamma.rvs(a = post_a, scale=1/post_b, size=1) 

	return(draw)

## gibbs sample...input prior hyperparameters and initial value and 
## number of Gibbs iterations to run.
def gibbs_mcmc(iter=1000, prec_init=.5, 
	prior_mu_mean=0, prior_mu_phi=100,
	prior_prec_a=10, prior_prec_b=200):

	# create shells for storing draws
	mu_shell = [0]*iter
	prec_shell = [prec_init]*iter

	for i in range(1, iter):
		
		# sample from conditional posterior of mu, conditional on precision
		draw_mu = cond_post_mu(y, prior_mu_mean, prior_mu_phi, prec_shell[i-1]);
		mu_shell[i] = draw_mu

		# sample from conditional posterior of precision, conditional on mu.
		draw_prec = cond_post_prec(y, prior_prec_a, prior_prec_b, draw_mu);
		prec_shell[i] = draw_prec


	return prec_shell, mu_shell

prec_shell, mu_shell = gibbs_mcmc(iter=1000, prec_init=.000001)

## plot MCMC chains.

plt.plot(1/(np.sqrt(prec_shell[100:1000])))
plt.show()

plt.plot(mu_shell[100:1000])
plt.show()
