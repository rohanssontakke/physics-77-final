This repository holds the code that we used to sample, train, and visualize the network we used. This repository was adapted from a jupyter notebook, one file that was split into the files below for the purposes of readability.

Navigating the repository is as follows:

imports.py: This file has global imports used across nearly every function in our repository. This list is non-exhaustive: some functions have local imports in case they are not ubiquitous.

shared_utils.py: This file holds functions that are used multiple times throughout the repository. These are classic linear algebra operations, amongst other things.

feed_forward_VMC.py: This file holds our feed-forward variational monte carlo approach. This approach was not actually used but it was used for testing and some 1D toy problems so we felt we should include it.

rbm_gibbs.py: This holds our code for our main sampling step. The restricted-boltzmann machine and the gibbs sampling, amongst other helper methods that were used in the actual sampling in this project.

1d_visualizations.py: This holds our code for generating all visualizations in 1D. There are some extra functions we included as we tested other visualizations than the ones we included in the report.

2d_geometry.py: This holds code that helps translate our functions from 1D to 2D. These are relatively light as the sampling step from 1D->2D just involves an increase in the dimensionality of the array modeling the spin system.

pytorch_rbm.py: This holds code for the rbm using MLX, cuda, and pytorch, which is what we used for the 2D visualization. Running 2D visualizations requires this being the most recent RBM.

2d_visualizations.py: This holds our code for generating all visualizations in 2D.

Major dependencies are MLX and CuPy(for 2d). For 2d examples, please expect around 100gb VRAM usage for $L\leq 5$, and around 500gb VRAM usage for $L=6$. This usage's main contribute is from the exact diagonalization computed for reference data. 


Functions are well labelled and their use-cases are commented by us.

