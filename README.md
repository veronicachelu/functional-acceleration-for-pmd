# Functional Acceleration for Policy Mirror Descent (PMD)
An analysis of the effect of functional acceleration techniques on the policy optimization dynamics of PMD.

## Requirements

See [requirements.txt](./requirements.txt) for a list of required Python packages. These can be installed using `pip install -r requirements.txt`.

## Code
Python source files are organized as follows.

* Algorithm `./src/landscape.py`
* Algorithm utils `./src/utils.py`
* Policy data structure `./src/policy.py`
* Environments:
  * `./src/tiny_mdps.py` (2state/2action MDPs)
  * `./src/discounted_mdp.py` (randomly constructed MDPs)
* Experiment runner `./src/runner.py` 
* Logger `./src/logger.py`
* Ploter `./src/plot_tiny_mdps.py` and `./src/plot_random_mdps.py`

## Numerical illustrations (Figures/Plots)
Run these to plot. Plots are saved in `./figures` in the root of the project.
IPython notebooks are organized as follows.

* `numerical_illustrations` - plots from the main body of the paper

* `supplementary_results` - plots from the Appendix
