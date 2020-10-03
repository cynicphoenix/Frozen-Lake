# Frozen Lake : Policy & Value Iteration

## Problem Statement
For this part of the assignment, we will make use of the Open AI Gym environment. You can read about the
Gym framework here: https://gym.openai.com/docs/<br /><br />
The gym framework bundles various environments to implement and compare various reinforcement
learning algorithms. We will use one of the bundled environments called the Frozen Lake environment:
https://gym.openai.com/envs/FrozenLake-v0/<br /><br />
You will implement the value iteration and policy iteration algorithms in your code for the Frozen Lake
environment from OpenAI Gym. A custom version of this environment in the starter code zipped file.<br />
- Read through vi_and_pi.py and implement policy_evaluation, policy_improvement
and policy_iteration. The stopping criteria (defined as ) is and
. The policy_iteration function should return the optimal value function and optimal
policy. Provide a 3-D plot for after each policy_evaluation step until convergence.
- Implement value_iteration in vi_and_pi.py. The stopping criteria is and . The
value_iteration function should return the optimal value function and optimal policy. Provide a 3-
D plot for for each iteration until convergence.
- Run both methods (value iteration and policy iteration) on the Deterministic-4x4-FrozenLake-v0 and
Stochastic-4x4-FrozenLake-v0 environments. In the second environment, the dynamics of the world are
stochastic. How does stochasticity affect the number of iterations required, and the resulting policy?
<br /><br />

## Execute
To run the code, goto code directory.<br />
In starter_code directory open cmd/bash/jupyter-notebook :<br />

(Note you need to install python3, matplotlib, numpy, openaigym)

- In cmd :
```
    Enter vi_and_pi.py
```
- In bash :
```
    python3 vi_and_pi.py
```
- In Conda/Jupyter Notebook:
```
    Open rl.ipynb & click on Run All Cells
```

<br /><br />
## Plots

#### Deterministic Policy Iteration

![](https://raw.githubusercontent.com/cynicphoenix/Frozen-Lake/main/plot/Detereministic_Policy_Iteration.png)

#### Deterministic Value Iteration
![alt text](https://raw.githubusercontent.com/cynicphoenix/Frozen-Lake/main/plot/Deterministic_Value_Iteration.png)

#### Stochastic Policy Iteration
![alt text](https://raw.githubusercontent.com/cynicphoenix/Frozen-Lake/main/plot/Stochastic_Policy_Iteration.png)

#### Stochastic Value Iteration
![alt text](https://raw.githubusercontent.com/cynicphoenix/Frozen-Lake/main/plot/Stochastic_Value_Iteration.png)













