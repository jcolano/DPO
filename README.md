# DPO
Direct Preference Optimization Implementation

![image](https://github.com/jcolano/DPO/assets/1131538/1c4683af-7c65-4207-bd20-e2fc580a9542)

1. Loss function of policy model with respect to reference model.
2. Expectation of the dataset with ‘x’ samples of ‘yw’ winner (chosen) and ‘yl’ loser (rejected) outputs.
3. The logarithmic of the sigmoid applied to the argument. In torch: ‘F.logsigmoid’
  This will scale the result between 0 and 1, providing a probabilistic interpretation.
4. Beta: Hyperparameter that weights the importance of the deviation between the policy model and the reference model.
5. Log Probability (in torch: log_softmax) ratio of the policy model’s probability of choosing the same ‘yw’ given the input ‘x’, divided by the reference model’s probability of choosing the same ‘yw’ given the same input ‘x’.
These rations indicate how much more or less likely the policy model is to choose a particular action compared to the reference model.


The formula calculates the difference between the log probability rations for the chosen and rejected actions, scaled by Beta, and then applies the sigmoid function to this difference.

The expectation ‘E’ of this value across the dataset provides a measure of how well the policy model aligns with the human preferences as exhibited by the chosen and rejected outputs.

The loss function ‘L’ is then minimized during training to adjust the parameters of the policy model so that it more closely aligns with the human-labeled preferences. By doing this, the policy model’s behavior should become more similar to the desired behavior as indicated by the reference model and the human choices in the dataset.  The use of the sigmoid function ensures that the loss is bounded and that the optimization process is stable.
