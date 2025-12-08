# SmallGPT
Based on (again) the LLM series of Andrej Karpathy

# To-Do
- Add requirement.txt
- Scale the model
- Adding Dropout
- Create projection layer for MultiHead and FeedForward
- Change to RMSNorm
- Moving to device
- Work on other database?

# Some interesting results
Overfitting before Dropout/ Becfore scaling
```
step 0: train loss 4.1722, val loss 4.2064
step 500: train loss 3.2092, val loss 3.9087
step 1000: train loss 2.9198, val loss 4.0781
step 1500: train loss 2.7546, val loss 4.1874
step 2000: train loss 2.6368, val loss 4.3715
step 2500: train loss 2.5735, val loss 4.4953
step 3000: train loss 2.5600, val loss 4.6538
step 3500: train loss 2.5268, val loss 4.7259
step 4000: train loss 2.4982, val loss 4.7442
step 4500: train loss 2.4921, val loss 4.8018
step 4999: train loss 2.4628, val loss 4.8977
```