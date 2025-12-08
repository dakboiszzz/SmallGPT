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

## First version
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
## Second version
Still overfititing after adding Dropout and Projection layers (I added some samples of our model :)

Also before scaling
```
step 0: train loss 4.2299, val loss 4.2563
step 500: train loss 3.3346, val loss 4.0576
...
step 4999: train loss 2.5199, val loss 4.7842
.' llndelach as plen. Cbvuralr make she toowe f .
LR
A: To ennd ivy wass bourd t:
An sinsotbucunasr me,
Bye, wuld ning; thyhhisir vitind le, hord, trirs thoRcy hadoravet hind bad
hop o mey ntovonbtiet hid w Mol win me he E U ditass bil teyichinKo I
AL:
ANWUSEu Co,
O zadeleesr d lad worghre dedhI
She cEEOnfiwe y, nourl;
Tuegorth jorny nhi
Tef lovis pcauath dis, macoy cukiw
; peme thy tee hird,
Go y lwUThe lonafan o y northd ciw sheso de MWaicm
Lowathoverl, odme sas cheed morr moyes LI E maadll I g
```