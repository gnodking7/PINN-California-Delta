# PINN-California-Delta

Study on salinity modeling in California-Delta estuary by Physics-informed neural networks (PINN).

** All .ipynb files are best viewed in Colab. **

## Introduction

### Importance of Salinity Modeling

Salinity management is key to the region's water supply and ecology. 
Estuarine salinity brings great impact to changes in migration patterns, fish distribution, and the water quality of freshwater withdrawals.

In particular, California-Delta (Sacramento–San Joaquin Delta) provides significant social, economic, and environmental values including but not limited to

* a habitat for 750 species of animals and plants
* drinking and irrigation water to over 25 million people in California
* water to meet the water supply needs of the projects

### Literature Review

Relying on the governing physics, process-based models have been traditionally developed and utilized for salinity modeling in California-Delta estuary.
In particular, models based on the 1-dimensional Advection-Dispersion equation have found success in salinity estimation by utilizing the outflow-salinity relationship [1,2]. However, applying these models can be time-consuming. Data-driven machine learning models have been developed for fast simulation to overcome the issue.

Earliest such model was the multi-layer perceptron (MLP) [7], a fully-connected feedforward aritficial neural network (ANN), to simulate outflow-salinity relationship in California-Delta. Making improvements in efficiency and accuracy to MLP, more complex deep learning models such as LSTM, GRU, ResNet [5], and Res-LSTM, Res-GRU [6] have shown promising results in both simulating and forecasting at multiple locations in California-Delta. However, these models are purely data-driven and do not take into account the underlying physics of the outflow-salinity relationship.

### Current Work

The current study attempts to tackle the limitations of each approach highlighted above. Specifically, this study bridges the gap between the process-based models and the deep learning models.

* Utilizing neural network, this study achieves efficiency on par with the above mentioned deep learning models and is much less time-consuming than process-based models.
* Not only is this study data-driven like the other deep learning models, but it is also physics-informed, utilizing the 1-dimensional Advection-Dispersion equation that governs the outflow-salinity relationship.

## Methodology

### Study Area and Dataset

![Delta_flow_map](https://user-images.githubusercontent.com/91911643/210476517-27ae2190-0d2e-449a-a02f-f14574e3c3e0.png)

This study focuses on the region (figure above) in California-Delta estuary where the 1D Advection-dispersion equation is applicable. Three stations are of interests as they align close to a line: Martinez at stations #359, Port Chicago at stations #358, and Chipps Island at #362. Outflow, as measured in cubic feet per second (CFS), and salinity, represented as electrical conductivity (EC) in micro-Siemens/cm (S/cm), are monitored at these stations. Outflow corresponds to the input of the model, and salinity the output of the model.

Daily simulated Delta SimulationModel II (DSM2) outflow and salinity data during the period from 1 January 1991 to 31 December 2017 at these three stations are used.

### Input (outflow) Preprocessing 

Outflow in the past several months have lagged impacts in the current salinity conditions. Following the strategy of [5,6], for each daily outflow value, 118 antecedent daily outflow values are aggregated into 18 values to form a 18-dimensional outflow vector. This outflow vector consists of 8 most recent daily values (including the current day) and 10 non-overlapping 11-day averages of the prior 110 days.

Linear min-max normalization is applied to outflow and salinity data as well as the spatio-temporal domain of the 1-dimensional Advection-Dispersion equation to the range of [0, 1]. With such normalization, $x=0$ corresponds to Martinez station, $x=0.442$ to Port Chicago station, and $x=1$ to Chipps Island stations. The first time step after $t=0$ corresponds to 1 January 1991 and $t=1$ corresponds to 31 December 2017.

### Models (ANN and PINN)

This study highlights the advantage of PINN to a fully-connected feedforward aritficial neural network (denoted as ANN onward).

The ANN model consists of one input layer, two hidden layers, and one output layer. Input layer consits of 18 neurons, the number corresponding to the 18-dimensional outflow vectors. The model outputs a single value, an estimation to salinity. The general structure of the ANN model is illustrated in the below figure.

<img width="615" alt="ANN" src="https://user-images.githubusercontent.com/91911643/216418648-d1151653-0796-4a78-91d3-07fa230e4370.png">

ANN is trained by minimizing the mean squared error 
$$\sum_n\|\hat{S}_n-S_n\|^2$$ 
where $\hat{S}_n$ is the output of the model and $S_n$ is the target DSM2 salinity value.

The PINN model has the same neural network structure as the ANN model, except that it has two additional inputs $x_n$ for location and $t_n$ for time. The two additional inputs in PINN enforce the outflow-salinity relationship following the 1-dimensional Advection-Dispersion equation
$$A\frac{\partial S}{\partial t}-Q\frac{\partial S}{\partial x}=KA\frac{\partial^2 S}{\partial x^2}$$
where $A$ is a constant representing cross-sectional area and $K$ is a constant representing dispersion coefficient. The general structure of the PINN model is illustrated in the below figure.

<img width="585" alt="PINN_st" src="https://user-images.githubusercontent.com/91911643/216418690-c7346457-7fa8-4e55-b287-84a235dba3ff.png">

PINN is trained by minimizing the sum of the mean squared error and the PDE loss
$$\sum_n\|\hat{S}_n-S_n\|^2 +\sum_n\bigg\|A\frac{\partial \hat{S}}{\partial t}\Bigr|\_{(x_n,t_n)}-\vec{Q}\_{n,1}\frac{\partial \hat{S}}{\partial x}\Bigr|\_{(x_n,t_n)}-KA\frac{\partial^2 \hat{S}}{\partial x^2}\Bigr|\_{(x_n,t_n)}\bigg\|^2$$
where $\vec{Q}\_{n,1}$ is the first component of the outflow vector $\vec{Q}_n$.

The optimal hyperparameters (number of neurons and type of activation functions) for the PINN model were obtained by random search using KerasTuner; see the file PINN_ANN_hyperparameters.ipynb for details. The hyperparameters for the ANN model are set identically as the PINN model.

### Evaluation Metrics

Both the ANN model and the PINN model are trained with the Adam optimization algorithm. Four statstical evaluation metrics, consisting of the L2-relative-error, the Nash–Sutcliffe efficiency coefficient (NSE), the square of the correlation coefficient (r2), and bias, are employed to assess the model performance. L2-relative-error measures the accuracy of salinity estimation; NSE compares the predictive capacity of the model with the global mean of target salinity; r2 quantifies the strength of the linear relationship between salinity estimation and target salinity; and percent bias indicates whether the model over- or underestimates the salinity. For L2-relative-error and bias, a value close to 0 indicates a good performance, while for NSE and r2, a value close to 1 indicates a good performance.

### Implementation Details

Experiments are carried out using Python on a public platform, the Google Colaboratory. The ANN model is trained using TensorFlow [3] and the PINN model is trained using DeepXDE package [4].

## Results (Chronological Split)

We consider two chronological-split schemes consisting of around 77% training and 22% testing:

* Training: 1997-2017, Testing: 1991-1996
* Training: 1991-2011, Testing: 2012-2017

The time-series plots of the estimated salinity in comparison to the target salinity are presented at three considered locations Martinez, Port Chicago, and Chipps Island. For the sake of space, only the plots at Martinez are shown on this page. See the file PINN_vs_ANN_Chrono_split.ipynb for the time-series plots at all three locations. 

The time-series plot for ANN is shown first then the plot for PINN. The improvement in salinity estimation by the PINN model to the ANN model is clear. For two considered chronological-split schemes, PINN estimates salinity more accurately than ANN, in visualization and in the four metrics considered. Moreover, ANN suffers from saturation for values near zero while PINN shows no such behavior.

### Training: 1997-2017, Testing: 1991-1996

At Martinez (Training)

![image](https://user-images.githubusercontent.com/91911643/210492627-e6b48a20-221c-40fc-984f-f7f60892946a.png)

![image](https://user-images.githubusercontent.com/91911643/210492801-389be6c3-0b57-4305-94b0-307b6dd41784.png)

At Martinez (Testing)

![image](https://user-images.githubusercontent.com/91911643/210493319-551868be-d1ac-4ad4-9c00-7fe34ffdf7d6.png)

![image](https://user-images.githubusercontent.com/91911643/210493355-1751104a-e378-4387-a35e-6efa9fb1cc3e.png)

### Training: 1991-2011, Testing: 2012-2017

At Martinez (Training)

![image](https://user-images.githubusercontent.com/91911643/210493430-93bf2ecd-382b-4b14-a451-008aae3954db.png)

![image](https://user-images.githubusercontent.com/91911643/210493463-06bf88e9-ed7e-4c93-86aa-4cff82e6f499.png)

At Martinez (Testing)

![image](https://user-images.githubusercontent.com/91911643/210493492-df1e7507-5aea-4d15-a440-75adfd934f0b.png)

![image](https://user-images.githubusercontent.com/91911643/210493535-b5148bd3-2c86-4c65-952b-93d9902b3fa0.png)

## Results (5-fold)

We conduct 5-fold Cross-Validation on 25 years of DSM2-simulated data from 1991-2015. For each fold, optimal hyperparameters were pre-computed separately for ANN and PINN.

The following scatter plots show evalutation results of PINN vs. ANN for each fold and each location. Smaller Bias indicates better performance, i.e., PINN performed better than ANN for dots above the dotted line; larger NSE indicates better performance, i.e., PINN performed better than ANN for dots belowe the dotted line.

<img width="400" alt="BiasM+P+C_test (Best)" src="https://user-images.githubusercontent.com/91911643/216421972-44377843-dedd-4575-b543-cf58460ac5b5.png"> <img width="400" alt="NSEM+P+C_test (Best)" src="https://user-images.githubusercontent.com/91911643/216421987-7af53867-f4a4-44c2-b77a-6ce927d7d5ac.png">

Below are a couple of time-series plots illustrating the improvement of using PINN over ANN. For all time-series plots, see the file 5_fold_Results_&_Plots.ipynb.

![fourth_Martinez_test (Best)](https://user-images.githubusercontent.com/91911643/216424571-087c0fbe-4bde-477e-8ef2-fd5eba17aa82.png)

![fifth_Port Chicago_test (Best)](https://user-images.githubusercontent.com/91911643/216424581-e43bac3a-3054-4f37-9106-33f2371464de.png)


# References

[1] Denton, R.A. Accounting for Antecedent Conditions in Seawater Intrusion Modeling—Applications for the San Francisco Bay-Delta. In Hydraulic Engineering; ASCE: Reston, FL, USA, 1993; pp. 448–453.

[2] Hutton, P.H.; Rath, J.S.; Chen, L.; Ungs, M.J.; Roy, S.B. Nine decades of salinity observations in the San Francisco Bay and Delta: Modeling and trend evaluations. J. Water Resour. Plan. Manag. 2016, 142, 04015069.

[3] Joshua V. Dillon, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, and Rif A. Saurous. Tensorflow distributions. arXiv preprint arXiv:1711.10604, 2017.

[4] Lu Lu, Xuhui Meng, Zhiping Mao, and George Em Karniadakis. Deepxde: A deep learning library for solving differential equations. SIAM Review, 63(1):208–228, 2021.

[5] Qi, S.; He, M.; Bai, Z.; Ding, Z.; Sandhu, P.; Zhou, Y.; Namadi, P.; Tom, B.; Hoang, R.; Anderson, J. Multi-Location Emulation of a Process-Based Salinity Model Using Machine Learning. Water 2022, 14, 2030.

[6] Siyu Qi, Minxue He, Zhaojun Bai, Zhi Ding, Prabhjot Sandhu, Francis Chung, Peyman Namadi, Yu Zhou, Raymond Hoang, Bradley Tom, et al. Novel salinity modeling using deep learning for the Sacramento–San Joaquin delta of california. Water, 14(22):3628, 2022.

[7] Sandhu, N.; Finch, R. Application of artificial neural networks to the Sacramento-San Joaquin Delta. In Estuarine and Coastal Modeling; ASCE: Reston, FL, USA, 1995; pp. 490–504.
