# Data Scientist

#### Technical Skills: Python, SQL, GoLang, PyTorch, TensorFlow. 
### Machine Learning Skills: Deep Learning, Reinforcement Learning, Time-series Analysis, Computer Vision, Tree-based Algorithms, Federated Learning, and Quantization

## Education
- Ph.D., Electrical and Computer Engineering | University of Western Ontario (_In Progress_)
- MESc., Electrical and Computer Engineering | University of Western Ontario (_August 2020_)
- BSc., Computer Science | American University of Beirut (_May 2017_)						       		             		

## Research Experience
**Research Assistant @ University of Western Ontario (_Sept 2018 - Present_)**
- Improved the performance of the hyper-parameter optimization of neural networks by 6.8% using a transformer-based Reinforcement Learning approach producing models that outperformed state-of-the-art processes on public datasets while gaining a 30% speedup in model generation. 
- Addressed availability and heterogeneity concerns in Federated Learning settings by leveraging Correlational Neural Networks to successfully fill the training gap of absent models, with 15% improvement in predictive quality.  
- Investigated the predictive quality of different Machine Learning models and time-series feature engineering methods for occupancy proxy prediction for aiding the Heating, Ventilation, and Air Conditioning (HVAC) systems using energy consumption while identifying air leaks as the reason for unexplained variances of wrong predictions.
- Implemented a hierarchical framework to accurately predict CO2 variations as occupancy proxy estimators that outperform state-of-the-art methods by 32% and facilitate the transferability of this method to other spatial settings.
- Integrated explainability into deep learning models under concept drift conditions by identifying 86% of drifting features via Autoencoders in an occupancy proxy prediction use case.

## Projects
### Efficient Transformer-based Hyper-parameter Optimization for Resource-constrained IoT Environments
[Publication](https://arxiv.org/abs/2403.12237)
[GitHub](https://github.com/ibrahimshaer/TRL-HPO)

Developed a novel hyper-parameter optimization strategy for **Convolutional Neural Networks** (CNN) using **transformer** architecture and **actor-critic reinforcement learning (RL)** approach, named _TRL-HPO_, implemented using **PyTorch**. This architecture harnesses the parallelization of multi-headed attention to facilitate the training process and the progressive generation of layers method. _TRL-HPO_ produced CNN models that outperformed state-of-the-art by 6.8% in terms of classification accuracy produced in the same time frame. The _TRL-HPO_ adds clarity to the model generation procedure by identifying the stacking of fully connected layers as the main reason for degradation in model performance. 


![Architecture](/assets/img/architecture_v3.png)

### CorrFL: Correlation-based Neural Network Architecture for Unavailability Concerns in a Heterogeneous IoT Environment 
[Publication](https://arxiv.org/abs/2307.12149)
[GitHub](https://github.com/Western-OC2-Lab/CorrFL)

This paper addresses the limitations of the Federated Learning (FL) environment related to the heterogeneity of participants' models and the unvailability constrained, coined as ``Oblique Federated Learning". We address this problem using Correlational Federated Learning (CorrFL), inspired by the multi-view representational learning field. For each available model weight, an **Autoencoder** is implemented to project the heterogeneous weights into a common representation, while maximizing the correlation between model latent representations when a model is absent. The validity of CorrFL is evaluated on the CO2 prediction use case, whereby a model becomes unavailable upon drastic changes in underlying conditions. Under this scenario, the models using the model weights compensated by CorrFL outperform models with outdated weights by a minimum of 15% in terms of predictive quality. 

![Architecture](/assets/img/methodology_v3.png)


### Hierarchical Modelling for CO2 Variation Prediction for HVAC System Operation 
[Publication](https://www.mdpi.com/1999-4893/16/5/256)
[GitHub](https://github.com/Western-OC2-Lab/hierarchical-CO2)

This paper devises a hierarchical model to accurately predict CO2 variations, acting as occupancy proxy estimators and facilitate the models' transferability to different office spaces. These predictions aid the HVAC systems in their decision-making process, reducing their carbon footprint. In the first step of this method, the collected environmental features of CO2, pressure, humidity, temperature, and Passive InfraRed (PIR) count are transformed into images using Gramian Angular Field (GAF). The second phase combines the predictions of the first phase with the differences of the environmental features. The combination of 2D-CNN of the first phase and decision trees produced the best result of Mean Absolute Error (MAE) = 27.74 for a 20-minute prediction window compared to the state-of-the-art models. In a similar manner, the fine-tuned model applied to different spatial settings outperformed the state-of-the-art models with MAE = 33.6 for a 20-minute prediction window. These great results showcase the utility of the developed framework in CO2 variation prediction in different spatial settings.  

![Architecture](/assets/img/occupancy_methodology_r1.png)
<!--- ![Bike Study](/assets/img/bike_study.jpeg) -->


## Publications
1.	I. Shaer and A. Shami, “Thwarting Cybersecurity Attacks with Explainable Concept Drift” (to appear after 2024 International Wireless Communications and Mobile Computing (IWCMC)). (https://arxiv.org/pdf/2403.13023.pdf)
2.	I. Shaer, S. Nikan, and A. Shami, “Efficient Transformer-based Hyper-parameter Optimization for Resource-constrained IoT Environments” (to appear in IEEE Internet of Things Magazine). (https://arxiv.org/pdf/2403.12237.pdf)
3.	I. Shaer, A. Haque, and A. Shami, “Availability-aware multi-component V2X application placement,” Vehicular Communications, vol. 43, p. 100653, Oct. 2023. doi:10.1016/j.vehcom.2023.100653. (I.F.: 6.7).
4.	I. Shaer and A. Shami, “Data-driven methods for the reduction of energy consumption in warehouses: Use-case driven analysis,” Internet of Things, vol. 23, p. 100882, Oct. 2023. doi:10.1016/j.iot.2023.100882. (I.F.: 5.9).
5.	I. Shaer and A. Shami, “Hierarchical modelling for CO2 variation prediction for HVAC System Operation,” Algorithms, vol. 16, no. 5, p. 256, May 2023. doi:10.3390/a16050256. (I.F.: 2.3).
6.	I. Shaer and A. Shami, “CorrFL: Correlation-based neural network architecture for unavailability concerns in a heterogeneous IoT environment,” IEEE Transactions on Network and Service Management, vol. 20, no. 7, pp. 1543–1557, Jun. 2023. doi:10.1109/tnsm.2023.3278937. (I.F.: 5.3).
8.	I. Shaer, G. Sidebottom, A. Haque, and A. Shami, “Efficient execution plan for egress traffic engineering,” Computer Networks, vol. 190, p. 107938, May 2021. doi:10.1016/j.comnet.2021.107938. (I.F.: 5.6). 
9.	I. Shaer and A. Shami, "Sound Event Classification in an Industrial Environment: Pipe Leakage Detection Use Case," 2022 International Wireless Communications and Mobile Computing (IWCMC), Dubrovnik, Croatia, 2022, pp. 1212-1217, doi: 10.1109/IWCMC55113.2022.9824540.
10.	I. Shaer, A. Haque and A. Shami, "Multi-Component V2X Applications Placement in Edge Computing Environment," ICC 2020 - 2020 IEEE International Conference on Communications (ICC), Dublin, Ireland, 2020, pp. 1-6, doi: 10.1109/ICC40277.2020.9148960. 
11.	D. M. Manias, I. Shaer, J. Naoum-Sawaya, and A. Shami, “Robust and reliable SFC placement in resource-constrained multi-tenant MEC-Enabled Networks,” IEEE Transactions on Network and Service Management, vol. 21, no. 1, pp. 187–199, Feb. 2024. doi:10.1109/tnsm.2023.3293027. (I.F.: 5.3). 
12.	D. M. Manias, I. Shaer, L. Yang and A. Shami, "Concept Drift Detection in Federated Networked Systems," 2021 IEEE Global Communications Conference (GLOBECOM), Madrid, Spain, 2021, pp. 1-6, doi: 10.1109/GLOBECOM46510.2021.9685083.


