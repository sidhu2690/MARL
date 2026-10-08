# Literature Review: Adaptive Multi-Agent Coordination with Learned Agent Reliability

## 1. Consensus and Distributed Estimation

- **Distributed Consensus Kalman Filtering Over Time-Varying Graphs**  
  Uses consensus between agents to combine their local state estimates over a network.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S2405896323012867)

- **Distributed Kalman Filtering over Sensor Networks with Fading Measurements and Random Link Failures**  
  Studies distributed state estimation when communication links can fail and measurements can be unreliable.  
  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0016003222009188)

- **Distributed Kalman filtering for sensor networks**  
  Uses distributed average consensus for sensor fusion and provides an early approach to handling unreliable/outlying sensor information.  
  [Paper](https://www.researchgate.net/publication/226960066_Distributed_Kalman_Filtering_and_Sensor_Fusion_in_Sensor_Networks)

- **DIFFUSION LMS FOR CLUSTERED MULTITASK NETWORKS**  
  Uses adaptive combination weights to combine information from neighboring agents during distributed estimation.
  [Paper](https://arxiv.org/pdf/1310.8615)


## 2. Fault-Tolerant and Adversarial (Byzantine) Consensus


- **Resilient consensus in multi-agent systems with state constraints**  
  Several agents are trying to agree on a common state, while some agents may be malicious and send fake values, AND each normal agent has restrictions on where its state   is allowed to go
  [paper](https://www.sciencedirect.com/science/article/pii/S0005109820304878)


- **Resilient Consensus Control for Multi-Agent Systems: A Comparative Survey**  
  Surveys methods for achieving consensus when some agents are faulty or malicious, including filtering and resilient consensus strategies.  
  [Paper](https://www.mdpi.com/1424-8220/23/6/2904)

- **Resilient Consensus with Multi-hop Communication**  
  How can normal agents reach consensus when some agents are malicious, if information can travel through multiple hops?
  [Paper](https://arxiv.org/pdf/2201.03214)

- **Active Secure Neighbor Selection in Multi-Agent Systems with Byzantine Attacks - (Important to us)**  
  Detects and isolates Byzantine agents and then reconstructs a communication structure using the remaining reliable agents.  
  [Paper](https://arxiv.org/html/2511.19522)

- **Byzantine-Resilient Distributed Observers for LTI Systems**  
  It uses redundant neighbor estimates, removes the largest and smallest reported values to limit Byzantine influence, and achieves resilient state estimation without explicitly identifying malicious agents. 
  [Paper](https://arxiv.org/abs/1802.09651)

- **Resilient Asymptotic Consensus in Robust Networks**  
  It proposes W-MSR, which filters extreme neighbor values to achieve consensus despite malicious agents.
  [Paper](https://ieeexplore.ieee.org/document/6481629/)

- **Secure distributed estimation under Byzantine attack and manipulation attack**  
  It detects Byzantine and manipulation attacks using distance and angle based checks, isolates suspicious neighbors, and performs distributed estimation using the remaining reliable information.
  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762200389X)


## 3. Distributed State Estimation and Data Fusion Under Attack

- **Security-Aware Sensor Fusion with MATE: the Multi-Agent Trust Estimator (Important to us)**  
  Estimates agent trust from discrepancies between expected and reported observations and then uses the trust estimates during sensor fusion. This is one of the closest works to our problem.  
  [Paper](https://arxiv.org/html/2503.04954)

- **Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy**  
  It uses a Hidden Markov Model (HMM) to continuously estimate the trustworthiness of agents from the consistency of their reported observations with what they should have observed, then uses those trust estimates to weight their information during sensor fusion 
  [Paper](https://arxiv.org/html/2507.17875)



## 4. Trust, Reputation and Reliability Estimation

- **Metrics for Computing Trust in a Multi-Agent Environment (Important to us)**  
  Combines direct experience and information from other agents to estimate trust while reducing the influence of unreliable witnesses.  
  [Paper](https://arxiv.org/pdf/1305.2981)

- **A Context-Aware Reputation-Based Model of Trust for Open Multi-agent Environments**  
  It estimates trust by weighting past reputation according to how relevant its context is to the current situation, where the context weights are manually specified and context similarity is calculated using WordNet.
  [Paper](https://www.researchgate.net/publication/221442139_A_Context-Aware_Reputation-Based_Model_of_Trust_for_Open_Multi-agent_Environments)

- **ST-CATrust: multidimensional trust fusion evaluation via spatio-temporal graph neural network**  
  Uses a spatio-temporal graph network to learn trust from direct and recommendation-based trust histories. It is architecturally similar to our history-to-trust approach, but is designed for service-rating networks rather than physical observations.  
  [Paper](https://link.springer.com/article/10.1007/s44443-026-00702-w)


## 5. Truth Discovery and Peer Prediction: Reliability Without Ground Truth

- **Truth Discovery with Multiple Conflicting Information Providers on the Web (classical trust discovery method)**  
  It iteratively estimates source trustworthiness and fact confidence from each other, using consistency among conflicting claims to identify trustworthy sources and true facts.
  [paper](https://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Trust/kdd07_xyin.pdf)

- **A Survey on Truth Discovery**  
  It surveys truth-discovery methods that infer source reliability from conflicting reports and use these learned weights to identify the most trustworthy information.
  [Paper](https://www.kdd.org/exploration_files/Article1_17_2.pdf)

- **On the Discovery of Evolving Truth**  
  Dynamically updates both the estimated truth and source reliability as new information becomes available over time.  
  [Paper](https://pubmed.ncbi.nlm.nih.gov/26705502/)


- **Eliciting Informative Feedback: The Peer-Prediction Method**  
  It proposes Peer Prediction, a mechanism that encourages agents to report their information honestly by rewarding them based on how well they predict another agent’s report, without needing an objective ground truth. 
  [Paper](https://nmiller.web.illinois.edu/documents/research/elicit.pdf)



## 6. Robust Statistics and Byzantine-Resilient Aggregation

- **Byzantine-Tolerant Machine Learning**  
  It proposes Krum, a Byzantine-robust aggregation method that selects the gradient update closest to its other peers, preventing malicious workers from corrupting distributed SGD
  [paper](https://arxiv.org/pdf/1703.02757)

- **Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates**  
  Reduces the impact of extreme or malicious values by replacing ordinary averaging with robust statistical aggregation.
  [paper](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf)



## 7. Multi-Agent Spatial Sampling, Exploration, Surveillance and Environmental Monitoring

- **Multi-robot Informative and Adaptive Planning for Persistent Environmental Monitoring**  
  Plans robot trajectories to collect informative measurements of an environment, commonly using Gaussian processes and information-theoretic objectives.
  [paper](https://link.springer.com/chapter/10.1007/978-3-319-73008-0_20)

- **Multi-Robot Information Gathering for Spatiotemporal Environment Modelling**  
  Studies multi-robot information gathering for reconstructing environments that change over space and time.
  [Paper](https://www.ri.cmu.edu/app/uploads/2023/08/MSR-Thesis-Final-Siva-Kailas.pdf)

- **Probabilistically Resilient Multi-Robot Informative Path Planning**  
  Plans informative robot paths while considering the possibility that some robots provide malicious information.  
  [Paper](https://arxiv.org/abs/2206.11789)

- **Multi-Robot Coordination and Planning in Uncertain and Adversarial Environments**  
  It is a survey of multi-robot coordination methods for handling uncertainty, failures, and adversarial attacks, including resilient, risk-aware, and GNN-based approaches 
  [Paper](http://raaslab.org/pubs/zhou2021multi.pdf)

