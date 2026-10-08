# Literature Review: Adaptive Multi-Agent Coordination with Learned Agent Reliability

## 1. Consensus and Distributed Estimation

- **Olfati-Saber — Consensus-based Kalman Filtering**  
  Uses consensus between agents to combine their local state estimates over a network.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S2405896323012867)

- **Distributed Kalman Filtering over Sensor Networks with Fading Measurements and Random Link Failures**  
  Studies distributed state estimation when communication links can fail and measurements can be unreliable.  
  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0016003222009188)

- **Xiao, Boyd & Lall (2005) — A Scheme for Robust Distributed Sensor Fusion Based on Average Consensus**  
  Uses distributed average consensus for sensor fusion and provides an early approach to handling unreliable/outlying sensor information.  
  [Paper](https://www.researchgate.net/publication/226960066_Distributed_Kalman_Filtering_and_Sensor_Fusion_in_Sensor_Networks)

- **Diffusion LMS — Sayed**  
  Uses adaptive combination weights to combine information from neighboring agents during distributed estimation. These weights are closely related to the idea of trust weights in our approach.


## 2. Fault-Tolerant and Adversarial (Byzantine) Consensus

- **LeBlanc et al. (2013) — W-MSR / MSR**  
  Removes extreme values reported by neighboring agents before performing consensus, allowing the system to tolerate a limited number of malicious agents.

- **Wang et al. (2023) — Resilient Consensus Control for Multi-Agent Systems: A Comparative Survey**  
  Surveys methods for achieving consensus when some agents are faulty or malicious, including filtering and resilient consensus strategies.  
  [Paper](https://www.mdpi.com/1424-8220/23/6/2904)

- **Resilient Consensus with Multi-hop Communication**  
  Extends resilient consensus to multi-hop communication, allowing agents to use information received through multiple communication paths.  
  [Paper](https://arxiv.org/pdf/2201.03214)

- **Active Secure Neighbor Selection (2025)**  
  Detects and isolates Byzantine agents and then reconstructs a communication structure using the remaining reliable agents.  
  [Paper](https://arxiv.org/html/2511.19522)

- **Mitra & Sundaram (2018) — Byzantine-Resilient Distributed Observers for LTI Systems**  
  Studies distributed state estimation when some agents can behave arbitrarily or maliciously and establishes conditions for resilient estimation.  
  [Paper](https://arxiv.org/abs/1802.09651)

- **An & Sundaram (2021) — Byzantine-Resilient Distributed State Estimation: A Min-Switching Approach**  
  Develops a resilient distributed state-estimation method that limits the influence of potentially malicious measurements.  
  [Paper](https://www.researchgate.net/publication/351332929_Byzantine-resilient_distributed_state_estimation_A_min-switching_approach)

- **Secure Distributed Estimation Under Byzantine and Manipulation Attacks**  
  Uses detectors to identify manipulated or Byzantine information before performing distributed estimation.  
  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762200389X)


## 3. Distributed State Estimation and Data Fusion Under Attack

- **Security-Aware Sensor Fusion with MATE (2025)**  
  Estimates agent trust from discrepancies between expected and reported observations and then uses the trust estimates during sensor fusion. This is one of the closest works to our problem.  
  [Paper](https://arxiv.org/html/2503.04954)

- **Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy (2025)**  
  Estimates trust from pairwise comparisons between agents' situational-awareness reports and uses the resulting trust to improve sensor fusion.  
  [Paper](https://arxiv.org/html/2507.17875)

**Main connection:** These works are particularly relevant because they use **cross-agent disagreement as evidence for estimating reliability**. Our approach differs by learning the mapping from cross-check history to trust rather than relying on hand-designed trust rules.


## 4. Trust, Reputation and Reliability Estimation

### 4.1 Agent Trust and Reputation Models

- **Metrics for Computing Trust in a Multi-Agent Environment**  
  Combines direct experience and information from other agents to estimate trust while reducing the influence of unreliable witnesses.  
  [Paper](https://arxiv.org/pdf/1305.2981)

- **FIRE Trust Model**  
  Uses reputation and statistical filtering to estimate the reliability of agents and recommendations in multi-agent environments.  
  [Paper](https://www.researchgate.net/publication/221442139_A_Context-Aware_Reputation-Based_Model_of_Trust_for_Open_Multi-agent_Environments)

### 4.2 Graph-Based and Learned Trust

- **ST-CATrust (2026)**  
  Uses a spatio-temporal graph network to learn trust from direct and recommendation-based trust histories. It is architecturally similar to our history-to-trust approach, but is designed for service-rating networks rather than physical observations.  
  [Paper](https://link.springer.com/article/10.1007/s44443-026-00702-w)


## 5. Truth Discovery and Peer Prediction: Reliability Without Ground Truth

- **Yin, Han & Yu (2008) — Truth Discovery**  
  Jointly estimates the unknown truth and the reliability of different information sources without requiring ground-truth labels.

- **Li et al. (2014) — CRH**  
  Estimates the reliability of different sources while simultaneously estimating the most likely truth from conflicting reports.  
  [Paper](https://www.kdd.org/exploration_files/Article1_17_2.pdf)

- **Zhang et al. (2015) — Modeling Truth Existence in Truth Discovery**  
  Considers situations where the true answer may not be present in any of the available sources, making it relevant to ground-truth-free estimation.  
  [Paper](https://dl.acm.org/doi/10.1145/2783258.2783339)

- **On the Discovery of Evolving Truth**  
  Dynamically updates both the estimated truth and source reliability as new information becomes available over time.  
  [Paper](https://pubmed.ncbi.nlm.nih.gov/26705502/)

- **Influence-Aware Truth Discovery**  
  Accounts for dependencies between sources, which is relevant when multiple agents may share correlated information or errors.  
  [Paper](https://dl.acm.org/doi/10.1145/2983323.2983785)

- **Miller, Resnick & Zeckhauser (2005) — Peer Prediction**  
  Provides a way to evaluate the quality of agents' reports by comparing them with other agents' reports without requiring ground truth.  
  [Paper](https://www.researchgate.net/publication/220535244_Eliciting_Informative_Feedback_The_Peer-Prediction_Method)

**Main connection:** Truth discovery and peer prediction are important foundations for our **ground-truth-unavailable** setting because they show how source reliability can be inferred from agreement or disagreement between agents.


## 6. Robust Statistics and Byzantine-Resilient Aggregation

- **Krum (Blanchard et al., 2017)**  
  Selects an update that is closest to the other updates, reducing the influence of malicious participants during aggregation.

- **Coordinate-wise Median and Trimmed Mean (Yin et al., 2018)**  
  Reduce the influence of extreme values by using robust statistics instead of ordinary averaging.

- **Bulyan**  
  Combines multiple robust aggregation ideas to improve resistance against Byzantine participants.

- **Local Model Poisoning Attacks to Byzantine-Robust Federated Learning**  
  Studies how attackers can manipulate their updates to bypass different Byzantine-robust aggregation rules.  
  [Paper](https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf)

- **FLTrust (Cao et al., 2021)**  
  Computes a trust score for each client by comparing its update with a trusted server reference and then uses this score to weight aggregation.  
  [Paper](https://www.emergentmind.com/papers/2012.13995)

- **Do We Really Need to Design New Byzantine-robust Aggregation Rules? (2025)**  
  Studies the effectiveness of existing robust aggregation rules and argues that improving established methods such as trimmed mean and median can already provide strong robustness.  
  [Paper](https://www.ndss-symposium.org/wp-content/uploads/2025-1796-paper.pdf)

- **Enhancing FL Robustness Through Clustering Non-IID Features**  
  Studies how heterogeneous/non-IID data can reduce the effectiveness of Byzantine-robust aggregation methods.  
  [Paper](https://openaccess.thecvf.com/content/ACCV2022W/AMLAVS/papers/Li_Enhancing_Federated_Learning_Robustness_Through_clustering_Non-IID_Features_ACCVW_2022_paper.pdf)

**Main connection:** These methods provide strong baselines for suppressing unreliable or malicious reports. However, they generally use fixed aggregation rules rather than learning a continuous trust value from cross-agent observation histories.


## 7. Multi-Agent Spatial Sampling, Exploration, Surveillance and Environmental Monitoring

- **Multi-robot Informative and Adaptive Planning for Persistent Environmental Monitoring**  
  Plans robot trajectories to collect informative measurements of an environment, commonly using Gaussian processes and information-theoretic objectives.

- **Multi-Robot Information Gathering for Spatiotemporal Environment Modelling**  
  Studies multi-robot information gathering for reconstructing an environment that changes over space and time.  
  [Paper](https://www.ri.cmu.edu/app/uploads/2023/08/MSR-Thesis-Final-Siva-Kailas.pdf)

- **Probabilistically Resilient Multi-Robot Informative Path Planning (2022)**  
  Plans informative robot paths while considering the possibility that some robots provide malicious information.  
  [Paper](https://www.researchgate.net/publication/361502756_Probabilistically_Resilient_Multi-Robot_Informative_Path_Planning)

- **Multi-Robot Coordination and Planning in Uncertain and Adversarial Environments**  
  Surveys uncertainty and adversarial issues in multi-robot coordination, including communication failures, robot failures, and malicious information.  
  [Paper](http://raaslab.org/pubs/zhou2021multi.pdf)

**Main connection:** These works provide the spatial exploration and information-gathering side of our problem. However, most resilient planning methods assume a fixed attack model or predefined robustness mechanism rather than learning agent reliability and coupling it directly to downstream sampling or reconstruction.

