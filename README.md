# HBPR Based QoS Routing in LEO Satellites

## Authors
- Sumanth Guptha M
- Sasank

## Mentor
- Rahul Agrawal

---

## Introduction

This project focuses on routing and Quality of Service (QoS) management in Low Earth Orbit (LEO) satellite communication networks. LEO satellites provide low latency and fast communication, making them ideal for global connectivity. The communication involves user terminals sending requests to satellites, which relay signals to ground stations and back, with seamless handover between satellites to maintain continuous communication.

Routing in LEO satellite networks is complex due to high mobility, dynamic topology, and large constellations. Efficient routing must address load balancing, fault tolerance, security, and QoS requirements such as latency, bandwidth, reliability, delay, and jitter.

---

## Problem Statement

Current routing, load balancing, and QoS protocols in LEO satellite networks often operate in isolation, leading to inefficiencies such as congestion and poor QoS. Existing routing algorithms may cause congestion due to static path selection, and load balancing techniques often neglect QoS constraints like latency and jitter. There is a need for a unified algorithm that integrates QoS, routing, and load balancing to optimize network performance dynamically.

---

## Approach

- Define QoS metrics prioritizing latency, jitter, packet loss, and throughput.
- Introduce a QoS-weighted backlog metric to refine Hops-Based Back-Pressure Routing (HB-BP) decision-making.
- Modify queue backlog calculations to include packet delay sensitivity.
- Adapt path selection to prioritize routes with lower congestion and better QoS scores.
- Implement continuous monitoring and feedback loops to optimize routing decisions in real-time.
- Validate the approach through simulations using NS-3 or OMNeT++ with real-world satellite mobility models.

---

## Challenges

- Simulation inconsistencies across different platforms (OMNeT++, NS2, NS3, OPNET) complicate validation.
- Balancing load evenly while meeting QoS requirements can lead to conflicting routing decisions.
- Maintaining real-time QoS constraints in a highly dynamic satellite network environment.

---

## Timeline

- Weeks 1-3: Preparing concrete solution
- Weeks 4-7: Simulations and testing
- Week 8: Term paper preparation

---

## References

### Routing
1. Westphal, C., Han, L., & Li, R. (2023). LEO satellite networking relaunched: Survey and current research challenges. arXiv preprint arXiv:2310.07646.
2. Xiaogang, Q. I., Jiulong, M., Dan, W., Lifang, L., & Shaolin, H. (2016). A survey of routing techniques for satellite networks. Journal of communications and information networks, 1(4), 66-85.
3. Taleb, T., Mashimo, D., Jamalipour, A., Kato, N., & Nemoto, Y. (2008). Explicit load balancing technique for NGEO satellite IP networks with on-board processing capabilities. IEEE/ACM transactions on Networking, 17(1), 281-293.
4. Han, C., Xiong, W., & Yu, R. (2023). Load-balancing routing for leo satellite network with distributed hops-based back-pressure strategy. Sensors, 23(24), 9789.
5. Deng, X., Chang, L., Zeng, S., Cai, L., & Pan, J. (2022). Distance-based back-pressure routing for load-balancing LEO satellite networks. IEEE Transactions on Vehicular Technology, 72(1), 1240-1253.
6. Liu, Z., Li, J., Wang, Y., Li, X., & Chen, S. (2017). HGL: A hybrid global-local load balancing routing scheme for the Internet of Things through satellite networks. International Journal of Distributed Sensor Networks, 13(3), 1550147717692586.

### QoS
1. Hou, C., & Zhu, Y. (2023, August). The QoS guaranteed routing strategy in low Earth orbit satellite constellations. In 2023 IEEE/CIC International Conference on Communications in China (ICCC Workshops) (pp. 1-6). IEEE.
2. Zuo, P., Wang, C., Yao, Z., Hou, S., & Jiang, H. (2021, September). An intelligent routing algorithm for LEO satellites based on deep reinforcement learning. In 2021 IEEE 94th Vehicular Technology Conference (VTC2021-Fall) (pp. 1-5). IEEE.
3. Han, C., Xiong, W., & Yu, R. (2024). Deep Reinforcement Learning-Based Multipath Routing for LEO Megaconstellation Networks. Electronics, 13(15), 3054.
4. Na, Z. Y., Deng, Z. A., Chen, N., Gao, Z. H., & Guo, Q. (2015, August). An active distributed QoS routing for LEO satellite communication network. In 2015 10th International Conference on Communications and Networking in China (ChinaCom) (pp. 538-543). IEEE.
5. Yu, S., Hao, N., Long, J., & Liu, L. (2024, October). A Multi-QoS-Constrained Routing Algorithm for Double-Layer Satellite Networks Based on Enhanced NSGA-II Algorithm. In 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 309-314). IEEE.

---

Thank you for reviewing this project.
