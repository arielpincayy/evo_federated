# Related Works Review

## Scope and Retrieval

Targeted search performed on 2026-09-25 for published scholarly works from 2021
through 2026. The search excluded records whose publication type was a preprint
or posted content. Journal articles were prioritized; peer-reviewed proceedings
were retained when they directly addressed federated client selection.

Sources queried:

- OpenAlex works endpoint, using targeted searches for federated neural architecture search, client selection, non-IID federated learning, and fairness.
- Crossref works and DOI endpoints for publication type, venue, dates, pages, and DOI metadata.
- Official USENIX metadata for Oort, including its proceedings citation and abstract.

The retrieval was targeted rather than exhaustive. OpenAlex returned broad search
sets because its search field includes title, abstract, and indexed full text;
the final set was manually filtered for direct methodological relevance and
publication status.

## Recommended Core Sources

### Federated NAS

1. **Zhu, Zhang, and Jin (2021), ``From federated learning to federated neural architecture search: a survey,'' Complex & Intelligent Systems.** DOI: `10.1007/s40747-020-00247-z`. This is the most useful field-level survey for positioning federated NAS by search paradigm and objective structure.
2. **Zhu and Jin (2022), ``Real-Time Federated Evolutionary Neural Architecture Search,'' IEEE Transactions on Evolutionary Computation.** DOI: `10.1109/tevc.2021.3099448`. It is the closest evolutionary-NAS precedent: double sampling reduces the payload and client computation during the search.
3. **Pan et al. (2021), ``Privacy-Preserving Multi-Granular Federated Neural Architecture Search: A General Framework,'' IEEE Transactions on Knowledge and Data Engineering.** DOI: `10.1109/tkde.2021.3116248`. MGFNAS searches at micro and macro architectural granularity and aggregates local architectures with NAPA; it differs from a single globally shared architecture trained by FedAvg.
4. **Yuan et al. (2024), ``Resource-Aware Federated Neural Architecture Search over Heterogeneous Mobile Devices,'' IEEE Transactions on Big Data.** DOI: `10.1109/tbdata.2022.3227403`. FedNAS separates search from training and reduces cost through partial-client training, early candidate dropping, and dynamic round counts.
5. **Zhang et al. (2022), ``Toward Tailored Models on Private AIoT Devices: Federated Direct Neural Architecture Search,'' IEEE Internet of Things Journal.** DOI: `10.1109/jiot.2022.3154605`. It addresses architecture search directly under non-IID data and device constraints, with emphasis on tailored edge models.
6. **Liu et al. (2024), ``Finch: Enhancing Federated Learning With Hierarchical Neural Architecture Search,'' IEEE Transactions on Mobile Computing.** DOI: `10.1109/tmc.2023.3315451`. Finch clusters clients by data distribution and searches subnets for clusters, making it a relevant contrast to update-similarity and update-dissimilarity participation rules.
7. **Yan et al. (2024), ``Peaches: Personalized Federated Learning With Neural Architecture Search in Edge Computing,'' IEEE Transactions on Mobile Computing.** DOI: `10.1109/tmc.2024.3373506`. Peaches emphasizes personalized architectures, whereas the present study deliberately evaluates one deployable global architecture.
8. **Yang and Liu (2025), ``Heterogeneity-Aware Personalized Federated Neural Architecture Search,'' Entropy.** DOI: `10.3390/e27070759`. This recent work jointly considers resource and statistical heterogeneity for personalized models; it motivates treating those forms of heterogeneity as distinct from the global-model objective used here.
9. **Zhang et al. (2024), ``ENASFL: A Federated Neural Architecture Search Scheme for Heterogeneous Deep Models in Distributed Edge Computing Systems,'' IEEE Transactions on Network Science and Engineering.** DOI: `10.1109/tnse.2023.3344850`. It targets heterogeneous deep models across edge devices and is useful as a model-personalization comparison.
10. **Zhang et al. (2025), ``Privacy-Preserving Federated Neural Architecture Search With Enhanced Robustness for Edge Computing,'' IEEE Transactions on Mobile Computing.** DOI: `10.1109/tmc.2024.3490835`. This paper adds robustness and privacy objectives to federated NAS, providing a recent extension beyond accuracy and cost.

### Client Selection, Cost, and Heterogeneity

11. **Lai et al. (2021), ``Oort: Efficient Federated Learning via Guided Participant Selection,'' OSDI 2021.** USENIX proceedings, pp. 19-35. Oort jointly uses data utility and execution speed to improve time-to-accuracy and federated testing; it is a systems-oriented precedent for cost-aware participation.
12. **Deng et al. (2022), ``AUCTION: Automated and Quality-Aware Client Selection Framework for Efficient Federated Learning,'' IEEE Transactions on Parallel and Distributed Systems.** DOI: `10.1109/tpds.2021.3134647`. AUCTION learns a client-selection policy with reinforcement learning from client quality, data, and budget feedback.
13. **Shin et al. (2022), ``FedBalancer,'' Proceedings of MobiSys 2022.** DOI: `10.1145/3498361.3538917`. FedBalancer selects informative local samples and controls deadlines; it is relevant to cost accounting but operates primarily at sample selection and straggler control rather than update geometry.
14. **Zhao et al. (2023), ``Participant Selection for Federated Learning With Heterogeneous Data in Intelligent Transport System,'' IEEE Transactions on Intelligent Transportation Systems.** DOI: `10.1109/tits.2022.3149753`. The method uses a participant utility designed for data and device heterogeneity, offering a direct comparison point for random-k selection.
15. **Zhang et al. (2024), ``Addressing Heterogeneity in Federated Learning with Client Selection via Submodular Optimization,'' ACM Transactions on Sensor Networks.** DOI: `10.1145/3638052`. Client selection is formulated as knapsack-constrained submodular maximization over systems and statistical heterogeneity; this is a useful optimization-based contrast to pairwise similarity matching.
16. **Ezzeldin et al. (2023), ``FairFed: Enabling Group Fairness in Federated Learning,'' Proceedings of AAAI 2023.** DOI: `10.1609/aaai.v37i6.25911`. FairFed changes aggregation to improve group fairness under heterogeneous populations; it supports positioning worst-client performance as a robustness objective rather than a fairness guarantee.

### Surveys for the Motivation

17. **Lu et al. (2024), ``Federated Learning With Non-IID Data: A Survey,'' IEEE Internet of Things Journal.** DOI: `10.1109/jiot.2024.3376548`. It organizes the effects of non-IID data on convergence, communication, class imbalance, and client selection.
18. **Ye et al. (2024), ``Heterogeneous Federated Learning: State-of-the-art and Research Challenges,'' ACM Computing Surveys.** DOI: `10.1145/3625558`. It separates statistical, model, communication, and device heterogeneity, which is useful for keeping the present experimental claim narrow.

## Proposed Related-Work Section

The following text is suitable for integration into `Paper/evofederated_paper.tex`
after the introductory FL/NAS background. It intentionally avoids claiming that
the present method improves on these works until the multiseed campaign is
analyzed.

> Recent work has established federated neural architecture search (FedNAS) as a distinct intersection of privacy-preserving optimization and neural architecture design. A field survey organizes FedNAS by search paradigm, execution mode, and objective structure, including evolutionary, reinforcement-learning, and gradient-based methods as well as single- and multi-objective formulations \cite{Zhu2021from}. Evolutionary FedNAS has been used to reduce search overhead through double sampling of submodels and clients \cite{Zhu2022real}. Other approaches search architectural structure at multiple granularities and aggregate local architecture information \cite{Pan2021privacy}, decouple architecture search from model training and reduce cost through partial-client training and early candidate elimination \cite{Yuan2024resource}, or target tailored and personalized models for heterogeneous edge devices \cite{Zhang2022toward,Liu2024finch,Yan2024peaches,Yang2025heterogeneity}. These studies generally optimize resource efficiency, personalization, or architecture heterogeneity; the present work instead evaluates each candidate as one global model trained by FedAvg and uses mean and worst-client validation accuracy as the search objectives.
>
> Client participation is a second line of related work. Oort combines data utility and execution speed to improve federated time-to-accuracy \cite{Lai2021oort}, while AUCTION learns quality-aware selection policies from client feedback \cite{Deng2022auction}. Other methods use participant utility under heterogeneous data and devices \cite{Zhao2023participant} or formulate selection as a submodular optimization problem with a systems-and-statistical heterogeneity objective \cite{Zhang2024addressing}. FedBalancer addresses a related cost problem through informative sample selection and adaptive deadlines rather than update-based client pairing \cite{Shin2022fedbalancer}. These approaches motivate measuring actual local-training and communication costs instead of equating a nominal participation fraction with total computational cost.
>
> Non-IID and heterogeneous federated learning surveys emphasize that statistical, model, communication, and device heterogeneity have different effects and should not be conflated \cite{Lu2024federated,Ye2024heterogeneous}. Accordingly, this study treats update similarity and dissimilarity as participation heuristics within a fixed global-model protocol, not as general solutions to all forms of heterogeneity. Fairness-aware aggregation such as FairFed further shows that worst-group or worst-client objectives can encode concerns not captured by average accuracy \cite{Ezzeldin2023fairfed}; here, the minimum client accuracy is reported as a robustness objective, without claiming demographic fairness.

## BibTeX and Validation

- `Paper/related_works_sources.bib` contains 20 DOI-resolved records plus the official USENIX Oort proceedings record.
- `Paper/related_works_dois.txt` records the DOI retrieval set.
- Citation validation found 20 DOI entries valid, no duplicates, and no errors. Two records carry explicit notes because Crossref did not assign a volume at retrieval time.
- No preprint or `posted-content` record was included in the bibliography.
