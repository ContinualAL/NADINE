# NADINE
Automatic Construction of Multi-layer Perceptron Network from Streaming Examples. The 28th ACM International Conference on Information and Knowledge Management (CIKM).

# Abstract
Autonomous construction of deep neural network (DNNs) is desired for data stream mining because it potentially offers two advantages: proper model’s capacity and quick reaction to drift and shift. While the self-organizing mechanism of DNNs remains an open issue, this task is even more challenging to be developed for a standard multi-layer DNNs than that using the different-depth structures, because addition of a new layer results in information loss of previously trained knowledge. A Neural Network with Dynamically Evolved Capacity (NADINE) is proposed in this paper. NADINE features a fully open structure where its network structure, depth and width, can be automatically evolved from scratch in the online manner and without the use of problem-specific thresholds. NADINE is structured under a standard MLP-like architecture and the catastrophic forgetting issue during the hidden layer addition phase is resolved using the proposal of soft-forgetting and adaptive memory methods without knowing task boundaries and time points of concept changes. The advantage of NADINE, namely elastic structure and online learning trait, is numerically validated using nine data stream classification and regression problems under the prequential test-then-train procedure where it demonstrates performance’s improvement over prominent algorithms in all problems and deals with data stream regression and classification problems equally well.


# Citation
If you use this code, please cite:

@inproceedings{10.1145/3357384.3357946,
author = {Pratama, Mahardhika and Za’in, Choiru and Ashfahani, Andri and Ong, Yew Soon and Ding, Weiping},
title = {Automatic Construction of Multi-Layer Perceptron Network from Streaming Examples},
year = {2019},
isbn = {9781450369763},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3357384.3357946},
doi = {10.1145/3357384.3357946},
booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
pages = {1171–1180},
numpages = {10},
keywords = {online learning, continual learning, data streams, concept drifts, deep learning},
location = {Beijing, China},
series = {CIKM ’19}
}
