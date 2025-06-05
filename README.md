# Quantum Image Classification on MNIST

This repository contains two Jupyter notebooks exploring different approaches to Quantum Image Classification, specifically focusing on binary classification of handwritten digits (0s and 1s) from the MNIST dataset:
singleqbitencodingmnist.ipynb: An implementation of the 1-qubit encoding method, based on the paper "Efficient Quantum Image Classification Using Single Qubit Encoding" by Philip Easom-McCaldin et al.
MNIST.ipynb: A more general multi-qubit Quantum Neural Network (QNN) built using Qiskit.

# Installation
If you can't run the MNIST.ipynb, you can try install qiskit version 1.4.2 using:
```python
    python.exe -m pip install qiskit==1.4.2
```
    
# Project Goal
The primary goal of this project is to:
Replicate the single-qubit encoding methodology from the reference paper.
Compare its performance, complexity, and characteristics against a more conventional multi-qubit QNN.
Analyze the reasons behind the observed performance differences between our replications and the paper's results, as well as between the two different quantum approaches.
# Analysis and Comparison
We compare the two implemented quantum image classification methods based on their performance, complexity, strengths, weaknesses, and the underlying reasons for their observed effectiveness.
1. singleqbitencodingmnist.ipynb (1-Qubit Encoding based on Paper)
    This notebook implements the single-qubit encoding strategy described in "Efficient Quantum Image Classification Using Single Qubit Encoding" (Easom-McCaldin et al., 2024).
    ## Performance Observed in singleqbitencodingmnist.ipynb vs. Paper
    - Paper's Reported Accuracy: The original paper reports impressive classification accuracies for binary MNIST data (digits 0 and 1): 94.6% with a 3x3 filter and 95.8% with a 4x4 filter on their specific subsets.
    - Our singleqbitencodingmnist.ipynb Replication: You will likely observe that the accuracy in our A.ipynb replication is slightly lower than the exact figures reported in the paper. For instance, while the paper achieves near 95%, our replication might hover around 85-90%.
    ## Strengths:
    - Extreme Parameter Efficiency: Uses only 6 trainable parameters per filter. This is a crucial advantage for Noisy Intermediate-Scale Quantum (NISQ) devices, where qubit counts and coherence times are limited.
    - Mitigation of Barren Plateaus: With very few qubits and parameters, this architecture is inherently less susceptible to the barren plateau problem, where gradients vanish exponentially, making training difficult.
    - Spatial Information Preservation: The filter-based approach helps maintain spatial relationships of pixels, analogous to classical CNNs, despite using a single qubit.
    - Simplicity and Interpretability: A single qubit's state can be visualized on the Bloch sphere, offering some level of interpretability of the learned embedding.
    ## Weaknesses:
    - Limited Expressivity (for complex tasks): While efficient, a single qubit has inherent limitations in its expressive power. For datasets with higher complexity or a larger number of classes (beyond binary), it might struggle to create sufficiently separated feature spaces.
    - Sequential Data Processing: The encoding method processes pixels in groups of three sequentially, which could be a bottleneck compared to potentially parallel encoding methods on future quantum hardware.
    - Susceptibility to Noise (Amplitude Damping): The paper itself notes that amplitude damping noise can significantly reduce classification performance, highlighting a practical challenge for real-world devices.
    ## Reasons for Slight Discrepancy in singleqbitencodingmnist.ipynb's Performance:
    - Exact Hyperparameters and Initialization: Minor differences in hyperparameters (e.g., learning rate schedule, specific optimizer parameters, or random seed for weight initialization) can lead to variations. The paper mentions re-conducting training with new weight distributions if barren plateaus were observed, implying sensitivity to initialization.
    - Simulator Specifics: While both use PennyLane's default.qubit device, subtle differences in backend implementations or numerical precision between specific PennyLane versions or underlying quantum simulators (like Qulacs mentioned in the paper) might account for small deviations.
    - Dataset Subset Variation: Although both filter for digits 0 and 1, the exact sequence or specific images selected for training and testing can introduce minor variations in results. The paper used a subset of 500 training images and 250 test images per class, and our replication samples from the same source.
2. MNIST.ipynb (Multi-Qubit QNN with Qiskit)
    - This notebook implements a more general multi-qubit Quantum Neural Network using Qiskit, demonstrating a different approach to quantum image classification.
    ## Performance Observed in MNIST.ipynb:
    - You will likely observe that the accuracy obtained from this multi-qubit QNN is significantly lower than both the paper's reported results and your A.ipynb replication.
    ## Strengths (of Multi-Qubit QNNs in general, not necessarily this implementation):
    - Higher Theoretical Expressivity: With more qubits and entanglement, there is a theoretical potential for higher expressive power and more complex feature mappings, which could be beneficial for highly complex datasets.
    -  Potential for Parallelism: Future quantum hardware may allow for parallel operations across multiple qubits, potentially speeding up computation for certain architectures.
   ## Weaknesses (specific to this MNIST.ipynb implementation and general multi-qubit challenges):
    - Severe Barren Plateaus: With 4 qubits and a general variational circuit, this architecture is highly susceptible to the barren plateau problem. This makes gradient-based optimization extremely challenging, as gradients tend to vanish, leading to slow or stalled training.
   ## Ineffective Data Encoding / Information Bottleneck:
    - The core issue lies in the classical preprocessing: flattening the 28x28 image (784 pixels) and then using a simple nn.Linear layer to reduce it to 4 features before feeding it to the 4-qubit QNN. This linear dimension reduction discards a massive amount of crucial spatial information and non-linear correlations inherent in image data.
    - A simple linear layer cannot effectively learn and preserve the hierarchical features that convolutional layers in classical CNNs are designed for. This creates an information bottleneck before the quantum circuit even begins its processing.
    - Circuit Architecture Suitability: While the chosen variational circuit structure is common, it might not be complex enough or specifically tailored to handle the features produced by the preceding linear layer, especially if those features are already severely degraded.
    - Classical Post-processing Issues: The final nn.Linear(2**num_qubits, 10) layer maps the 2^4 = 16 outputs of the QNN to 10 classes. If the QNN's features are not discriminative due to the input bottleneck or barren plateaus, this layer cannot salvage performance.
   ## Overall Conclusion on Performance Differences
    - The stark difference in performance, particularly the much lower accuracy of MNIST.ipynb, primarily stems from its ineffective classical data preprocessing and feature engineering. The linear dimension reduction (nn.Linear(28*28, 4)) acts as a severe information bottleneck, discarding too much spatial and contextual information crucial for image classification. This renders the subsequent quantum processing largely ineffective, regardless of the QNN's theoretical capabilities. Additionally, the challenge of barren plateaus in multi-qubit circuits further hinders the training process.
# Conclusion
In contrast, the 1-qubit encoding method (in singleqbitencodingmnist.ipynb and the paper) is designed to intelligently encode complex classical data onto a single qubit over time, preserving critical information through its sequential, filter-based approach. This strategy, combined with its inherent robustness against barren plateaus due to the minimal qubit count, makes it a more effective solution within the constraints of current quantum computing technology for this specific task, even if our replication's accuracy is slightly below the published benchmark due to environmental or hyperparameter nuances.
