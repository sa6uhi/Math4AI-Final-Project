From Linear Scores to a Single Hidden Layer: A Mathematical Study
Authors: Sabuhi Nazarov & Agshin Fataliyev
Institution: National AI Center — AI Academy, Baku
Project Overview
This project explores the fundamental conditions under which a one-hidden-layer nonlinear classifier outperforms a traditional linear rule. By analyzing different data geometries, the study provides controlled evidence regarding the necessity of representational depth and nonlinearity in machine learning models.
Core Hypotheses
Linear Sufficiency: When class structures are near an affine subspace, a linear model is geometrically sufficient for classification.
Representational Gain: When class geometry is curved, a hidden nonlinear layer is required to warp the features into a linearly separable configuration.
Methodology
The study emphasizes a fair and reproducible comparison between models.
Implementations: Both Softmax Regression and a Single-Hidden-Layer Tanh Network were built from scratch using NumPy.
Protocol: Shared preprocessing, train-fit standardization, and a deterministic seed (42) were used to ensure consistency.
Stability: Models utilized L2 regularization ($\lambda = 0.01$), monitored for NaNs, and verified probability normalization at every epoch
Experimental Results
1. Linear Gaussian Dataset (Linear Sufficiency)
   Tested on two mildly overlapping Gaussian classes.
   Performance: Both the linear and nonlinear models achieved a test accuracy of 95.00%.
   Conclusion: There is no gain from a hidden layer when the decision boundary is geometrically linear.
2. Moons Dataset (Nonlinear Necessity)
   Tested on curved class geometry where feature warping is required.
   Softmax (Linear): 85.00% Accuracy.
   Hidden-Layer (Tanh): 96.25% Accuracy.
   Visual Evidence: Decision boundaries show that while Softmax is limited by straight lines, the hidden layer captures curvature through nonlinear warping.
   3. Dimensionality & SVD Analysis (Advanced Track)
   The study analyzed the intrinsic dimensionality of digit data.
   Variance Capture: The first principal component explains 15% of variance; the first 10 explain ~75%.
   Compression: Using $m=40$ components recovers 99.6% accuracy while achieving a 37.5% reduction in feature space.
   Practical Threshold: $m=40$ is identified as the optimal balance between performance and signal preservation.
Summary of Findings
Key Takeaway: Model choice should be dictated by data geometry. Hidden nonlinearity is essential for curved geometries but redundant for linear ones.
Mechanism: Tanh units create curved feature transforms, allowing a final linear softmax to act on a warped space that is easier to separate.
Limitations & Next Steps
Current Limits: The study is currently limited to synthetic datasets and lacks fixed-digit benchmarks or repeated-seed confidence intervals.
Future Work: Next steps involve running benchmarks on real-world datasets, exploring alternative activation functions, and adding statistical intervals.
Keywords: Softmax Regression, Neural Networks, Backpropagation, NumPy, SVD.
