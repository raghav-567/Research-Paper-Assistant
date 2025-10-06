# Literature Review

## 1. Lecture Notes: Optimization for Machine Learning
**Authors**: Elad Hazan

**Summary**: These lecture notes cover optimization techniques for machine learning, focusing on both theoretical foundations and practical algorithms. Key topics include:

*   **Fundamentals:** Convexity, optimality conditions, gradient descent, and stochastic gradient descent (SGD).
*   **Generalization:** Regret minimization, online gradient descent, and their connection to generalization performance.
*   **Regularization:** The regularization framework, mirrored descent, and its application to deriving online algorithms.
*   **Adaptive Optimization:** Adaptive learning rates, AdaGrad, and related state-of-the-art algorithms.
*   **Variance Reduction:** Techniques to reduce variance in stochastic gradient estimates for faster convergence.
*   **Acceleration:** Nesterov acceleration for improved convergence rates.

The notes present algorithms, theoretical analyses, and examples of applications in machine learning problems such as empirical risk minimization, matrix completion, and neural network training. Methods include mathematical analysis, algorithm design, and connections between online learning and stochastic optimization.

## 2. An Optimal Control View of Adversarial Machine Learning
**Authors**: Xiaojin Zhu

**Summary**: This paper proposes framing adversarial machine learning as an optimal control problem. The author argues that adversarial attacks, unlike standard machine learning scenarios, exhibit non-i.i.d. structures better suited to a control theory framework. The paper defines the machine learning system as the "plant," adversarial actions as "control inputs," and the adversary's objectives (harm and stealth) as "control costs."  The author illustrates this framework with examples like training-data poisoning, test-time attacks, and adversarial reward shaping, focusing on deterministic discrete-time optimal control. The paper suggests this perspective can leverage advances in control theory and reinforcement learning to better understand and defend against adversarial attacks. The paper specifically models batch training set poisoning as a degenerate one-step control problem, using Support Vector Machines (SVM) as an example.

## 3. Minimax deviation strategies for machine learning and recognition with short learning samples
**Authors**: Michail Schlesinger, Evgeniy Vodolazskiy

**Summary**: This paper addresses the problem of small learning samples in machine learning. It demonstrates that traditional methods like maximum likelihood learning can perform worse than simply ignoring the learning sample when the sample size is very small. The authors illustrate this with examples where maximum likelihood estimation leads to higher risk than a minimax strategy that doesn't use the learning sample. The paper then introduces the concept of minimax deviation learning as a potential solution to overcome the flaws of existing methods when dealing with limited data.

