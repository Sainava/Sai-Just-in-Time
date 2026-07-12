# Audit: Just-in-Time (JiT) - Training-Free Spatial Acceleration for Diffusion Transformers

## 1. Core Observations & Motivation

The JiT framework is built upon the following observations regarding Diffusion Transformers (DiTs):

* **Spatial Redundancy:** Diffusion models inherently synthesize low-frequency global structures early in the generation process and refine high-frequency details later. Existing methods waste compute by applying uniform computational effort to all spatial regions simultaneously across all timesteps.


* **Subspace Diffusion Inspiration:** Generation can be confined to low-dimensional subspaces without requiring explicit spatial resizing (e.g., upsampling operators), which frequently introduce aliasing artifacts and require error-prone distribution corrections.



---

## 2. Architectural Context & The Baseline Diffusion Process

JiT operates within the Flow Matching paradigm of Diffusion Transformers (such as FLUX.1-dev).

* **The Latent Space:** The generation process operates on a compressed continuous latent tensor obtained via a VAE, denoted as $x \in \mathbb{R}^{C \times H \times W}$.


* **Tokenization:** For DiTs, this tensor is patched into a sequence of $N$ tokens, denoted as $y(t) \in \mathbb{R}^{Nd}$, where $d$ is the token dimension.


* **The Standard Governing ODE:** The DiT network $u_\theta$ models the velocity field to drive the tokens from pure noise ($t=0$) to data ($t=1$):



$$\frac{dy(t)}{dt} = u_\theta(y(t), t), \quad t \in [0, 1]$$


* **The Bottleneck:** Evaluating $u_\theta$ requires processing all $N$ tokens at every timestep, incurring a massive $\mathcal{O}(N^2)$ computational cost due to self-attention. JiT operates by manipulating the spatial dimension ($N$) entering these blocks.



---

## 3. The Foundational Matrices & Operators (Token-Subset Chain)

To bypass the $\mathcal{O}(N^2)$ bottleneck, JiT constructs a mathematically rigorous system to mask and manipulate dynamic token subsets.

* **The Nested Hierarchy:** JiT utilizes a coarse-to-fine generation strategy defined by a nested sequence of active token indices:



$$\Omega_K \subset \Omega_{K-1} \subset \dots \subset \Omega_1 \subset \Omega_0 = \{1, 2, \dots, N\}$$


* **Token Counts:** The number of active anchor tokens at a given stage $k$ is $m_k$, where $\vert{}\Omega_k\vert{} = m_k$.


* **The Selector Matrix ($S_k$):** A matrix $S_k \in \{0, 1\}^{Nd \times m_k d}$ used to extract only the active anchor tokens from the full sequence.



$$y_k = S_k^\top y$$


* **The Orthogonal Projector ($P_k$):** A spatial mask matrix over the full $N$-dimensional space to isolate the anchor subspace:



$$P_k := S_k S_k^\top$$


* **The Transition Projector ($Q_k$):** An operator that precisely isolates the newly activated tokens $R_k := \Omega_{k-1} \setminus \Omega_k$ required during a stage transition:



$$Q_k := P_{k-1} - P_k$$



---

## 4. Core Component 1: Spatially Approximated Generative ODE (SAG-ODE)

Instead of feeding all tokens to the DiT, SAG-ODE computes the velocity field strictly on the sparse anchor tokens and extrapolates it.

* **The SAG-ODE Formula:**

$$\frac{dy(t)}{dt} = \Pi_k u_\theta(S_k^\top y(t), t) = v_t$$


* **The Augmented Lifter Operator ($\Pi_k$):** This operator bridges the $m_k$-dimensional computed velocity back to the full $N$-dimensional space:



$$\Pi_k u_\theta := S_k u_\theta + \mathcal{I}_k(u_\theta)$$


* $S_k u_\theta$: Acts as an embedding map, placing exact computed velocities back into their corresponding anchor coordinates.


* $\mathcal{I}_k(u_\theta)$: A spatial interpolation operator (nearest-neighbor + masked Gaussian blur) that approximates the velocity for the inactive background subspace.




* **The Consistency Property Proof:** By design, the interpolation operator has zero effect on the anchor positions ($S_k^\top \mathcal{I}_k(u_\theta) = 0$). This guarantees zero error introduction on computed tokens:



$$S_k^\top (\Pi_k u_\theta) = S_k^\top (S_k u_\theta + \mathcal{I}_k(u_\theta)) = u_\theta$$



---

## 5. Core Component 2: Deterministic Micro-Flow (DMF) & Stage Transitions

When moving from stage $k$ to $k-1$, JiT must activate new tokens without causing structural discontinuities or statistical mismatches.

* **Importance-Guided Token Activation (ITA):** To decide *which* tokens to add, the model computes the local variance of the velocity field using an average pooling window $\mathcal{W}$ and selects tokens with the highest dynamic activity:



$$I(t) = \mathbb{E}_{\mathcal{W}}[u_\theta(y(t), t) \odot u_\theta(y(t), t)] - (\mathbb{E}_{\mathcal{W}}[u_\theta(y(t), t)]) \odot (\mathbb{E}_{\mathcal{W}}[u_\theta(y(t), t)])$$


* **Target State Construction ($y^\star_k$):** The mathematically correct target state for new tokens at transition time $T_k$, combining structural priors and required noise levels.


1. Tweedie's Formula predicts the clean image: $\hat{y}(1) = y_{t_{i-1}} + (1 - t_{i-1})v_{t_{i-1}}$.


2. The final target state formula:

$$y^\star_k = Q_k(T_k\Phi_k(S_k^\top \hat{y}(1)) + (1 - T_k)\epsilon)$$



*(Where $\Phi_k$ is the structural prior operator and $\epsilon \sim \mathcal{N}(0, I)$ is standard Gaussian noise)*.




* **The Hitting ODE (The Micro-Flow):** New tokens transition to $y^\star_k$ via a finite-time ODE over interval $[T_k - \delta, T_k]$. The time-varying rate $(T_k - t)^{-1}$ ensures perfect convergence to the target:



$$Q_k \dot{y}(t) = \frac{y^\star_k - Q_k y(t)}{T_k - t}$$


* **Anchor Isolation:** During this micro-flow, existing anchors are frozen to prevent interference:



$$P_k \dot{y}(t) = 0$$



---

## 6. The End-to-End JiT Sampling Process

The comprehensive workflow during inference unfolds as follows:

1. **Initialization:** The model initializes the full-dimensional pure noise vector $y_{t_0} \sim \mathcal{N}(0, I)$ and sets the initial sparse token set $\Omega_K$ based on a deterministic strided grid with boundary constraints.


2. **Generative Loop (SAG-ODE):** The ODE solver begins stepping through time. At standard timesteps, the model extracts the anchor tokens $S_k^\top y_t$, runs them through the DiT, lifts them to full dimensions using $\Pi_k$, and updates the image.


3. **Stage Transition Check:** If the current timestep matches a scheduled transition time $T_k$, the transition protocol triggers.


4. **Execute Transition (DMF):** The variance map $I(T_k)$ is calculated, new tokens $R_k$ are selected, the target state $y^\star_k$ is constructed using Tweedie's formula, and the micro-flow safely bridges the new tokens into the active pool.


5. **Finalization:** The loop continues with a newly expanded subset of tokens until $t=1$, where all $N$ tokens are active for final high-frequency detail refinement, followed by VAE decoding into the final image.