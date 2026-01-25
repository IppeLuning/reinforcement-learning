## Transfer of  winning Lottery Tickets in RL (SAC on reach v-3 and push-v3)



### Architecture

### Architecture & Pipeline

* **`src/agents/`**: Contains the primary SAC logic and the custom `SACTrainState`. This state is the "heart" of the pruning logic, as it explicitly re-applies binary weight masks during every gradient update to prevent pruned parameters from regenerating.
* **`src/networks/`**: Implements the Actor (Gaussian policy with Tanh squashing) and the Critic (Twin-Q networks) using Flax Linen MLPs.
* **`src/lth/`**: The Lottery Ticket Hypothesis toolkit, featuring algorithms for Magnitude pruning and Gradient-based (Taylor expansion) saliency.
* **`src/data/`**: A high-performance Replay Buffer utilizing NumPy for CPU-side storage to save VRAM, with optimized JAX sampling for training.

---

#### Experimental Pipeline (`scripts/`)


1. **`01_train_dense.py`**: Establishes the performance baseline by training a full-parameter model and saving the initialization and rewind anchors.
2. **`02_create_mask.py`**: Analyzes the trained dense model to identify essential weights and generates the binary PyTree masks.
3. **`03_train_ticket.py`**: Performs the "Ticket Run." It rewinds the model to an earlier state, applies the mask, and retrains the sparse network.
4. **`04_transfer_ticket.py`**: Tests the generality of the discovered tickets by attempting to transfer a sparse mask from a source task (e.g., Reach) to a target task (e.g., Push).


---

### The LTH Pipeline

The experiment workflow follows a three-stage iterative process managed by the orchestrator:

1. **Dense Training (Round 0):** Train a standard agent and capture a "rewind anchor" (e.g., at step 20,000).
2. **Pruning:** Analyze the final trained weights to create a binary mask representing the most salient connections.
3. **Ticket Retraining:** Rewind the model to the anchor step, apply the mask, and retrain to verify if the sparse subnetwork (the "winning ticket") can match dense performance.

---

### Installation

```bash
# Clone the repository
git_clone https://github.com/IppeLuning/reinforcement-learning.git
cd reinforcement-learning

# Install dependencies
pip install -r requirements.txt

```

---

### Running Experiments

The entire pipeline is controlled via `run_pipeline.py`.

```python
python -m run_pipeline

```

#### Configuration

All hyperparameters are managed in `config.yaml`, including:

* `total_steps`: Number of environment steps per round.
* `rewind_steps`: The anchor step used for Late Rewinding.
* `parallel`: Settings for vectorized environment collection.


