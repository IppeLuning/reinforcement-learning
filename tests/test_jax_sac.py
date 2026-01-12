"""Unit tests for JAX SAC implementation.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

Tests are designed to be:
- Fast (< 1 second each)
- CPU-compatible (no Metal/GPU required)
- Deterministic (fixed seeds)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Force CPU for CI compatibility
jax.config.update("jax_platform_name", "cpu")


# =============================================================================
# Network Tests
# =============================================================================

class TestMLP:
    """Tests for MLP network module."""
    
    def test_mlp_output_shape(self):
        """MLP should produce correct output dimensions."""
        from src.jax.networks.mlp import MLP
        
        mlp = MLP(hidden_dims=(64, 64), output_dim=10)
        params = mlp.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        
        output = mlp.apply(params, jnp.ones((32, 5)))
        assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    
    def test_mlp_deterministic(self):
        """MLP forward pass should be deterministic."""
        from src.jax.networks.mlp import MLP
        
        mlp = MLP(hidden_dims=(32,), output_dim=4)
        params = mlp.init(jax.random.PRNGKey(42), jnp.ones((1, 8)))
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
        
        out1 = mlp.apply(params, x)
        out2 = mlp.apply(params, x)
        
        np.testing.assert_array_equal(out1, out2)


class TestActor:
    """Tests for GaussianActor network."""
    
    def test_actor_output_shapes(self):
        """Actor should output mean and log_std with correct shapes."""
        from src.jax.networks.actor import GaussianActor
        
        actor = GaussianActor(act_dim=4, hidden_dims=(64, 64))
        params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
        
        mean, log_std = actor.apply(params, jnp.ones((32, 10)))
        
        assert mean.shape == (32, 4), f"Mean shape mismatch: {mean.shape}"
        assert log_std.shape == (32, 4), f"Log_std shape mismatch: {log_std.shape}"
    
    def test_log_std_bounded(self):
        """Log_std should be bounded within specified range."""
        from src.jax.networks.actor import GaussianActor
        
        actor = GaussianActor(
            act_dim=4, 
            hidden_dims=(64,),
            log_std_bounds=(-20.0, 2.0)
        )
        params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
        
        # Test with various inputs
        for seed in range(5):
            x = jax.random.normal(jax.random.PRNGKey(seed), (100, 10))
            _, log_std = actor.apply(params, x)
            
            assert jnp.all(log_std >= -20.0), "Log_std below minimum"
            assert jnp.all(log_std <= 2.0), "Log_std above maximum"
    
    def test_sample_action_shape(self):
        """Sampled action should have correct shape."""
        from src.jax.networks.actor import GaussianActor, sample_action
        
        actor = GaussianActor(act_dim=4, hidden_dims=(32,))
        params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
        
        mean, log_std = actor.apply(params, jnp.ones((16, 10)))
        action, log_prob = sample_action(mean, log_std, jax.random.PRNGKey(42))
        
        assert action.shape == (16, 4)
        assert log_prob.shape == (16, 1)
    
    def test_action_bounded(self):
        """Sampled actions should be in [-1, 1] due to tanh."""
        from src.jax.networks.actor import GaussianActor, sample_action
        
        actor = GaussianActor(act_dim=4, hidden_dims=(32,))
        params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
        
        mean, log_std = actor.apply(params, jnp.ones((100, 10)))
        action, _ = sample_action(mean, log_std, jax.random.PRNGKey(42))
        
        assert jnp.all(action >= -1.0), "Action below -1"
        assert jnp.all(action <= 1.0), "Action above 1"


class TestCritic:
    """Tests for Q-network (Critic)."""
    
    def test_twin_q_output_shapes(self):
        """Twin Q-networks should produce two scalar outputs per sample."""
        from src.jax.networks.critic import TwinQNetwork
        
        critic = TwinQNetwork(hidden_dims=(64, 64))
        params = critic.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 10)),  # obs
            jnp.ones((1, 4))   # action
        )
        
        q1, q2 = critic.apply(params, jnp.ones((32, 10)), jnp.ones((32, 4)))
        
        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)
    
    def test_twin_q_independent(self):
        """Twin Q-networks should produce different values."""
        from src.jax.networks.critic import TwinQNetwork
        
        critic = TwinQNetwork(hidden_dims=(64, 64))
        params = critic.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 10)),
            jnp.ones((1, 4))
        )
        
        x = jax.random.normal(jax.random.PRNGKey(1), (32, 10))
        a = jax.random.normal(jax.random.PRNGKey(2), (32, 4))
        q1, q2 = critic.apply(params, x, a)
        
        # Q1 and Q2 should be different (independent networks)
        assert not jnp.allclose(q1, q2), "Q1 and Q2 should be independent"


# =============================================================================
# Training State Tests
# =============================================================================

class TestTrainState:
    """Tests for SAC training state."""
    
    def test_create_train_state(self):
        """Should create valid training state."""
        from src.jax.training.train_state import create_sac_train_state
        
        state = create_sac_train_state(
            key=jax.random.PRNGKey(0),
            obs_dim=10,
            act_dim=4,
            hidden_dims=(64, 64),
        )
        
        assert state.step == 0
        assert state.actor_params is not None
        assert state.critic_params is not None
        assert state.target_critic_params is not None
    
    def test_soft_update(self):
        """Soft update should interpolate target params."""
        from src.jax.training.train_state import create_sac_train_state
        
        state = create_sac_train_state(
            key=jax.random.PRNGKey(0),
            obs_dim=10,
            act_dim=4,
            hidden_dims=(32,),
        )
        
        # Store original target
        original_target = jax.tree.map(jnp.copy, state.target_critic_params)
        
        # Modify critic params
        state = state.replace(
            critic_params=jax.tree.map(lambda x: x + 1.0, state.critic_params)
        )
        
        # Soft update with tau=0.5
        state = state.soft_update_target(tau=0.5)
        
        # Target should have moved toward critic
        def check_moved(orig, new):
            return not jnp.allclose(orig, new)
        
        moved = jax.tree.map(check_moved, original_target, state.target_critic_params)
        flat_moved = jax.tree.leaves(moved)
        assert all(flat_moved), "Target should have moved"
    
    def test_alpha_property(self):
        """Alpha property should return exp(log_alpha)."""
        from src.jax.training.train_state import create_sac_train_state
        
        state = create_sac_train_state(
            key=jax.random.PRNGKey(0),
            obs_dim=10,
            act_dim=4,
            init_alpha=0.5,
        )
        
        expected = jnp.exp(jnp.log(0.5))
        np.testing.assert_almost_equal(state.alpha, expected, decimal=5)


# =============================================================================
# Replay Buffer Tests
# =============================================================================

class TestReplayBuffer:
    """Tests for replay buffer."""
    
    def test_store_and_sample(self):
        """Buffer should store and sample transitions correctly."""
        from src.jax.buffers.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(obs_dim=10, act_dim=4, max_size=100)
        
        # Store some transitions
        for i in range(50):
            buffer.store(
                obs=np.random.randn(10).astype(np.float32),
                action=np.random.randn(4).astype(np.float32),
                reward=float(i),
                next_obs=np.random.randn(10).astype(np.float32),
                done=0.0,
            )
        
        assert len(buffer) == 50
        
        # Sample batch
        batch = buffer.sample(batch_size=16)
        
        assert batch.obs.shape == (16, 10)
        assert batch.actions.shape == (16, 4)
        assert batch.rewards.shape == (16, 1)
        assert batch.next_obs.shape == (16, 10)
        assert batch.dones.shape == (16, 1)
    
    def test_circular_buffer(self):
        """Buffer should overwrite old data when full."""
        from src.jax.buffers.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(obs_dim=5, act_dim=2, max_size=10)
        
        # Fill beyond capacity
        for i in range(25):
            buffer.store(
                obs=np.ones(5) * i,
                action=np.zeros(2),
                reward=0.0,
                next_obs=np.zeros(5),
                done=0.0,
            )
        
        assert len(buffer) == 10  # Should cap at max_size
        assert buffer.ptr == 5   # Should wrap around


# =============================================================================
# Agent Tests
# =============================================================================

class TestSACAgent:
    """Tests for SAC agent."""
    
    def test_agent_creation(self):
        """Agent should initialize without errors."""
        from src.jax.agents.sac import SACAgent, SACConfig
        
        config = SACConfig(hidden_dims=(32, 32))
        agent = SACAgent(
            obs_dim=10,
            act_dim=4,
            act_low=np.array([-1.0] * 4),
            act_high=np.array([1.0] * 4),
            config=config,
            seed=42,
        )
        
        assert agent.obs_dim == 10
        assert agent.act_dim == 4
    
    def test_action_selection(self):
        """Agent should select valid actions."""
        from src.jax.agents.sac import SACAgent, SACConfig
        
        config = SACConfig(hidden_dims=(32,))
        agent = SACAgent(
            obs_dim=10,
            act_dim=4,
            act_low=np.array([-1.0] * 4),
            act_high=np.array([1.0] * 4),
            config=config,
            seed=42,
        )
        
        obs = np.random.randn(10).astype(np.float32)
        
        # Stochastic action
        action = agent.select_action(obs, eval_mode=False)
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        
        # Deterministic action
        action_det = agent.select_action(obs, eval_mode=True)
        assert action_det.shape == (4,)
    
    def test_single_update(self):
        """Agent should perform one update step."""
        from src.jax.agents.sac import SACAgent, SACConfig
        from src.jax.buffers.replay_buffer import ReplayBuffer
        
        config = SACConfig(hidden_dims=(32,))
        agent = SACAgent(
            obs_dim=10,
            act_dim=4,
            act_low=np.array([-1.0] * 4),
            act_high=np.array([1.0] * 4),
            config=config,
            seed=42,
        )
        
        buffer = ReplayBuffer(obs_dim=10, act_dim=4, max_size=1000)
        
        # Fill buffer
        for _ in range(100):
            buffer.store(
                obs=np.random.randn(10).astype(np.float32),
                action=np.random.randn(4).astype(np.float32),
                reward=np.random.randn(),
                next_obs=np.random.randn(10).astype(np.float32),
                done=float(np.random.rand() < 0.1),
            )
        
        # Perform update
        metrics = agent.update(buffer, batch_size=32)
        
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
        assert metrics["step"] == 1


# =============================================================================
# Pruning Tests
# =============================================================================

class TestMaskOperations:
    """Tests for mask operations."""
    
    def test_compute_sparsity(self):
        """Should correctly compute sparsity."""
        from src.jax.pruning.masks import compute_sparsity
        
        # 50% sparse mask
        mask = {
            "layer": jnp.array([1.0, 1.0, 0.0, 0.0])
        }
        
        sparsity = compute_sparsity(mask)
        assert abs(sparsity - 0.5) < 1e-6
    
    def test_union_masks(self):
        """Union should keep weights active in ANY mask."""
        from src.jax.pruning.masks import union_masks, compute_sparsity
        
        mask1 = {"w": jnp.array([1.0, 0.0, 0.0, 0.0])}
        mask2 = {"w": jnp.array([0.0, 1.0, 0.0, 0.0])}
        mask3 = {"w": jnp.array([0.0, 0.0, 1.0, 0.0])}
        
        union = union_masks([mask1, mask2, mask3])
        
        expected = jnp.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(union["w"], expected)
    
    def test_intersection_masks(self):
        """Intersection should keep weights active in ALL masks."""
        from src.jax.pruning.masks import intersection_masks
        
        mask1 = {"w": jnp.array([1.0, 1.0, 0.0, 0.0])}
        mask2 = {"w": jnp.array([1.0, 0.0, 1.0, 0.0])}
        
        inter = intersection_masks([mask1, mask2])
        
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(inter["w"], expected)
    
    def test_apply_mask(self):
        """Applying mask should zero out pruned weights."""
        from src.jax.pruning.masks import apply_mask
        
        params = {"w": jnp.array([1.0, 2.0, 3.0, 4.0])}
        mask = {"w": jnp.array([1.0, 0.0, 1.0, 0.0])}
        
        masked = apply_mask(params, mask)
        
        expected = jnp.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(masked["w"], expected)


class TestStructuralAnalysis:
    """Tests for structural analysis metrics."""
    
    def test_jaccard_identical(self):
        """Jaccard of identical masks should be 1.0."""
        from src.jax.pruning.analysis import jaccard_similarity
        
        mask = {"w": jnp.array([1.0, 1.0, 0.0, 0.0])}
        
        sim = jaccard_similarity(mask, mask)
        assert abs(sim - 1.0) < 1e-6
    
    def test_jaccard_disjoint(self):
        """Jaccard of disjoint masks should be 0.0."""
        from src.jax.pruning.analysis import jaccard_similarity
        
        mask1 = {"w": jnp.array([1.0, 1.0, 0.0, 0.0])}
        mask2 = {"w": jnp.array([0.0, 0.0, 1.0, 1.0])}
        
        sim = jaccard_similarity(mask1, mask2)
        assert abs(sim - 0.0) < 1e-6
    
    def test_jaccard_symmetric(self):
        """Jaccard should be symmetric."""
        from src.jax.pruning.analysis import jaccard_similarity
        
        mask1 = {"w": jnp.array([1.0, 1.0, 1.0, 0.0])}
        mask2 = {"w": jnp.array([1.0, 0.0, 1.0, 1.0])}
        
        sim12 = jaccard_similarity(mask1, mask2)
        sim21 = jaccard_similarity(mask2, mask1)
        
        assert abs(sim12 - sim21) < 1e-6


class TestMaskManager:
    """Tests for MaskManager."""
    
    def test_store_and_retrieve(self):
        """Should store and retrieve task masks."""
        from src.jax.pruning.masks import MaskManager
        
        manager = MaskManager()
        
        actor_mask = {"w": jnp.ones(10)}
        critic_mask = {"w": jnp.ones(20)}
        
        manager.store_task_mask("task_a", actor_mask, critic_mask)
        
        retrieved_actor, retrieved_critic = manager.get_task_mask("task_a")
        
        np.testing.assert_array_equal(retrieved_actor["w"], actor_mask["w"])
        np.testing.assert_array_equal(retrieved_critic["w"], critic_mask["w"])
    
    def test_union_mask(self):
        """Should compute union of all stored masks."""
        from src.jax.pruning.masks import MaskManager
        
        manager = MaskManager()
        
        manager.store_task_mask(
            "task_a",
            {"w": jnp.array([1.0, 0.0, 0.0])},
            {"w": jnp.array([1.0, 0.0, 0.0])}
        )
        manager.store_task_mask(
            "task_b", 
            {"w": jnp.array([0.0, 1.0, 0.0])},
            {"w": jnp.array([0.0, 1.0, 0.0])}
        )
        
        union_actor, union_critic = manager.get_union_mask()
        
        expected = jnp.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(union_actor["w"], expected)


# =============================================================================
# Normalizer Tests
# =============================================================================

class TestNormalizer:
    """Tests for running normalizer."""
    
    def test_normalize_zero_mean(self):
        """After enough updates, normalized data should have ~zero mean."""
        from src.jax.utils.normalizer import RunningNormalizer
        
        normalizer = RunningNormalizer.create(obs_dim=5)
        
        # Update with standard normal data
        for _ in range(100):
            batch = jax.random.normal(jax.random.PRNGKey(_), (64, 5))
            normalizer = normalizer.update(batch)
        
        # Normalize new data
        test_data = jax.random.normal(jax.random.PRNGKey(999), (1000, 5))
        normalized = normalizer.normalize(test_data)
        
        # Mean should be close to 0
        mean = jnp.mean(normalized, axis=0)
        assert jnp.all(jnp.abs(mean) < 0.5), f"Mean too far from 0: {mean}"


# =============================================================================
# Saliency Pruning Tests
# =============================================================================

class TestSaliencyPruning:
    """Tests for saliency-based pruning."""
    
    def test_compute_saliency_shape(self):
        """Saliency should have same shape as params."""
        from src.jax.pruning.saliency import compute_saliency
        from src.jax.utils.types import Batch
        
        params = {"w": jnp.ones((10, 5)), "b": jnp.ones(5)}
        
        def loss_fn(params, batch):
            return jnp.sum(params["w"]) + jnp.sum(params["b"])
        
        batch = Batch(
            obs=jnp.ones((8, 10)),
            actions=jnp.ones((8, 4)),
            rewards=jnp.ones((8, 1)),
            next_obs=jnp.ones((8, 10)),
            dones=jnp.zeros((8, 1)),
        )
        
        saliency = compute_saliency(params, loss_fn, batch)
        
        assert saliency["w"].shape == params["w"].shape
        assert saliency["b"].shape == params["b"].shape
    
    def test_saliency_is_nonnegative(self):
        """Saliency scores should be non-negative (|w * g|)."""
        from src.jax.pruning.saliency import compute_saliency
        from src.jax.utils.types import Batch
        
        params = {"w": jax.random.normal(jax.random.PRNGKey(0), (10, 5))}
        
        def loss_fn(params, batch):
            return jnp.mean(params["w"] ** 2)
        
        batch = Batch(
            obs=jnp.ones((8, 10)),
            actions=jnp.ones((8, 4)),
            rewards=jnp.ones((8, 1)),
            next_obs=jnp.ones((8, 10)),
            dones=jnp.zeros((8, 1)),
        )
        
        saliency = compute_saliency(params, loss_fn, batch)
        
        assert jnp.all(saliency["w"] >= 0), "Saliency should be non-negative"
    
    def test_prune_by_saliency_sparsity(self):
        """Pruning should achieve target sparsity."""
        from src.jax.pruning.saliency import prune_by_saliency
        from src.jax.pruning.masks import compute_sparsity
        
        params = {"w": jnp.ones((100, 50))}
        saliency = {"w": jax.random.uniform(jax.random.PRNGKey(0), (100, 50))}
        
        mask, _ = prune_by_saliency(params, saliency, target_sparsity=0.8)
        
        actual_sparsity = compute_sparsity(mask)
        # Allow small tolerance due to threshold edge effects
        assert abs(actual_sparsity - 0.8) < 0.02, f"Sparsity {actual_sparsity:.2%} not close to 80%"
    
    def test_prune_keeps_high_saliency(self):
        """Pruning should keep high-saliency weights."""
        from src.jax.pruning.saliency import prune_by_saliency
        
        # Create saliency with clear high/low split
        saliency = {"w": jnp.array([10.0, 9.0, 1.0, 0.0])}
        params = {"w": jnp.ones(4)}
        
        mask, _ = prune_by_saliency(params, saliency, target_sparsity=0.5)
        
        # Top 2 saliency values should be kept
        assert mask["w"][0] == 1.0, "Highest saliency should be kept"
        assert mask["w"][1] == 1.0, "Second highest should be kept"


class TestSaliencyForSAC:
    """Tests for SAC-specific saliency computation."""
    
    def test_compute_saliency_for_sac_returns_valid(self):
        """Should return valid saliency scores for actor and critic."""
        from src.jax.pruning.saliency import compute_saliency_for_sac
        from src.jax.networks.actor import GaussianActor
        from src.jax.networks.critic import TwinQNetwork
        from src.jax.utils.types import Batch
        
        # Create networks
        actor = GaussianActor(act_dim=4, hidden_dims=(32,))
        critic = TwinQNetwork(hidden_dims=(32,))
        
        key = jax.random.PRNGKey(0)
        actor_params = actor.init(key, jnp.ones((1, 10)))
        critic_params = critic.init(key, jnp.ones((1, 10)), jnp.ones((1, 4)))
        
        batch = Batch(
            obs=jax.random.normal(key, (16, 10)),
            actions=jax.random.normal(key, (16, 4)),
            rewards=jnp.ones((16, 1)),
            next_obs=jax.random.normal(key, (16, 10)),
            dones=jnp.zeros((16, 1)),
        )
        
        actor_sal, critic_sal = compute_saliency_for_sac(
            actor_params, critic_params,
            actor.apply, critic.apply,
            batch, alpha=0.2
        )
        
        # Should return pytrees with same structure
        actor_leaves = jax.tree.leaves(actor_sal)
        critic_leaves = jax.tree.leaves(critic_sal)
        
        assert len(actor_leaves) > 0
        assert len(critic_leaves) > 0
        assert all(jnp.all(l >= 0) for l in actor_leaves)


# =============================================================================
# Checkpointing Tests
# =============================================================================

class TestCheckpointing:
    """Tests for checkpointing functionality."""
    
    def test_save_and_restore(self, tmp_path):
        """Should save and restore training state."""
        from src.jax.utils.checkpointing import Checkpointer
        from src.jax.training.train_state import create_sac_train_state
        
        # Create state
        state = create_sac_train_state(
            key=jax.random.PRNGKey(42),
            obs_dim=10,
            act_dim=4,
            hidden_dims=(32,),
        )
        
        # Save
        ckpt = Checkpointer(str(tmp_path))
        ckpt.save(state, step=100)
        
        # Create template for restore (fresh state)
        template = create_sac_train_state(
            key=jax.random.PRNGKey(0),  # Different key
            obs_dim=10,
            act_dim=4,
            hidden_dims=(32,),
        )
        
        # Restore
        restored = ckpt.restore(template, step=100)
        
        assert restored.step == 100
        # Check params were restored (should match original)
        orig_actor_flat = jax.tree.leaves(state.actor_params)
        rest_actor_flat = jax.tree.leaves(restored.actor_params)
        for o, r in zip(orig_actor_flat, rest_actor_flat):
            np.testing.assert_array_almost_equal(o, r)
    
    def test_save_and_load_masks(self, tmp_path):
        """Should save and load masks separately."""
        from src.jax.utils.checkpointing import Checkpointer
        from flax.core import freeze
        
        ckpt = Checkpointer(str(tmp_path))
        
        actor_mask = freeze({"params": {"layer": jnp.array([1.0, 0.0, 1.0])}})
        critic_mask = freeze({"params": {"layer": jnp.array([0.0, 1.0, 0.0])}})
        
        ckpt.save_masks(actor_mask, critic_mask, "test_task")
        
        loaded_actor, loaded_critic = ckpt.load_masks("test_task")
        
        np.testing.assert_array_equal(
            loaded_actor["params"]["layer"],
            actor_mask["params"]["layer"]
        )
        np.testing.assert_array_equal(
            loaded_critic["params"]["layer"],
            critic_mask["params"]["layer"]
        )
    
    def test_restore_latest(self, tmp_path):
        """Should restore 'latest' checkpoint when step=None."""
        from src.jax.utils.checkpointing import Checkpointer
        from src.jax.training.train_state import create_sac_train_state
        
        state = create_sac_train_state(
            key=jax.random.PRNGKey(42),
            obs_dim=5,
            act_dim=2,
            hidden_dims=(16,),
        )
        
        ckpt = Checkpointer(str(tmp_path))
        ckpt.save(state, step=200)
        
        template = create_sac_train_state(
            key=jax.random.PRNGKey(0),
            obs_dim=5,
            act_dim=2,
            hidden_dims=(16,),
        )
        
        # Restore without specifying step
        restored = ckpt.restore(template, step=None)
        assert restored.step == 200


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation utilities."""
    
    def test_compute_iqm_basic(self):
        """IQM should compute interquartile mean correctly."""
        from src.jax.training.evaluation import compute_iqm
        
        # Values from 1 to 100
        values = list(range(1, 101))
        
        iqm = compute_iqm(values, trim_fraction=0.25)
        
        # IQM of 1-100 with 25% trim should be mean of 26-75
        expected = np.mean(range(26, 76))
        assert abs(iqm - expected) < 1.0, f"IQM {iqm} not close to {expected}"
    
    def test_compute_iqm_empty(self):
        """IQM of empty list should be 0."""
        from src.jax.training.evaluation import compute_iqm
        
        assert compute_iqm([]) == 0.0
    
    def test_compute_auc_basic(self):
        """AUC should compute area under curve."""
        from src.jax.training.evaluation import compute_auc
        
        # Linear increase from 0 to 100 over 100 steps
        steps = list(range(0, 101, 10))
        values = list(range(0, 101, 10))
        
        auc = compute_auc(steps, values, normalize=True)
        
        # Normalized AUC of linear 0-100 should be ~50
        assert abs(auc - 50.0) < 1.0, f"AUC {auc} not close to 50"
    
    def test_compute_auc_unnormalized(self):
        """Unnormalized AUC should equal trapezoid area."""
        from src.jax.training.evaluation import compute_auc
        
        steps = [0, 10]
        values = [0, 10]
        
        auc = compute_auc(steps, values, normalize=False)
        
        # Trapezoid area: (0 + 10) / 2 * 10 = 50
        assert abs(auc - 50.0) < 0.01


# =============================================================================
# Multi-Task SAC Tests
# =============================================================================

class TestMultiTaskSAC:
    """Tests for Multi-Task SAC agent."""
    
    def test_mtsac_train_state_creation(self):
        """Should create valid multi-task training state."""
        from src.jax.agents.multi_task_sac import create_mtsac_train_state
        
        state = create_mtsac_train_state(
            key=jax.random.PRNGKey(0),
            obs_dim=10,
            act_dim=4,
            num_tasks=3,
            encoder_dims=(64,),
            head_dims=(32,),
        )
        
        assert state.step == 0
        assert state.actor_params is not None
        assert state.critic_params is not None
    
    def test_mtsac_agent_creation(self):
        """MTSACAgent should initialize without errors."""
        from src.jax.agents.multi_task_sac import MTSACAgent
        from src.jax.agents.sac import SACConfig
        
        config = SACConfig(hidden_dims=(32,))
        agent = MTSACAgent(
            obs_dim=10,
            act_dim=4,
            num_tasks=3,
            act_low=np.array([-1.0] * 4),
            act_high=np.array([1.0] * 4),
            config=config,
            seed=42,
        )
        
        assert agent.obs_dim == 10
        assert agent.act_dim == 4
        assert agent.num_tasks == 3
    
    def test_mtsac_action_selection(self):
        """MTSACAgent should select valid actions with task encoding."""
        from src.jax.agents.multi_task_sac import MTSACAgent
        from src.jax.agents.sac import SACConfig
        
        config = SACConfig(hidden_dims=(32,))
        agent = MTSACAgent(
            obs_dim=10,
            act_dim=4,
            num_tasks=3,
            act_low=np.array([-1.0] * 4),
            act_high=np.array([1.0] * 4),
            config=config,
            seed=42,
        )
        
        # Observation with task one-hot appended
        base_obs = np.random.randn(10).astype(np.float32)
        task_one_hot = np.array([1.0, 0.0, 0.0])  # Task 0
        obs = np.concatenate([base_obs, task_one_hot])
        
        action = agent.select_action(obs, eval_mode=False)
        
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_mtsac_update(self):
        """MTSACAgent should perform update step."""
        from src.jax.agents.multi_task_sac import MTSACAgent
        from src.jax.agents.sac import SACConfig
        from src.jax.buffers.replay_buffer import ReplayBuffer
        
        config = SACConfig(hidden_dims=(32,))
        num_tasks = 3
        obs_dim = 10
        act_dim = 4
        
        agent = MTSACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_tasks=num_tasks,
            act_low=np.array([-1.0] * act_dim),
            act_high=np.array([1.0] * act_dim),
            config=config,
            seed=42,
        )
        
        # Buffer stores task-augmented observations
        buffer = ReplayBuffer(
            obs_dim=obs_dim + num_tasks,
            act_dim=act_dim,
            max_size=1000
        )
        
        # Fill buffer
        for i in range(100):
            task_id = i % num_tasks
            task_one_hot = np.zeros(num_tasks)
            task_one_hot[task_id] = 1.0
            
            obs = np.concatenate([
                np.random.randn(obs_dim).astype(np.float32),
                task_one_hot
            ])
            next_obs = np.concatenate([
                np.random.randn(obs_dim).astype(np.float32),
                task_one_hot
            ])
            
            buffer.store(
                obs=obs,
                action=np.random.randn(act_dim).astype(np.float32),
                reward=np.random.randn(),
                next_obs=next_obs,
                done=float(np.random.rand() < 0.1),
            )
        
        # Perform update
        metrics = agent.update(buffer, batch_size=32)
        
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics


class TestSharedEncoderNetworks:
    """Tests for shared encoder multi-task networks."""
    
    def test_shared_encoder_actor_output(self):
        """SharedEncoderActor should produce valid outputs."""
        from src.jax.agents.multi_task_sac import SharedEncoderActor
        
        actor = SharedEncoderActor(
            act_dim=4,
            num_tasks=3,
            encoder_dims=(32,),
            head_dims=(16,),
        )
        
        params = actor.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 10)),  # obs
            jnp.ones((1, 3)),   # task_id
        )
        
        mean, log_std = actor.apply(params, jnp.ones((8, 10)), jnp.ones((8, 3)))
        
        assert mean.shape == (8, 4)
        assert log_std.shape == (8, 4)
    
    def test_shared_encoder_critic_output(self):
        """SharedEncoderCritic should produce twin Q-values."""
        from src.jax.agents.multi_task_sac import SharedEncoderCritic
        
        critic = SharedEncoderCritic(
            num_tasks=3,
            encoder_dims=(32,),
            head_dims=(16,),
        )
        
        params = critic.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 10)),  # obs
            jnp.ones((1, 4)),   # action
            jnp.ones((1, 3)),   # task_id
        )
        
        q1, q2 = critic.apply(
            params,
            jnp.ones((8, 10)),
            jnp.ones((8, 4)),
            jnp.ones((8, 3)),
        )
        
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)


# =============================================================================
# Iterative Pruning Tests
# =============================================================================

class TestIterativePruning:
    """Tests for iterative pruning functionality."""
    
    def test_iterative_pruning_achieves_target(self):
        """Iterative pruning should reach target sparsity."""
        from src.jax.pruning.saliency import iterative_pruning
        from src.jax.pruning.masks import compute_sparsity
        from src.jax.utils.types import Batch
        
        params = {"w": jax.random.normal(jax.random.PRNGKey(0), (50, 50))}
        
        def loss_fn(params, batch):
            return jnp.mean(params["w"] ** 2)
        
        batches = [
            Batch(
                obs=jnp.ones((8, 10)),
                actions=jnp.ones((8, 4)),
                rewards=jnp.ones((8, 1)),
                next_obs=jnp.ones((8, 10)),
                dones=jnp.zeros((8, 1)),
            )
            for _ in range(5)
        ]
        
        mask, _ = iterative_pruning(
            params, loss_fn, batches,
            target_sparsity=0.8,
            num_iterations=5,
        )
        
        sparsity = compute_sparsity(mask)
        assert abs(sparsity - 0.8) < 0.05, f"Sparsity {sparsity:.2%} not close to 80%"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
