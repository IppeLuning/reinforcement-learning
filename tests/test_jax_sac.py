"""Unit tests for JAX SAC implementation.

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
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
