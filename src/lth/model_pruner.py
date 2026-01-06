import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class SaliencyPruner:
    """
    Saliency-based pruning that uses gradient information to determine parameter importance.
    Parameters with low saliency (small gradient magnitude) are pruned first.
    """
    
    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.2,
        prune_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
    ):
        """
        Initialize the saliency-based pruner.
        
        Args:
            model: The neural network model to prune
            prune_ratio: Fraction of parameters to prune (0.0 to 1.0)
            prune_layers: List of layer names to prune. If None, prunes all linear layers
            exclude_layers: List of layer names to exclude from pruning
        """
        self.model = model
        self.prune_ratio = prune_ratio
        self.prune_layers = prune_layers
        self.exclude_layers = exclude_layers or []
        
        # Store masks for each parameter
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Store saliency scores
        self.saliency_scores: Dict[str, torch.Tensor] = {}
        
        # Initialize masks to all ones (no pruning initially)
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize pruning masks for all parameters."""
        for name, param in self.model.named_parameters():
            if self._should_prune_parameter(name):
                self.masks[name] = torch.ones_like(param.data)
    
    def _should_prune_parameter(self, param_name: str) -> bool:
        """Determine if a parameter should be pruned."""
        # Don't prune bias terms
        if 'bias' in param_name:
            return False
        
        # Check if in exclude list
        if any(exclude in param_name for exclude in self.exclude_layers):
            return False
        
        # If prune_layers is specified, only prune those
        if self.prune_layers is not None:
            return any(layer in param_name for layer in self.prune_layers)
        
        # By default, prune all weight parameters
        return 'weight' in param_name
    
    def compute_saliency(self, loss: torch.Tensor):
        """
        Compute saliency scores for all parameters based on gradient magnitude.
        Saliency = |gradient * weight|
        
        Args:
            loss: The loss tensor to compute gradients from
        """
        # Compute gradients
        loss.backward()
        
        # Compute saliency for each parameter
        for name, param in self.model.named_parameters():
            if self._should_prune_parameter(name) and param.grad is not None:
                # Saliency is the absolute value of gradient * weight
                saliency = torch.abs(param.grad * param.data)
                
                # Accumulate saliency scores (useful for averaging over multiple samples)
                if name in self.saliency_scores:
                    self.saliency_scores[name] += saliency.detach()
                else:
                    self.saliency_scores[name] = saliency.detach()
    
    def reset_saliency(self):
        """Reset accumulated saliency scores."""
        self.saliency_scores = {}
    
    def prune(self, normalize: bool = True):
        """
        Prune parameters based on accumulated saliency scores.
        
        Args:
            normalize: Whether to normalize saliency scores before pruning
        """
        if not self.saliency_scores:
            raise ValueError("No saliency scores computed. Call compute_saliency() first.")
        
        # Normalize saliency scores if requested
        if normalize:
            for name in self.saliency_scores:
                self.saliency_scores[name] = self.saliency_scores[name] / self.saliency_scores[name].max()
        
        # Collect all saliency values across all parameters
        all_scores = []
        for name in self.saliency_scores:
            all_scores.append(self.saliency_scores[name].flatten())
        
        all_scores = torch.cat(all_scores)
        
        # Determine the threshold for pruning
        k = int(len(all_scores) * self.prune_ratio)
        if k == 0:
            return
        
        threshold = torch.kthvalue(all_scores, k).values
        
        # Create masks based on threshold
        for name, param in self.model.named_parameters():
            if name in self.saliency_scores:
                # Prune parameters with saliency below threshold
                self.masks[name] = (self.saliency_scores[name] > threshold).float()
                
                # Apply mask to parameter
                param.data *= self.masks[name]
    
    def apply_masks(self):
        """Apply current masks to model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def get_sparsity(self) -> Dict[str, float]:
        """
        Calculate sparsity for each pruned layer.
        
        Returns:
            Dictionary mapping layer names to sparsity ratios
        """
        sparsity_info = {}
        total_params = 0
        total_pruned = 0
        
        for name, mask in self.masks.items():
            n_params = mask.numel()
            n_pruned = (mask == 0).sum().item()
            
            total_params += n_params
            total_pruned += n_pruned
            
            sparsity_info[name] = n_pruned / n_params if n_params > 0 else 0.0
        
        if total_params > 0:
            sparsity_info['overall'] = total_pruned / total_params
        
        return sparsity_info
    
    def iterative_prune(
        self, 
        dataloader, 
        n_iterations: int = 5,
        samples_per_iteration: int = 100,
    ):
        """
        Perform iterative pruning by computing saliency over multiple batches.
        
        Args:
            dataloader: DataLoader providing batches for saliency computation
            n_iterations: Number of pruning iterations
            samples_per_iteration: Number of samples to use for computing saliency
        """
        original_prune_ratio = self.prune_ratio
        
        for iteration in range(n_iterations):
            # Prune incrementally
            self.prune_ratio = original_prune_ratio / n_iterations
            
            # Reset and accumulate saliency
            self.reset_saliency()
            
            n_samples = 0
            for batch in dataloader:
                if n_samples >= samples_per_iteration:
                    break
                
                # Forward pass and compute saliency
                # (This is a placeholder - actual implementation depends on your training loop)
                # loss = your_loss_function(batch)
                # self.compute_saliency(loss)
                
                n_samples += batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            
            # Prune based on accumulated saliency
            self.prune(normalize=True)
            
            print(f"Iteration {iteration + 1}/{n_iterations} - Sparsity: {self.get_sparsity()['overall']:.4f}")


class MagnitudePruner:
    """
    Magnitude-based pruning that prunes parameters with smallest absolute values.
    Simpler than saliency-based pruning but doesn't require gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.2,
        prune_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
    ):
        """
        Initialize the magnitude-based pruner.
        
        Args:
            model: The neural network model to prune
            prune_ratio: Fraction of parameters to prune (0.0 to 1.0)
            prune_layers: List of layer names to prune. If None, prunes all linear layers
            exclude_layers: List of layer names to exclude from pruning
        """
        self.model = model
        self.prune_ratio = prune_ratio
        self.prune_layers = prune_layers
        self.exclude_layers = exclude_layers or []
        
        self.masks: Dict[str, torch.Tensor] = {}
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize pruning masks for all parameters."""
        for name, param in self.model.named_parameters():
            if self._should_prune_parameter(name):
                self.masks[name] = torch.ones_like(param.data)
    
    def _should_prune_parameter(self, param_name: str) -> bool:
        """Determine if a parameter should be pruned."""
        if 'bias' in param_name:
            return False
        
        if any(exclude in param_name for exclude in self.exclude_layers):
            return False
        
        if self.prune_layers is not None:
            return any(layer in param_name for layer in self.prune_layers)
        
        return 'weight' in param_name
    
    def prune(self):
        """Prune parameters based on magnitude (absolute value)."""
        # Collect all weight magnitudes
        all_weights = []
        for name, param in self.model.named_parameters():
            if self._should_prune_parameter(name):
                all_weights.append(torch.abs(param.data).flatten())
        
        all_weights = torch.cat(all_weights)
        
        # Determine threshold
        k = int(len(all_weights) * self.prune_ratio)
        if k == 0:
            return
        
        threshold = torch.kthvalue(all_weights, k).values
        
        # Create and apply masks
        for name, param in self.model.named_parameters():
            if self._should_prune_parameter(name):
                self.masks[name] = (torch.abs(param.data) > threshold).float()
                param.data *= self.masks[name]
    
    def apply_masks(self):
        """Apply current masks to model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def get_sparsity(self) -> Dict[str, float]:
        """Calculate sparsity for each pruned layer."""
        sparsity_info = {}
        total_params = 0
        total_pruned = 0
        
        for name, mask in self.masks.items():
            n_params = mask.numel()
            n_pruned = (mask == 0).sum().item()
            
            total_params += n_params
            total_pruned += n_pruned
            
            sparsity_info[name] = n_pruned / n_params if n_params > 0 else 0.0
        
        if total_params > 0:
            sparsity_info['overall'] = total_pruned / total_params
        
        return sparsity_info