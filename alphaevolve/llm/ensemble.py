"""LLM Ensemble for multi-model routing and quota management."""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RoutingStrategy(Enum):
    """Routing strategies for model selection."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PRIORITY_BASED = "priority_based"
    COST_AWARE = "cost_aware"


class LLMEnsemble:
    """Ensemble of LLM models with intelligent routing.

    Features:
    - Multiple model support
    - Priority-based routing
    - Quota management
    - Usage statistics per model
    - Automatic fallback on failure

    Usage:
        ensemble = LLMEnsemble()
        ensemble.register_client("claude_sonnet", sonnet_client)
        ensemble.register_client("claude_opus", opus_client)

        response = ensemble.generate(prompt, priority=Priority.HIGH)
    """

    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.PRIORITY_BASED,
        default_priority: Priority = Priority.NORMAL,
    ) -> None:
        """Initialize LLM ensemble.

        Args:
            routing_strategy: Strategy for model selection
            default_priority: Default priority for requests
        """
        self._clients: Dict[str, BaseLLMClient] = {}
        self._routing_strategy = routing_strategy
        self._default_priority = default_priority
        self._weights: Dict[str, float] = {}
        self._priority_models: Dict[Priority, str] = {}
        self._round_robin_index = 0
        self._usage_history: List[Dict[str, Any]] = []

        # Default priority routing
        self._priority_models = {
            Priority.LOW: None,
            Priority.NORMAL: None,
            Priority.HIGH: None,
            Priority.CRITICAL: None,
        }

    def register_client(
        self,
        name: str,
        client: BaseLLMClient,
        weight: float = 1.0,
    ) -> None:
        """Register an LLM client with the ensemble.

        Args:
            name: Unique identifier for the client
            client: LLM client instance
            weight: Weight for weighted routing (default 1.0)
        """
        self._clients[name] = client
        self._weights[name] = weight
        logger.info(f"Registered LLM client: {name} ({client.name})")

    def set_priority_model(self, priority: Priority, client_name: str) -> None:
        """Set which model to use for a specific priority level.

        Args:
            priority: Priority level
            client_name: Name of registered client
        """
        if client_name not in self._clients:
            raise ValueError(f"Unknown client: {client_name}")
        self._priority_models[priority] = client_name

    def set_routing_rule(
        self,
        strategy: RoutingStrategy,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Configure routing strategy.

        Args:
            strategy: New routing strategy
            weights: Optional weights for weighted routing
        """
        self._routing_strategy = strategy
        if weights:
            for name, weight in weights.items():
                if name in self._clients:
                    self._weights[name] = weight

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        priority: Optional[Priority] = None,
        preferred_model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using the ensemble.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            priority: Request priority (uses default if not specified)
            preferred_model: Specific model to use (bypasses routing)
            **kwargs: Additional arguments for LLM client

        Returns:
            LLMResponse from selected model
        """
        priority = priority or self._default_priority

        # Select model based on routing strategy
        if preferred_model:
            selected = preferred_model
        else:
            selected = self._select_model(priority)

        if not selected:
            raise RuntimeError("No model available for request")

        client = self._clients[selected]

        try:
            response = client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

            # Record usage
            self._record_usage(selected, response)

            return response

        except Exception as e:
            logger.warning(f"Model {selected} failed: {e}, attempting fallback")
            return self._fallback_generate(
                prompt, system_prompt, excluded=[selected], **kwargs
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble usage statistics.

        Returns:
            Dictionary with per-model and aggregate statistics
        """
        stats = {
            "models": {},
            "total_requests": sum(
                c.get_usage().get("total_requests", 0)
                for c in self._clients.values()
            ),
            "routing_strategy": self._routing_strategy.value,
        }

        for name, client in self._clients.items():
            usage = client.get_usage()
            stats["models"][name] = {
                **usage,
                "weight": self._weights.get(name, 1.0),
                "priority": self._get_priority_for_model(name),
            }

        return stats

    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        for client in self._clients.values():
            client.reset_usage()
        self._usage_history.clear()

    def list_models(self) -> List[Dict[str, str]]:
        """List all registered models.

        Returns:
            List of dicts with model info
        """
        return [
            {
                "name": name,
                "provider": client.name,
                "model": client.config.model if hasattr(client, "config") else "unknown",
            }
            for name, client in self._clients.items()
        ]

    def _select_model(self, priority: Priority) -> Optional[str]:
        """Select model based on routing strategy.

        Args:
            priority: Request priority

        Returns:
            Selected model name or None
        """
        if not self._clients:
            return None

        # Check priority-based routing first
        if self._priority_models.get(priority):
            return self._priority_models[priority]

        # Apply routing strategy
        if self._routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select()
        elif self._routing_strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_select()
        elif self._routing_strategy == RoutingStrategy.PRIORITY_BASED:
            # Default: use highest weighted model for high priority
            return max(self._weights, key=self._weights.get) if self._weights else list(self._clients.keys())[0]
        else:
            return list(self._clients.keys())[0]

    def _round_robin_select(self) -> str:
        """Round-robin model selection."""
        names = list(self._clients.keys())
        selected = names[self._round_robin_index % len(names)]
        self._round_robin_index += 1
        return selected

    def _weighted_select(self) -> str:
        """Weighted random model selection."""
        import random
        names = list(self._weights.keys())
        weights = [self._weights.get(n, 1.0) for n in names]
        return random.choices(names, weights=weights)[0]

    def _get_priority_for_model(self, name: str) -> Optional[str]:
        """Get priority level associated with a model."""
        for priority, model_name in self._priority_models.items():
            if model_name == name:
                return priority.value
        return None

    def _fallback_generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        excluded: List[str],
        **kwargs: Any,
    ) -> LLMResponse:
        """Try alternative models when primary fails.

        Args:
            prompt: User prompt
            system_prompt: System instruction
            excluded: Models to skip
            **kwargs: Additional arguments

        Returns:
            LLMResponse from fallback model
        """
        available = [n for n in self._clients.keys() if n not in excluded]

        if not available:
            raise RuntimeError("No fallback models available")

        # Try each available model
        for model_name in available:
            try:
                client = self._clients[model_name]
                response = client.generate(prompt, system_prompt, **kwargs)
                self._record_usage(model_name, response)
                logger.info(f"Fallback successful with model: {model_name}")
                return response
            except Exception as e:
                logger.warning(f"Fallback model {model_name} also failed: {e}")
                continue

        raise RuntimeError("All fallback models failed")

    def _record_usage(self, model_name: str, response: LLMResponse) -> None:
        """Record usage for tracking."""
        self._usage_history.append({
            "model": model_name,
            "tokens": response.total_tokens,
            "timestamp": time.time() if "time" in dir() else 0,
        })


# Import time at module level
import time
