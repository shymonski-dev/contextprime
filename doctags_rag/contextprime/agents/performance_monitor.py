"""
Performance Monitor for tracking and optimizing system performance.

Monitors:
- Real-time metrics (latency, throughput)
- Agent utilization
- Resource usage
- Trend analysis
- Anomaly detection
- Automatic optimization triggers
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Snapshot of performance metrics."""
    timestamp: datetime
    latency_ms: float
    throughput_qps: float  # queries per second
    success_rate: float
    cache_hit_rate: float
    agent_utilization: Dict[str, float]
    memory_usage_mb: float
    error_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    severity: str  # info, warning, critical
    metric_name: str
    threshold: float
    actual_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Monitor and optimize system performance.

    Tracks metrics, detects anomalies, and triggers optimizations.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of sliding window for metrics
            alert_thresholds: Thresholds for alerting
        """
        self.window_size = window_size

        # Metric history
        self.latency_history: deque = deque(maxlen=window_size)
        self.throughput_history: deque = deque(maxlen=window_size)
        self.success_history: deque = deque(maxlen=window_size)
        self.cache_hit_history: deque = deque(maxlen=window_size)

        # Real-time counters
        self.query_count = 0
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Timing
        self.start_time = time.time()
        self.last_metric_time = time.time()

        # Alerts
        self.alerts: List[Alert] = []
        self.alert_thresholds = alert_thresholds or {
            "latency_ms": 5000,
            "error_rate": 0.1,
            "cache_hit_rate": 0.3
        }

        # Agent metrics
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "action_count": 0,
                "total_time_ms": 0.0,
                "errors": 0
            }
        )

    def record_query(
        self,
        latency_ms: float,
        success: bool,
        cache_hit: bool,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Record a query execution.

        Args:
            latency_ms: Query latency
            success: Whether query succeeded
            cache_hit: Whether result was cached
            agent_id: ID of agent that handled query
        """
        self.query_count += 1

        # Record success/error
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Record cache hit/miss
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Update history
        self.latency_history.append(latency_ms)
        self.success_history.append(1.0 if success else 0.0)
        self.cache_hit_history.append(1.0 if cache_hit else 0.0)

        # Calculate throughput
        elapsed = time.time() - self.last_metric_time
        if elapsed >= 1.0:  # Update every second
            qps = len(self.latency_history) / elapsed
            self.throughput_history.append(qps)
            self.last_metric_time = time.time()

        # Record agent metrics
        if agent_id:
            self.agent_metrics[agent_id]["action_count"] += 1
            self.agent_metrics[agent_id]["total_time_ms"] += latency_ms
            if not success:
                self.agent_metrics[agent_id]["errors"] += 1

        # Check thresholds
        self._check_thresholds(latency_ms)

    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            Current metrics snapshot
        """
        # Calculate averages
        avg_latency = (
            np.mean(self.latency_history)
            if self.latency_history else 0.0
        )

        avg_throughput = (
            np.mean(self.throughput_history)
            if self.throughput_history else 0.0
        )

        success_rate = (
            self.success_count / self.query_count
            if self.query_count > 0 else 0.0
        )

        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )

        error_rate = (
            self.error_count / self.query_count
            if self.query_count > 0 else 0.0
        )

        # Calculate agent utilization
        agent_utilization = {}
        total_time = sum(
            metrics["total_time_ms"]
            for metrics in self.agent_metrics.values()
        )

        for agent_id, metrics in self.agent_metrics.items():
            if total_time > 0:
                utilization = metrics["total_time_ms"] / total_time
                agent_utilization[agent_id] = utilization

        return PerformanceMetrics(
            timestamp=datetime.now(),
            latency_ms=avg_latency,
            throughput_qps=avg_throughput,
            success_rate=success_rate,
            cache_hit_rate=cache_hit_rate,
            agent_utilization=agent_utilization,
            memory_usage_mb=0.0,  # Would integrate with actual memory tracking
            error_rate=error_rate
        )

    def get_trends(
        self,
        time_window_minutes: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance trends.

        Args:
            time_window_minutes: Time window for trend analysis

        Returns:
            Trend analysis
        """
        metrics = self.get_current_metrics()

        # Calculate trends (simplified)
        recent_latencies = list(self.latency_history)
        if len(recent_latencies) >= 2:
            early_avg = np.mean(recent_latencies[:len(recent_latencies)//2])
            late_avg = np.mean(recent_latencies[len(recent_latencies)//2:])
            latency_trend = "improving" if late_avg < early_avg else "degrading"
        else:
            latency_trend = "stable"

        return {
            "time_window_minutes": time_window_minutes,
            "latency_trend": latency_trend,
            "latency_p50": float(np.percentile(recent_latencies, 50)) if recent_latencies else 0,
            "latency_p95": float(np.percentile(recent_latencies, 95)) if recent_latencies else 0,
            "latency_p99": float(np.percentile(recent_latencies, 99)) if recent_latencies else 0,
            "current_metrics": metrics.__dict__
        }

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies.

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if len(self.latency_history) < 10:
            return anomalies

        # Detect latency spikes
        recent_latencies = list(self.latency_history)
        mean_latency = np.mean(recent_latencies)
        std_latency = np.std(recent_latencies)

        for i, latency in enumerate(recent_latencies[-10:]):
            if latency > mean_latency + 3 * std_latency:
                anomalies.append({
                    "type": "latency_spike",
                    "value": latency,
                    "threshold": mean_latency + 3 * std_latency,
                    "timestamp": datetime.now()
                })

        # Detect throughput drops
        if len(self.throughput_history) >= 10:
            recent_qps = list(self.throughput_history)
            mean_qps = np.mean(recent_qps)
            if recent_qps[-1] < mean_qps * 0.5:
                anomalies.append({
                    "type": "throughput_drop",
                    "value": recent_qps[-1],
                    "expected": mean_qps,
                    "timestamp": datetime.now()
                })

        # Detect error rate spikes
        if len(self.success_history) >= 10:
            recent_errors = [1 - s for s in list(self.success_history)[-10:]]
            error_rate = np.mean(recent_errors)
            if error_rate > 0.2:
                anomalies.append({
                    "type": "high_error_rate",
                    "value": error_rate,
                    "threshold": 0.2,
                    "timestamp": datetime.now()
                })

        return anomalies

    def _check_thresholds(self, latency_ms: float) -> None:
        """
        Check if metrics exceed thresholds and create alerts.

        Args:
            latency_ms: Current query latency
        """
        # Check latency threshold
        if latency_ms > self.alert_thresholds["latency_ms"]:
            alert = Alert(
                alert_id=f"alert_{len(self.alerts)}",
                severity="warning",
                metric_name="latency_ms",
                threshold=self.alert_thresholds["latency_ms"],
                actual_value=latency_ms,
                message=f"High latency detected: {latency_ms:.0f}ms"
            )
            self.alerts.append(alert)
            logger.warning(alert.message)

        # Check error rate
        error_rate = self.error_count / self.query_count if self.query_count > 0 else 0
        if error_rate > self.alert_thresholds["error_rate"]:
            alert = Alert(
                alert_id=f"alert_{len(self.alerts)}",
                severity="critical",
                metric_name="error_rate",
                threshold=self.alert_thresholds["error_rate"],
                actual_value=error_rate,
                message=f"High error rate: {error_rate:.1%}"
            )
            self.alerts.append(alert)
            logger.error(alert.message)

        # Check cache hit rate
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 1.0
        )
        if cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alert = Alert(
                alert_id=f"alert_{len(self.alerts)}",
                severity="info",
                metric_name="cache_hit_rate",
                threshold=self.alert_thresholds["cache_hit_rate"],
                actual_value=cache_hit_rate,
                message=f"Low cache hit rate: {cache_hit_rate:.1%}"
            )
            self.alerts.append(alert)

    def get_optimization_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on metrics.

        Returns:
            List of recommendations
        """
        recommendations = []
        metrics = self.get_current_metrics()

        # Latency recommendations
        if metrics.latency_ms > 3000:
            recommendations.append("Enable aggressive caching to reduce latency")
            recommendations.append("Consider parallel query execution")

        # Cache recommendations
        if metrics.cache_hit_rate < 0.4:
            recommendations.append("Increase cache TTL or capacity")
            recommendations.append("Review cache key strategy")

        # Error rate recommendations
        if metrics.error_rate > 0.1:
            recommendations.append("Investigate error sources")
            recommendations.append("Implement retry logic with backoff")

        # Agent utilization recommendations
        if metrics.agent_utilization:
            max_util = max(metrics.agent_utilization.values())
            min_util = min(metrics.agent_utilization.values())

            if max_util > 0.8:
                recommendations.append("Scale up overloaded agents")
            if max_util - min_util > 0.5:
                recommendations.append("Rebalance agent workloads")

        return recommendations

    def reset_counters(self) -> None:
        """Reset all counters and history."""
        self.latency_history.clear()
        self.throughput_history.clear()
        self.success_history.clear()
        self.cache_hit_history.clear()

        self.query_count = 0
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self.start_time = time.time()
        self.last_metric_time = time.time()

        self.agent_metrics.clear()
        self.alerts.clear()

        logger.info("Performance monitor counters reset")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Performance summary
        """
        metrics = self.get_current_metrics()
        trends = self.get_trends()
        anomalies = self.detect_anomalies()
        recommendations = self.get_optimization_recommendations()

        return {
            "current_metrics": metrics.__dict__,
            "trends": trends,
            "anomalies": anomalies,
            "recommendations": recommendations,
            "alerts": [
                {
                    "severity": alert.severity,
                    "metric": alert.metric_name,
                    "message": alert.message
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            "uptime_seconds": time.time() - self.start_time,
            "total_queries": self.query_count
        }
