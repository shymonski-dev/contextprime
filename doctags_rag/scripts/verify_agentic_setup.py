"""
Verification script for agentic system installation.

Checks:
- All modules can be imported
- Dependencies are installed
- Basic functionality works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check that all modules can be imported."""
    print("Checking imports...")

    try:
        from src.agents import (
            BaseAgent, AgentRole, AgentMessage, AgentState,
            PlanningAgent, QueryPlan, PlanStep,
            ExecutionAgent, ExecutionResult,
            EvaluationAgent, QualityAssessment,
            LearningAgent, LearningMetrics,
            AgentCoordinator, CoordinationResult,
            FeedbackAggregator, AggregatedFeedback,
            RLModule, RewardSignal,
            MemorySystem, ShortTermMemory, LongTermMemory,
            PerformanceMonitor, PerformanceMetrics,
            AgenticPipeline, AgenticMode
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_basic_functionality():
    """Check basic functionality."""
    print("\nChecking basic functionality...")

    try:
        from src.agents import PlanningAgent, ExecutionAgent, EvaluationAgent

        # Create agents
        planner = PlanningAgent(agent_id="test_planner")
        executor = ExecutionAgent(agent_id="test_executor")
        evaluator = EvaluationAgent(agent_id="test_evaluator")

        print("  ✓ Agents created successfully")

        # Check state
        assert planner.state.value == "idle"
        assert executor.state.value == "idle"
        assert evaluator.state.value == "idle"

        print("  ✓ Agent states correct")

        # Check capabilities
        assert len(planner.capabilities) > 0
        assert len(executor.capabilities) > 0
        assert len(evaluator.capabilities) > 0

        print("  ✓ Agent capabilities defined")

        return True

    except Exception as e:
        print(f"  ✗ Functionality check failed: {e}")
        return False


def check_async_functionality():
    """Check async functionality."""
    print("\nChecking async functionality...")

    try:
        import asyncio
        from src.agents import PlanningAgent

        async def test_async():
            planner = PlanningAgent()
            plan = await planner.create_plan("Test query")
            return plan is not None

        result = asyncio.run(test_async())

        if result:
            print("  ✓ Async operations working")
            return True
        else:
            print("  ✗ Async test failed")
            return False

    except Exception as e:
        print(f"  ✗ Async check failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("  Agentic System Verification")
    print("=" * 60)

    checks = [
        ("Import Check", check_imports),
        ("Basic Functionality", check_basic_functionality),
        ("Async Functionality", check_async_functionality),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ Check crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    all_passed = all(success for _, success in results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print()

    if all_passed:
        print("✓ All checks passed! The agentic system is ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: pytest tests/test_agents.py")
        print("  3. Run demo: python scripts/demo_agentic.py")
        return 0
    else:
        print("✗ Some checks failed. Please install dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
