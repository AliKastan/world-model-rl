"""Test script for the affordance analysis system.

- Generates a Level 5 puzzle (key + door + box)
- Runs ContextualAffordance.analyze_scene()
- Prints rich formatted analysis
- Generates training data and pre-trains AffordanceNet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env.level_generator import LevelGenerator
from agents.model_based.affordance import (
    AffordanceNet,
    ContextualAffordance,
    generate_training_data,
    pretrain,
)


def test_scene_analysis() -> None:
    """Analyse a Level 5 puzzle and print results."""
    gen = LevelGenerator()

    # Generate a key+door+box level
    world = gen.generate(difficulty=5, seed=5000)
    opt_steps = getattr(world, "_optimal_steps", "?")

    print("=" * 60)
    print("SCENE ANALYSIS -- Difficulty 5")
    print("=" * 60)
    print(f"\nGrid ({world.width}x{world.height}), optimal={opt_steps} steps:")
    print(world.render_ascii())

    analyser = ContextualAffordance()
    analysis = analyser.analyze_scene(world)

    # Objects
    print(f"\n--- Objects ({len(analysis.objects)}) ---")
    icon_map = {
        "box": "[Box]", "key": "[Key]", "door": "[Door]",
        "target": "[Tgt]", "ice": "[Ice]", "switch": "[Sw]",
        "switch_wall": "[SW]",
    }
    for obj in analysis.objects:
        icon = icon_map.get(obj.obj_type, "[?]")
        a = obj.affordance
        line = (
            f"  {icon} {obj.obj_type}@{obj.pos}: "
            f"interact={a.can_interact:.2f}  "
        )
        if a.push_score > 0.01:
            line += f"push={a.push_score:.2f}  "
        if a.collect_score > 0.01:
            line += f"collect={a.collect_score:.2f}  "
        line += f"risk={a.risk_score:.2f}  utility={a.utility_score:.2f}"
        if a.prerequisite_keys:
            line += f"  (needs key {a.prerequisite_keys})"
        if obj.extra.get("on_target"):
            line += "  [ON TARGET]"
        if obj.extra.get("locked"):
            line += f"  [LOCKED, door_id={obj.extra['door_id']}]"
        if obj.extra.get("key_id") is not None and obj.obj_type == "key":
            line += f"  [key_id={obj.extra['key_id']}]"
        print(line)

    # Relationships
    print(f"\n--- Relationships ({len(analysis.relationships)}) ---")
    for rel in analysis.relationships:
        print(f"  [{rel.relation}] {rel.description}  (score={rel.score:.2f})")

    # Required sequence
    if analysis.required_sequence:
        print(f"\n--- Required Sequence ---")
        for i, step in enumerate(analysis.required_sequence):
            print(f"  {i + 1}) {step}")

    # Suggested actions
    print(f"\n--- Priority Actions ---")
    for action in analysis.suggested_actions:
        print(f"  {action}")

    # Danger zones
    print(f"\n--- Danger Zones ({len(analysis.danger_zones)}) ---")
    if analysis.danger_zones:
        zones_str = ", ".join(f"({x},{y})" for x, y in analysis.danger_zones[:15])
        if len(analysis.danger_zones) > 15:
            zones_str += f" ... (+{len(analysis.danger_zones) - 15} more)"
        print(f"  Pushing a box here = deadlock: {zones_str}")
    else:
        print("  None detected")


def test_pretrain() -> None:
    """Generate training data and pre-train AffordanceNet."""
    print(f"\n{'=' * 60}")
    print("AFFORDANCE NET PRE-TRAINING")
    print("=" * 60)

    print("\n  Generating training data (1000 samples)...")
    dataset = generate_training_data(n_samples=1000, seed=123)
    print(f"  Generated {len(dataset)} samples")

    # Show label distribution
    import numpy as np
    labels = np.array([
        [s.can_interact, s.push_score, s.collect_score, s.risk_score, s.utility_score]
        for s in dataset
    ])
    names = ["interact", "push", "collect", "risk", "utility"]
    print("\n  Label means:")
    for i, name in enumerate(names):
        print(f"    {name:>10s}: {labels[:, i].mean():.3f}")

    print("\n  Training AffordanceNet (50 epochs)...")
    net = AffordanceNet()
    history = pretrain(net, dataset, epochs=50, lr=0.001, verbose=True)

    print(f"\n  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final accuracy: {history['accuracy'][-1]:.2f}")

    # Test the trained net on a scene
    print("\n  Testing trained net on a Level 3 puzzle...")
    gen = LevelGenerator()
    world = gen.generate(difficulty=3, seed=12345)
    print(world.render_ascii())

    analyser = ContextualAffordance(net=net)
    analysis = analyser.analyze_scene(world)

    print(f"\n  Objects with neural affordance (blended):")
    for obj in analysis.objects:
        a = obj.affordance
        print(
            f"    {obj.obj_type}@{obj.pos}: "
            f"interact={a.can_interact:.2f} "
            f"risk={a.risk_score:.2f} "
            f"utility={a.utility_score:.2f}"
        )


def main() -> None:
    test_scene_analysis()
    test_pretrain()
    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
