# Standalone Python gear optimizer

This folder provides a CLI that reuses the in-app damage and GCD formulas to search for high-damage gearsets without the web UI.

The loader shells out to `node` to parse the existing `data/out/*.js` exports, so a Node runtime available in the repository is required.

## Key formulas

* **Damage scalar**: combines weapon damage, main stat scaling, Determination, Tenacity, trait multiplier, and the crit/direct-hit mixture (see `calculate_effects`). The computation mirrors `equippedEffects` from `src/stores/Store.ts`.
* **GCD**: calculated from the skill speed and optional job-specific GCD modifier in the same way as the frontend (also from `equippedEffects`).
* **Base stats**: derived from `BASE_STATS` and `LEVEL_MODIFIERS` using the same level coefficients that appear in `src/game.ts`.

## Usage

1. Copy `config.example.json` and adjust it to your job, level, and thresholds. The example keeps `allow_food` disabled and
   `max_items_per_slot` small so the sample search finishes quickly; widen them gradually.
2. Run the solver: `python tools/python_optimizer/optimizer.py --config path/to/config.json`.

The command prints a JSON summary with the chosen gear, food, final stats, damage value, and GCD. Limit `max_items_per_slot` in the config to keep the search tractable.
