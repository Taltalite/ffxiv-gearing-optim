"""Standalone gear optimizer using FFXIV gearing formulas.

This script mirrors the in-app damage and GCD calculations (see src/stores/Store.ts)
so that users can search for high-damage gearsets outside the web UI.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Formulas and constants reproduced from src/stores/Store.ts and src/game.ts.
LEVEL_MODIFIERS: Dict[int, Dict[str, float]] = {
    50: {
        "main": 202,
        "sub": 341,
        "div": 341,
        "det": 202,
        "detTrunc": 5,
        "ap": 75,
        "apTank": 53,
        "hp": 14,
        "vit": 11.2,
        "vitTank": 15.5,
    },
    60: {
        "main": 218,
        "sub": 354,
        "div": 600,
        "det": 600,
        "detTrunc": 2,
        "ap": 100,
        "apTank": 78,
        "hp": 15,
        "vit": 12.9,
        "vitTank": 17.5,
    },
    70: {
        "main": 292,
        "sub": 364,
        "div": 900,
        "det": 900,
        "detTrunc": 1,
        "ap": 125,
        "apTank": 105,
        "hp": 17,
        "vit": 14.0,
        "vitTank": 18.8,
    },
    80: {
        "main": 340,
        "sub": 380,
        "div": 1300,
        "det": 1300,
        "detTrunc": 1,
        "ap": 165,
        "apTank": 115,
        "hp": 20,
        "vit": 18.8,
        "vitTank": 26.6,
    },
    90: {
        "main": 390,
        "sub": 400,
        "div": 1900,
        "det": 1900,
        "detTrunc": 1,
        "ap": 195,
        "apTank": 156,
        "hp": 30,
        "vit": 24.3,
        "vitTank": 34.6,
    },
    100: {
        "main": 440,
        "sub": 420,
        "div": 2780,
        "det": 2780,
        "detTrunc": 1,
        "ap": 237,
        "apTank": 190,
        "hp": 40,
        "vit": 30.1,
        "vitTank": 43.0,
    },
}

BASE_STATS: Dict[str, object] = {
    "STR": "main",
    "DEX": "main",
    "INT": "main",
    "MND": "main",
    "VIT": "main",
    "CRT": "sub",
    "DHT": "sub",
    "DET": "main",
    "SKS": "sub",
    "SPS": "sub",
    "TEN": "sub",
    "PIE": "main",
    "CMS": 0,
    "CRL": 0,
    "CP": 180,
    "GTH": 0,
    "PCP": 0,
    "GP": 400,
    "PDMG": 0,
    "MDMG": 0,
}

MATERIAS: Dict[str, List[int]] = {
    "PIE": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "DHT": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "CRT": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "DET": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "SPS": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "SKS": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "TEN": [1, 2, 3, 4, 6, 16, 8, 24, 12, 36, 18, 54],
    "CMS": [3, 4, 5, 6, 11, 16, 14, 21, 18, 27, 22, 33],
    "CRL": [1, 2, 3, 4, 7, 10, 9, 13, 12, 18, 15, 23],
    "CP": [1, 2, 3, 4, 6, 8, 7, 9, 8, 10, 9, 11],
    "GTH": [3, 4, 5, 6, 10, 15, 12, 20, 14, 25, 20, 36],
    "PCP": [3, 4, 5, 6, 10, 15, 12, 20, 14, 25, 20, 36],
    "GP": [1, 2, 3, 4, 6, 8, 7, 9, 8, 10, 9, 11],
}

MATERIA_GRADES: List[int] = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
MATERIA_GRADE_REQUIRED_LEVELS: List[int] = [
    15,
    30,
    45,
    70,
    160,
    290,
    420,
    420,
    560,
    560,
    690,
    690,
]
MATERIA_GRADE_IS_RESTRICTED: List[bool] = [False] * len(MATERIA_GRADE_REQUIRED_LEVELS)
for restricted_grade in [12, 10, 8, 6]:
    MATERIA_GRADE_IS_RESTRICTED[restricted_grade - 1] = True

StatDict = Dict[str, float]
MateriaChoice = Tuple[str, int]


def load_js_export(path: Path) -> Any:
    """Load an ``export default`` JS module using Node and parse as JSON."""
    module_path = path.resolve().as_uri()
    script = "import(process.argv[1]).then(m=>console.log(JSON.stringify(m.default ?? m)))"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, module_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def load_cap_tables(config: Dict[str, Any]) -> Dict[str, Any]:
    data_paths = config["data_paths"]
    level_caps = load_js_export(Path(data_paths.get("level_caps", "data/out/levelCaps.js")))
    slot_caps = load_js_export(Path(data_paths.get("slot_caps", "data/out/slotCaps.js")))
    role_caps = load_js_export(Path(data_paths.get("role_caps", "data/out/roleCaps.js")))

    level_index: Dict[int, int] = {}
    for i, level in enumerate(level_caps["level"]):
        level_index[int(level)] = i

    return {
        "level_caps": level_caps,
        "slot_caps": slot_caps,
        "role_caps": role_caps,
        "level_index": level_index,
    }


def get_caps(gear: Dict[str, Any], cap_tables: Dict[str, Any], sync_level: Optional[int] = None) -> StatDict:
    level = sync_level or int(gear.get("level", 0))
    slot = int(gear.get("rawSlot", gear.get("slot", 0)))
    role = int(gear.get("role", 0))
    level_caps: Dict[str, List[float]] = cap_tables["level_caps"]
    slot_caps: Dict[str, List[float]] = cap_tables["slot_caps"]
    role_caps: Dict[str, List[float]] = cap_tables["role_caps"]
    level_index: Dict[int, int] = cap_tables["level_index"]

    index = level_index[level]
    caps: StatDict = {}
    for stat, values in level_caps.items():
        if stat == "level":
            continue
        if stat == "DLY":
            caps[stat] = math.inf
            continue
        caps[stat] = round(values[index] * slot_caps[stat][slot] * role_caps[stat][role] / 100000)
    return caps


def build_base_stats(
    schema_stats: Sequence[str],
    level_mod: Dict[str, float],
    stat_modifiers: StatDict,
    clan_adjustments: Optional[StatDict],
) -> StatDict:
    stats: StatDict = {"PDMG": 0, "MDMG": 0}
    for stat in schema_stats:
        base = BASE_STATS.get(stat, 0)
        if isinstance(base, (int, float)):
            stats[stat] = float(base)
        elif base == "main":
            stats[stat] = math.floor(level_mod["main"] * (stat_modifiers.get(stat, 100) / 100))
        elif base == "sub":
            stats[stat] = math.floor(level_mod["sub"] * (stat_modifiers.get(stat, 100) / 100))
    if clan_adjustments:
        for stat, delta in clan_adjustments.items():
            stats[stat] = stats.get(stat, 0) + delta
    return stats


def apply_food(stats: StatDict, food: Optional[Dict[str, Any]]) -> StatDict:
    if food is None:
        return stats
    boosted = dict(stats)
    stat_rates: StatDict = food.get("statRates", {})
    for stat, bonus in food.get("stats", {}).items():
        current = stats.get(stat, 0)
        rate_cap = stat_rates.get(stat, math.inf)
        if math.isinf(rate_cap):
            boosted[stat] = current + bonus
        else:
            boosted[stat] = current + min(bonus, math.floor(current * rate_cap / 100))
    return boosted


def best_materia_grade_value(gear_level: int, stat: str, *, allow_restricted: bool) -> Optional[Tuple[int, int]]:
    values = MATERIAS.get(stat)
    if values is None:
        return None
    for grade in MATERIA_GRADES:
        if gear_level < MATERIA_GRADE_REQUIRED_LEVELS[grade - 1]:
            continue
        if not allow_restricted and MATERIA_GRADE_IS_RESTRICTED[grade - 1]:
            continue
        return grade, values[grade - 1]
    return None


def calculate_effects(
    stats: StatDict,
    level_mod: Dict[str, float],
    *,
    stat_modifiers: StatDict,
    main_stat: str,
    trait_damage_multiplier: float,
    party_bonus: float,
    job_level: int,
) -> Dict[str, float]:
    sub = level_mod["sub"]
    main = level_mod["main"]
    div = level_mod["div"]
    det = level_mod["det"]
    det_trunc = level_mod["detTrunc"]

    attack_main_stat = "STR" if main_stat == "VIT" else main_stat
    weapon_damage = math.floor(main * (stat_modifiers.get(attack_main_stat, 100) / 1000))
    if attack_main_stat in {"INT", "MND"}:
        weapon_damage += stats.get("MDMG", 0)
    else:
        weapon_damage += stats.get("PDMG", 0)

    main_damage = math.floor(
        (level_mod["apTank"] if main_stat == "VIT" else level_mod["ap"])
        * (math.floor(stats.get(attack_main_stat, 0) * party_bonus) - main)
        / main
        + 100
    ) / 100

    crt_chance = math.floor(200 * (stats.get("CRT", 0) - sub) / div + 50) / 1000
    crt_damage = math.floor(200 * (stats.get("CRT", 0) - sub) / div + 1400) / 1000
    det_damage = math.floor((140 * (stats.get("DET", 0) - main) / det + 1000) / det_trunc) * det_trunc / 1000
    dht_chance = math.floor(550 * (stats.get("DHT", 0) - sub) / div) / 1000
    ten_damage = math.floor(112 * (stats.get("TEN", 0) - sub) / div + 1000) / 1000

    damage = (
        0.01
        * weapon_damage
        * main_damage
        * det_damage
        * ten_damage
        * trait_damage_multiplier
        * ((crt_damage - 1) * crt_chance + 1)
        * (0.25 * dht_chance + 1)
    )

    gcd_modifier = stat_modifiers.get("gcd", 100)
    gcd = math.floor(
        math.floor((1000 - math.floor(130 * (stats.get("SKS", 0) - sub) / div)) * 2500 / 1000)
        * (gcd_modifier if job_level >= 80 else 100)
        / 1000
    ) / 100

    return {
        "damage": damage,
        "gcd": gcd,
        "crt_chance": crt_chance,
        "crt_damage": crt_damage,
        "dht_chance": dht_chance,
        "det_damage": det_damage,
        "ten_damage": ten_damage,
    }


def iter_gear_files(config: Dict[str, Any]) -> Iterable[Path]:
    paths = set()
    for path in glob.glob(config["data_paths"]["gear_glob"]):
        paths.add(Path(path))
    for extra in config["data_paths"].get("extra_gears", []):
        paths.add(Path(extra))
    return sorted(paths)


def load_gears(config: Dict[str, Any], job_categories: List[Optional[Dict[str, bool]]]) -> List[Dict[str, Any]]:
    gears: List[Dict[str, Any]] = []
    job = config["job"]
    for path in iter_gear_files(config):
        for gear in load_js_export(path):
            if gear is None:
                continue
            category = job_categories[gear["jobCategory"]]
            if not category or not category.get(job, False):
                continue
            if not config.get("include_obsolete", False) and gear.get("obsolete"):
                continue
            if gear.get("level", 0) < config.get("min_item_level", 0):
                continue
            if gear.get("level", 0) > config.get("max_item_level", 9999):
                continue
            gears.append(gear)
    return gears


def group_by_slot(gears: List[Dict[str, Any]], limit: int) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for gear in gears:
        grouped.setdefault(gear["slot"], []).append(gear)
    for slot, items in grouped.items():
        items.sort(key=lambda g: (g.get("level", 0), g.get("rarity", 0)), reverse=True)
        grouped[slot] = items[:limit]
    return grouped


def summarize_stats(selection: Sequence[Dict[str, Any]], base_stats: StatDict) -> StatDict:
    stats = dict(base_stats)
    for gear in selection:
        for stat, value in gear.get("stats", {}).items():
            stats[stat] = stats.get(stat, 0) + value
    return stats


def build_materia_slots(gear: Dict[str, Any]) -> int:
    base_slots = int(gear.get("materiaSlot", 0))
    if gear.get("materiaAdvanced"):
        return 5
    return base_slots


def optimize_materia_for_selection(
    selection: Sequence[Dict[str, Any]],
    base_stats: StatDict,
    level_mod: Dict[str, float],
    schema: Dict[str, Any],
    stat_modifiers: StatDict,
    job_level: int,
    gcd_threshold: float,
    cap_tables: Dict[str, Any],
    *,
    allowed_materia_stats: Optional[Sequence[str]] = None,
    food: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[StatDict, Dict[int, List[Optional[MateriaChoice]]], Dict[str, float]]]:
    allowed_stats = set(allowed_materia_stats or schema.get("stats", [])) & set(MATERIAS)
    speed_stat = "SKS" if "SKS" in allowed_stats else "SPS" if "SPS" in allowed_stats else None

    stats = summarize_stats(selection, base_stats)
    meldable: List[StatDict] = []
    materia_slots: List[int] = []
    for gear in selection:
        caps = get_caps(gear, cap_tables)
        meldable.append({stat: max(caps.get(stat, 0) - gear.get("stats", {}).get(stat, 0), 0) for stat in caps})
        materia_slots.append(build_materia_slots(gear))

    materia_plan: Dict[int, List[Optional[MateriaChoice]]] = {}
    for idx, slots in enumerate(materia_slots):
        if slots > 0:
            materia_plan[idx] = [None for _ in range(slots)]

    def current_effects(local_stats: StatDict) -> Tuple[StatDict, Dict[str, float]]:
        stats_with_food = apply_food(local_stats, food) if food else local_stats
        return stats_with_food, calculate_effects(
            stats_with_food,
            level_mod,
            stat_modifiers=stat_modifiers,
            main_stat=schema["main_stat"],
            trait_damage_multiplier=schema.get("trait_damage_multiplier", 1.0),
            party_bonus=schema.get("party_bonus", 1.0),
            job_level=job_level,
        )

    stats_with_food, effects = current_effects(stats)
    slots_remaining: List[Tuple[int, int]] = [
        (gear_index, slot_index)
        for gear_index, slots in enumerate(materia_slots)
        for slot_index in range(slots)
    ]

    if effects["gcd"] > gcd_threshold:
        if speed_stat is None:
            return None
        while effects["gcd"] > gcd_threshold:
            best_move: Optional[Tuple[Tuple[int, int], MateriaChoice, StatDict, StatDict, Dict[str, float]]] = None
            for gear_index, slot_index in slots_remaining:
                allow_restricted = slot_index < int(selection[gear_index].get("materiaSlot", 0))
                best_grade = best_materia_grade_value(
                    int(selection[gear_index].get("level", 0)),
                    speed_stat,
                    allow_restricted=allow_restricted,
                )
                if best_grade is None:
                    continue
                grade, value = best_grade
                add = min(value, meldable[gear_index].get(speed_stat, 0))
                if add <= 0:
                    continue
                trial_stats = dict(stats)
                trial_stats[speed_stat] = trial_stats.get(speed_stat, 0) + add
                trial_stats_with_food, trial_effects = current_effects(trial_stats)
                if trial_effects["gcd"] < effects["gcd"]:
                    move = (
                        (gear_index, slot_index),
                        (speed_stat, grade),
                        trial_stats,
                        trial_stats_with_food,
                        trial_effects,
                    )
                    if best_move is None or trial_effects["gcd"] < best_move[4]["gcd"]:
                        best_move = move
            if best_move is None:
                return None
            (gear_index, slot_index), choice, stats, stats_with_food, effects = best_move
            materia_plan[gear_index][slot_index] = choice
            stat, grade = choice
            meldable[gear_index][stat] = max(meldable[gear_index].get(stat, 0) - MATERIAS[stat][grade - 1], 0)
            slots_remaining.remove((gear_index, slot_index))

    while slots_remaining:
        best_move: Optional[Tuple[Tuple[int, int], MateriaChoice, StatDict, StatDict, Dict[str, float]]] = None
        for gear_index, slot_index in slots_remaining:
            allow_restricted = slot_index < int(selection[gear_index].get("materiaSlot", 0))
            gear_level = int(selection[gear_index].get("level", 0))
            for stat in allowed_stats:
                best_grade = best_materia_grade_value(gear_level, stat, allow_restricted=allow_restricted)
                if best_grade is None:
                    continue
                grade, value = best_grade
                add = min(value, meldable[gear_index].get(stat, 0))
                if add <= 0:
                    continue
                trial_stats = dict(stats)
                trial_stats[stat] = trial_stats.get(stat, 0) + add
                trial_stats_with_food, trial_effects = current_effects(trial_stats)
                delta_damage = trial_effects["damage"] - effects["damage"]
                if best_move is None or delta_damage > (best_move[4]["damage"] - effects["damage"]):
                    best_move = (
                        (gear_index, slot_index),
                        (stat, grade),
                        trial_stats,
                        trial_stats_with_food,
                        trial_effects,
                    )
        if best_move is None:
            break
        (gear_index, slot_index), choice, stats, stats_with_food, effects = best_move
        materia_plan[gear_index][slot_index] = choice
        stat, grade = choice
        meldable[gear_index][stat] = max(meldable[gear_index].get(stat, 0) - MATERIAS[stat][grade - 1], 0)
        slots_remaining.remove((gear_index, slot_index))

    return stats_with_food, materia_plan, effects


def optimize(config: Dict[str, Any]) -> Dict[str, Any]:
    job_level = int(config["job_level"])
    level_mod = LEVEL_MODIFIERS[job_level]
    schema = config["job_schema"]
    stat_modifiers: StatDict = {k: float(v) for k, v in schema.get("stat_modifiers", {}).items()}
    job = config["job"]

    base_stats = build_base_stats(schema["stats"], level_mod, stat_modifiers, config.get("clan_adjustments"))

    job_categories = load_js_export(Path(config["data_paths"]["job_categories"]))
    cap_tables = load_cap_tables(config)
    gears = load_gears(config, job_categories)
    grouped = group_by_slot(gears, config.get("max_items_per_slot", 6))

    combination_limit = int(config.get("max_combination_limit", 300000))
    combination_count = 1
    for items in grouped.values():
        combination_count *= max(len(items), 1)
    if combination_count > combination_limit:
        raise SystemExit(
            f"Search space too large ({combination_count} combinations). "
            "Lower max_items_per_slot or raise max_combination_limit to continue."
        )

    foods: List[Optional[Dict[str, Any]]] = [None]
    if config.get("allow_food", True):
        for food in load_js_export(Path(config["data_paths"]["foods"])):
            category = job_categories[food["jobCategory"]]
            if category and category.get(job, False):
                foods.append(food)

    evaluation_limit = int(config.get("max_operation_limit", combination_limit))
    if combination_count * max(len(foods), 1) > evaluation_limit:
        raise SystemExit(
            f"Food-inclusive search space too large ({combination_count * len(foods)} iterations). "
            "Reduce max_items_per_slot, disable food, or raise max_operation_limit."
        )

    slots = sorted(grouped)
    best: Optional[Dict[str, Any]] = None

    def record_best(
        selection: Sequence[Dict[str, Any]],
        food: Optional[Dict[str, Any]],
        effects: Dict[str, float],
        stats: StatDict,
        materia_plan: Dict[int, List[Optional[MateriaChoice]]],
    ):
        nonlocal best
        if best is None or effects["damage"] > best["effects"]["damage"]:
            best = {
                "selection": list(selection),
                "food": food,
                "effects": effects,
                "stats": stats,
                "materias": materia_plan,
            }

    def evaluate(selection: Sequence[Dict[str, Any]]):
        for food in foods:
            optimized = optimize_materia_for_selection(
                selection,
                base_stats,
                level_mod,
                schema,
                stat_modifiers,
                job_level,
                config["gcd_threshold"],
                cap_tables,
                allowed_materia_stats=config.get("materia_stats"),
                food=food,
            )
            if optimized is None:
                continue
            final_stats, materia_plan, effects = optimized
            if effects["gcd"] <= config["gcd_threshold"]:
                record_best(selection, food, effects, final_stats, materia_plan)

    def iterate(index: int, current: List[Dict[str, Any]]):
        if index == len(slots):
            evaluate(current)
            return
        for gear in grouped[slots[index]]:
            current.append(gear)
            iterate(index + 1, current)
            current.pop()

    iterate(0, [])
    if best is None:
        raise SystemExit("No gearset meets the GCD threshold; try relaxing constraints or raising max_items_per_slot.")
    return best


def format_selection(selection: Sequence[Dict[str, Any]], materia_plan: Dict[int, List[Optional[MateriaChoice]]]) -> List[Dict[str, Any]]:
    formatted = []
    for idx, gear in enumerate(selection):
        materias = []
        for materia in materia_plan.get(idx, []):
            if materia is None:
                materias.append(None)
            else:
                stat, grade = materia
                materias.append([stat, grade])
        formatted.append({
            "id": gear["id"],
            "name": gear.get("name"),
            "slot": gear.get("slot"),
            "level": gear.get("level"),
            "stats": gear.get("stats", {}),
            "materias": materias,
        })
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize FFXIV gear for damage under a GCD threshold.")
    parser.add_argument("--config", required=True, type=Path, help="Path to optimizer config (JSON)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    best = optimize(config)

    output = {
        "job": config["job"],
        "gcd_threshold": config["gcd_threshold"],
        "effects": best["effects"],
        "stats": best["stats"],
        "food": None
        if best["food"] is None
        else {"id": best["food"]["id"], "name": best["food"].get("name"), "stats": best["food"].get("stats")},
        "gear": format_selection(best["selection"], best.get("materias", {})),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
