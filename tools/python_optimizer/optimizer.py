"""
FFXIV Universal Gear Optimizer (修正武器伤害权重 & 平滑收益求解算法)
用法: python solver_universal.py config.json
"""
from __future__ import annotations

import json
import math
import itertools
import glob
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==========================================
# 1. 常量定义 (Level 100 / 7.0 Dawntrail)
# ==========================================
# Level 100 参数
LEVEL_MOD = {
    "main": 440, "sub": 420, "div": 2780, "det": 2780, "detTrunc": 1,
    "ap": 237, "apTank": 190, "hp": 40,
    "vit": 30.1, "vitTank": 43.0
}

# 职业补正 (Job Mod) - 用于计算武器伤害
# MNK/NIN/VPR = 110, Tanks/DRG/RPR = 100, Casters/Healers/Ranged = 115 (大致)
# 这里为了通用性，默认设为 115 (影响不大，因为只用于对比)，但最好在 config 指定
DEFAULT_JOB_MOD = 115 

STATS = ["CRT", "DET", "DHT", "SKS", "SPS", "TEN", "VIT", "STR", "DEX", "INT", "MND"]
# 求解器重点关注的副属性
SUB_STATS = ["CRT", "DET", "DHT", "SKS", "SPS", "TEN"]

SLOT_WEAPON_1H = 1
SLOT_WEAPON_2H = 13 
SLOT_OFF_HAND = 2   
SLOT_HEAD = 3
SLOT_BODY = 4
SLOT_HANDS = 5
SLOT_LEGS = 7
SLOT_FEET = 8
SLOT_EARS = 9
SLOT_NECK = 10
SLOT_WRIST = 11
SLOT_RING = 12

SLOTS_LEFT = [SLOT_HEAD, SLOT_BODY, SLOT_HANDS, SLOT_LEGS, SLOT_FEET]
SLOTS_ACC = [SLOT_EARS, SLOT_NECK, SLOT_WRIST]

RARITY_GREEN = 2 
RARITY_BLUE = 3  

MATERIA_HI = 54 
MATERIA_LO = 18 

# ==========================================
# 2. 核心数学公式 (官方公式 & 平滑公式)
# ==========================================

def get_stat_with_food(stat_name: str, base_val: int, food_config: Dict) -> int:
    """计算食物加成后的最终属性"""
    if not food_config or "stats" not in food_config:
        return base_val
    buff = food_config["stats"].get(stat_name)
    if not buff: return base_val
    percent, cap = buff
    bonus = min(math.floor(base_val * percent / 100), cap)
    return base_val + bonus

def get_gcd(speed_stat: int, haste_reduction: int = 0, gcd_modifier: int = 100) -> float:
    """计算 GCD，兼容职业 GCD 修正与外部急速减免"""
    sub = LEVEL_MOD["sub"]
    div = LEVEL_MOD["div"]
    step1 = 1000 - math.floor(130 * (speed_stat - sub) / div)
    step2 = math.floor(step1 * 2500 / 1000)
    step3 = math.floor(step2 * (100 - haste_reduction) / 100)
    step4 = math.floor(step3 * gcd_modifier / 1000)
    return math.floor(step4 * 100 / 1000) / 100

def calc_damage_multiplier(stats: Dict[str, int], config: Dict, use_floor: bool = True) -> float:
    """按照前端同款公式计算综合伤害期望倍率"""
    main_key = config["main_stat"]
    job_schema = config.get("job_schema", {})
    stat_mod = job_schema.get("stat_modifiers", {})
    trait_mult = job_schema.get("trait_damage_multiplier", 1.0)
    party_bonus = job_schema.get("party_bonus", 1.05)

    # 0. 辅助函数：根据模式选择是否 floor
    def do_math(val):
        return math.floor(val) if use_floor else val

    main_base = LEVEL_MOD["main"]
    sub_base = LEVEL_MOD["sub"]
    div = LEVEL_MOD["div"]
    det_trunc = LEVEL_MOD.get("detTrunc", 1)

    # 1. 武器伤害 (Weapon Damage)
    attack_main = "STR" if main_key == "VIT" else main_key
    job_mod = stat_mod.get(attack_main, config.get("job_mod", DEFAULT_JOB_MOD))
    wd_val = stats.get("MDMG", 0) if main_key in ["INT", "MND"] else stats.get("PDMG", 0)
    weapon_damage = do_math(main_base * job_mod / 1000) + wd_val

    # 2. 主属性 (Attack Power)
    is_tank = (config.get("role") == "tank") or (main_key == "VIT")
    ap_coeff = LEVEL_MOD["apTank"] if is_tank else LEVEL_MOD["ap"]
    attack_stat = stats.get(attack_main, 0)
    attack_with_party = do_math(attack_stat * party_bonus)
    main_damage = (do_math(ap_coeff * (attack_with_party - main_base) / main_base) + 100) / 100

    # 3. 副属性计算
    crt = stats.get("CRT", 0)
    prob_crt = do_math(200 * (crt - sub_base) / div + 50) / 1000
    dmg_crt = do_math(200 * (crt - sub_base) / div + 1400) / 1000
    f_crt = 1 + (prob_crt * (dmg_crt - 1))

    dht = stats.get("DHT", 0)
    prob_dht = do_math(550 * (dht - sub_base) / div) / 1000
    f_dht = 1 + (prob_dht * 0.25)

    det = stats.get("DET", 0)
    f_det = do_math((140 * (det - main_base) / LEVEL_MOD["det"] + 1000) / det_trunc) * det_trunc / 1000

    f_ten = 1.0
    if is_tank:
        ten = stats.get("TEN", 0)
        f_ten = do_math(112 * (ten - sub_base) / div + 1000) / 1000

    # 速度系数仅用于平滑求解，避免完全无视速度
    f_spd = 1.0
    if not use_floor:
        sks = stats.get("SKS", 0) + stats.get("SPS", 0)
        f_spd = 1.0 + (sks * 0.00001)

    return 0.01 * weapon_damage * main_damage * f_det * f_ten * trait_mult * f_crt * f_dht * f_spd

# ==========================================
# 3. 智能魔晶石求解器 (平滑寻路版)
# ==========================================
class SmartMateriaSolver:
    def __init__(self, gear_set, base_stats, config):
        self.gear_set = gear_set
        self.raw_stats = base_stats.copy()
        self.config = config
        self.speed_stat_name = "SPS" if config.get("main_stat") in ["INT", "MND"] else "SKS"
        self.food_config = config.get("food", {})
        
        # 装备 Cap 初始化
        self.gear_caps = []
        self.gear_sim_state = []
        for item in gear_set:
            stats = item.get("stats", {})
            # Cap 判定：取该装备最高的副属性值
            sub_vals = [stats.get(k, 0) for k in SUB_STATS]
            local_cap = max(sub_vals) if sub_vals else 0
            self.gear_caps.append(local_cap)
            self.gear_sim_state.append(stats.copy())

        self.slots_pool = self._build_slots_pool()
        
    def _build_slots_pool(self):
        pool = []
        for idx, item in enumerate(self.gear_set):
            rarity = item.get("rarity", RARITY_BLUE)
            slot = item["slot"]
            
            is_left_side = (slot in [SLOT_WEAPON_1H, SLOT_OFF_HAND] + SLOTS_LEFT)
            is_right_side = (slot in SLOTS_ACC + [SLOT_RING])

            guaranteed_slots = item.get("materiaSlot", 0)
            if guaranteed_slots == 0:
                if is_left_side: guaranteed_slots = 2
                elif is_right_side: guaranteed_slots = 2 if rarity == RARITY_GREEN else 1

            melds = []
            if rarity == RARITY_GREEN:
                # 生产禁断：3颗大 + 2颗小
                melds = [MATERIA_HI] * 3 + [MATERIA_LO] * 2 
            else:
                # 蓝装：只填必得孔
                melds = [MATERIA_HI] * guaranteed_slots
            
            for val in melds:
                pool.append({'val': val, 'gear_idx': idx})
        
        # 排序：大石头优先
        pool.sort(key=lambda x: x['val'], reverse=True)
        return pool

    def _get_effective_stats(self, current_raw_stats):
        eff_stats = {}
        for k, v in current_raw_stats.items():
            eff_stats[k] = get_stat_with_food(k, v, self.food_config)
        return eff_stats

    def _try_meld(self, gear_idx: int, stat_name: str, materia_val: int) -> int:
        current_val = self.gear_sim_state[gear_idx].get(stat_name, 0)
        cap = self.gear_caps[gear_idx]
        space = cap - current_val
        return max(0, min(materia_val, space))

    def _apply_meld(self, gear_idx: int, stat_name: str, gain: int):
        self.gear_sim_state[gear_idx][stat_name] = self.gear_sim_state[gear_idx].get(stat_name, 0) + gain
        self.raw_stats[stat_name] = self.raw_stats.get(stat_name, 0) + gain

    def solve(self):
        slots_queue = self.slots_pool[:]
        melds_log = {i: [] for i in range(len(self.gear_set))}
        
        # --- Phase 1: 强制满足 GCD 阈值 ---
        haste = self.config.get("haste_reduction", 0)
        gcd_modifier = self.config.get("job_schema", {}).get("stat_modifiers", {}).get("gcd", 100)
        if self.config.get("job_level", 100) < 80:
            gcd_modifier = 100
        target_gcd = self.config["gcd_threshold"]
        
        # 将速度孔需求单独提取，避免大石头被浪费
        # 这里的逻辑是：必须优先满足 GCD，哪怕牺牲高优先级的孔
        # 但是为了最优，应该尽量用小石头填补微小差距吗？
        # 简化策略：按顺序尝试插速度，直到达标
        
        temp_skipped_slots = []
        
        while True:
            eff_stats = self._get_effective_stats(self.raw_stats)
            current_spd = eff_stats.get(self.speed_stat_name, 0)
            if get_gcd(current_spd, haste, gcd_modifier) <= target_gcd + 0.001:
                break # 达标
            
            if not slots_queue:
                # 孔用完了还没达标，此配装方案无效
                return None, 0, {}
            
            # 寻找能插速度的孔
            found_slot = False
            # 临时队列用于遍历
            idx_to_remove = -1
            
            for i, slot in enumerate(slots_queue):
                gain = self._try_meld(slot['gear_idx'], self.speed_stat_name, slot['val'])
                if gain > 0:
                    # 插进去
                    self._apply_meld(slot['gear_idx'], self.speed_stat_name, gain)
                    melds_log[slot['gear_idx']].append(f"{self.speed_stat_name} +{gain}")
                    idx_to_remove = i
                    found_slot = True
                    break
            
            if found_slot:
                slots_queue.pop(idx_to_remove)
            else:
                # 所有剩余孔都插不了速度（全满），GCD 无法达成
                return None, 0, {}

        # --- Phase 2: 填充剩余孔 (使用平滑收益判断) ---
        candidates = ["CRT", "DET", "DHT"]
        if self.config.get("role") == "tank" or self.config.get("main_stat") == "VIT":
             candidates.append("TEN")
        
        for slot in slots_queue:
            gear_idx = slot['gear_idx']
            val = slot['val']
            
            best_stat = None
            max_smooth_gain = -1
            chosen_gain = 0
            
            # 基础伤害 (平滑模式)
            base_eff = self._get_effective_stats(self.raw_stats)
            # 关键：这里用 use_floor=False，避免阈值陷阱
            base_dmg_smooth = calc_damage_multiplier(base_eff, self.config, use_floor=False)
            
            for stat in candidates:
                gain = self._try_meld(gear_idx, stat, val)
                if gain <= 0: continue
                
                temp_raw = self.raw_stats.copy()
                temp_raw[stat] = temp_raw.get(stat, 0) + gain
                temp_eff = self._get_effective_stats(temp_raw)
                
                # 计算平滑后的收益
                new_dmg_smooth = calc_damage_multiplier(temp_eff, self.config, use_floor=False)
                delta = new_dmg_smooth - base_dmg_smooth
                
                if delta > max_smooth_gain:
                    max_smooth_gain = delta
                    best_stat = stat
                    chosen_gain = gain
            
            # 如果 CRT/DET/DHT 都打不了，或者没收益，就打 DET 填空 (或者看 log)
            if best_stat:
                self._apply_meld(gear_idx, best_stat, chosen_gain)
                melds_log[gear_idx].append(f"{best_stat} +{chosen_gain}")
            else:
                melds_log[gear_idx].append("X (Full)")

        # --- 最终评分: 使用 Floor 模式 ---
        final_eff = self._get_effective_stats(self.raw_stats)
        real_score = calc_damage_multiplier(final_eff, self.config, use_floor=True)
        return final_eff, real_score, melds_log

# ==========================================
# 4. 数据加载 (路径清洗修复版)
# ==========================================
def load_and_filter_data(config: Dict) -> Dict:
    gears_flat = []

    buckets = {i: [] for i in [SLOT_WEAPON_1H, SLOT_OFF_HAND] + SLOTS_LEFT + SLOTS_ACC + [SLOT_RING]}

    raw_path = config["data_paths"]
    # 路径清洗
    paths = glob.glob(raw_path.strip().strip("'").strip('"'))

    if not paths:
        print(f"❌ Error: 找不到数据文件: {raw_path}")
        sys.exit(1)

    print(f"📂 读取数据: {paths[0]}")
    script = "import(process.argv[1]).then(m=>console.log(JSON.stringify(m.default ?? m)))"
    try:
        file_uri = Path(paths[0]).resolve().as_uri()
        res = subprocess.run(
            ["node", "--input-type=module", "-e", script, file_uri],
            capture_output=True, text=True, encoding="utf-8"
        )
        data = json.loads(res.stdout)
        if isinstance(data, list): gears_flat.extend(data)
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        sys.exit(1)

    valid_cats = config.get("job_cat_ids", [])
    base_min_il = config.get("min_il", 0)
    base_max_il = config.get("max_il", 9999)
    extra_ranges = config.get("extra_il_ranges", [])

    loaded_count = 0
    for g in gears_flat:
        try:
            lvl = int(g.get("level", 0))
            cat = int(g.get("jobCategory", 0))
            slot = int(g.get("slot", -1))
        except: continue

        if valid_cats and cat not in valid_cats: continue

        range_id = None
        if base_min_il <= lvl <= base_max_il:
            range_id = 0
        else:
            for idx, r in enumerate(extra_ranges, start=1):
                if lvl >= r.get("min_il", 0) and lvl <= r.get("max_il", 9999):
                    range_id = idx
                    break

        if range_id is None:
            continue

        if slot == SLOT_WEAPON_2H: slot = SLOT_WEAPON_1H

        if slot in buckets:
            g["_il_range"] = range_id
            buckets[slot].append(g)
            loaded_count += 1

    print(f"✅ 加载完成: {loaded_count} 个符合条件的装备。")
    return buckets

# ==========================================
# 5. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    clean_path = args.config_file.strip().strip("'").strip('"').rstrip("\\").rstrip("/")
    
    try:
        with open(clean_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        sys.exit(1)

    print(f"=== FFXIV 求解器 (平滑收益版) Job: {config.get('job')} ===")
    
    buckets = load_and_filter_data(config)
    
    if not buckets[SLOT_WEAPON_1H]:
        print("❌ 错误: 无武器数据")
        sys.exit(1)
    
    target_slots = [SLOT_WEAPON_1H]
    if buckets[SLOT_OFF_HAND]: target_slots.append(SLOT_OFF_HAND)
    target_slots.extend(SLOTS_LEFT)
    target_slots.extend(SLOTS_ACC)
    
    if not buckets[SLOT_RING]:
        print("❌ 错误: 无戒指数据")
        sys.exit(1)

    # 准备基础属性
    default_base = {k: 420 for k in STATS}
    for m in ["STR", "DEX", "INT", "MND"]: default_base[m] = 440
    user_base = config.get("base_stats", {})
    base_stats_clean = default_base.copy()
    for k, v in user_base.items(): base_stats_clean[k] = v

    # 组合生成
    non_ring_combos = list(itertools.product(*[buckets[s] for s in target_slots]))
    rings = buckets[SLOT_RING]
    blue_rings = [r for r in rings if r.get("rarity", 0) != RARITY_GREEN]
    green_rings = [r for r in rings if r.get("rarity", 0) == RARITY_GREEN]
    
    ring_pairs = []
    ring_pairs.extend(list(itertools.combinations(blue_rings, 2)))
    ring_pairs.extend(list(itertools.product(blue_rings, green_rings)))
    ring_pairs.extend(list(itertools.combinations_with_replacement(green_rings, 2)))
    
    total_ops = len(non_ring_combos) * len(ring_pairs)
    print(f"📊 待计算组合数: {total_ops}")
    
    best_result = None
    best_score = -1

    extra_ranges = config.get("extra_il_ranges", [])
    range_limits = {idx + 1: r.get("max_items", None) for idx, r in enumerate(extra_ranges)}
    
    counter = 0
    for gear_tuple in non_ring_combos:
        for r_pair in ring_pairs:
            counter += 1
            if counter % 50000 == 0: print(f"⏳ {counter}/{total_ops} ...")
            
            full_set = list(gear_tuple) + list(r_pair)

            # 限制额外高装等装备数量，模拟部分刷取高装等的开荒场景
            valid_combo = True
            range_counter = {}
            for item in full_set:
                rid = item.get("_il_range", 0)
                if rid > 0:
                    range_counter[rid] = range_counter.get(rid, 0) + 1
                    limit = range_limits.get(rid)
                    if limit is not None and range_counter[rid] > limit:
                        valid_combo = False
                        break

            if not valid_combo:
                continue
            
            # 计算装备白值
            current_raw = base_stats_clean.copy()
            for item in full_set:
                for k, v in item.get("stats", {}).items():
                    if k in current_raw: current_raw[k] += v
            
            solver = SmartMateriaSolver(full_set, current_raw, config)
            final_stats, score, melds = solver.solve()
            
            if final_stats is None: continue 
            
            if score > best_score:
                best_score = score
                best_result = {
                    "gear": full_set,
                    "stats": final_stats,
                    "melds": melds,
                    "score": score
                }

    if best_result:
        print("\n" + "="*50)
        print(f"🎉 最优配装 (Score: {best_result['score']:.4f})")
        print("="*50)
        
        spd_key = "SPS" if config.get("main_stat") in ["INT", "MND"] else "SKS"
        gcd_modifier = config.get("job_schema", {}).get("stat_modifiers", {}).get("gcd", 100)
        if config.get("job_level", 100) < 80:
            gcd_modifier = 100
        final_gcd = get_gcd(best_result['stats'][spd_key], config.get("haste_reduction", 0), gcd_modifier)
        print(f"⏱️ GCD: {final_gcd}s")
        
        print("\n[ 面板属性 (含食物) ]")
        for k in STATS:
            val = int(best_result['stats'].get(k, 0))
            if val > 0: print(f"{k:<5}: {val}")
            
        print("\n[ 装备与镶嵌 ]")
        sorted_gear = sorted(best_result['gear'], key=lambda x: (x.get("_original_slot", x['slot']), x['id']))
        slot_names = {1:"Weapon", 2:"Shield", 3:"Head", 4:"Body", 5:"Hands", 7:"Legs", 8:"Feet", 9:"Ear", 10:"Neck", 11:"Wrist", 12:"Ring"}
        
        for idx, item in enumerate(best_result['gear']):
            real_slot = item.get("_original_slot", item['slot'])
            sname = slot_names.get(real_slot, f"Slot{real_slot}")
            if real_slot == 13: sname = "Weapon(2H)"
            melds_str = " | ".join(best_result['melds'][idx]) if best_result['melds'][idx] else "-"
            print(f"{sname:<12} | {item['name'][:20]:<20} | {melds_str}")
    else:
        print("❌ 无有效配装")

if __name__ == "__main__":
    main()
