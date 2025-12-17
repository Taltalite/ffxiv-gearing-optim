# check_ids_universal.py
# 作用：扫描数据文件，按部位列出所有高装等装备的 Category ID
# 适配：所有职业（单手/双手武器、盾牌、防具、首饰）

import json
import subprocess
from pathlib import Path
import sys

# ================= 配置区 =================
# 1. 请填入 gear-recent.js 的绝对路径 (注意 Windows 路径要用双斜杠 \\ 或反斜杠 /)
DATA_PATH = "D:\\ffxiv-gearing-optim\\data\\out\\gears-recent.js"

# 2. 筛选装等下限 (只看这个装等以上的装备)
MIN_IL = 770
# ==========================================

# 部位 ID 对照表
SLOT_MAP = {
    1:  "⚔️ 单手武器 (Main Hand 1H) [PLD, BLM, WHM等]",
    13: "⚔️ 双手武器 (Main Hand 2H) [MNK, WAR, DRK等]",
    2:  "🛡️ 副手/盾牌 (Off Hand) [PLD]",
    3:  "🧢 头部 (Head)",
    4:  "👕 身体 (Body)",
    5:  "🧤 手部 (Hands)",
    7:  "👖 腿部 (Legs)",
    8:  "👞 脚部 (Feet)",
    9:  "👂 耳环 (Ears)",
    10: "📿 项链 (Neck)",
    11: "⌚ 手镯 (Wrist)",
    12: "💍 戒指 (Ring)"
}

def main():
    print(f"📂 正在读取数据文件: {DATA_PATH} ...")
    
    # 1. 使用 Node.js 读取 .js 文件 (处理 export default)
    script = "import(process.argv[1]).then(m=>console.log(JSON.stringify(m.default ?? m)))"
    try:
        res = subprocess.run(
            ["node", "--input-type=module", "-e", script, Path(DATA_PATH).resolve().as_uri()], 
            capture_output=True, text=True, encoding="utf-8"
        )
        
        if res.returncode != 0:
            print("❌ Node.js 读取失败，错误信息:")
            print(res.stderr)
            return

        data = json.loads(res.stdout)
        print(f"✅ 成功加载 {len(data)} 条数据。正在筛选 Level >= {MIN_IL} ...\n")
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 'node' 命令。请确保已安装 Node.js 并配置了环境变量。")
        return
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return

    # 2. 数据分组: results[slot_id][category_id] = "Example Item Name"
    results = {}
    found_count = 0

    for item in data:
        try:
            # 强制类型转换，防止数据格式异常
            lvl = int(item.get("level", 0))
            if lvl < MIN_IL: continue
            
            sid = int(item.get("slot", 0))
            cat = int(item.get("jobCategory", 0))
            name = item.get("name", "Unknown")
            
            if sid not in results:
                results[sid] = {}
            
            # 记录该部位下，每个 Category ID 的第一个示例名称
            if cat not in results[sid]:
                results[sid][cat] = name
                
            found_count += 1
        except:
            continue

    if found_count == 0:
        print(f"❌ 未找到任何装等 >= {MIN_IL} 的装备。请检查 MIN_IL 设置或数据文件路径。")
        return

    # 3. 打印结果
    sorted_slots = sorted(results.keys())
    
    print(f"{'='*60}")
    print(f"📊 扫描结果 (Min IL: {MIN_IL})")
    print(f"{'='*60}")

    all_found_cats = set()

    for sid in sorted_slots:
        sname = SLOT_MAP.get(sid, f"❓ 未知部位 (ID: {sid})")
        print(f"\n{sname}")
        print(f"{'-'*40}")
        
        cats = results[sid]
        if not cats:
            print("  (无符合条件的装备)")
            continue

        for cat_id, example_name in cats.items():
            print(f"  👉 Category ID: {cat_id:<5} | 示例: {example_name}")
            all_found_cats.add(cat_id)

    print(f"\n{'='*60}")
    print("📝 配置指南 (Config Guide)")
    print(f"{'='*60}")
    print("请查看上面输出中，属于你当前职业装备的 Category ID。")
    print("例如，如果你是武僧，请找到 '双手武器'、'身体'、'耳环' 下对应的 ID。")
    print("\n然后将这些 ID 填入你的 config.json 文件中：")
    print(f'"job_cat_ids": [填入你找到的数字, ...]')
    print(f"\n(参考: 本次扫描共发现了这些 ID: {list(all_found_cats)})")

if __name__ == "__main__":
    main()