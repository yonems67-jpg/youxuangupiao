import cnlunar
import json
import os
from datetime import datetime, timedelta, timezone

# ---- 出生信息(自用单人网站,硬编码在这里,不会上传到任何第三方)----
BIRTH_DATETIME = datetime(1997, 3, 22, 23, 29, 00)  # 阳历
GENDER = "male"  # 乾造(目前的运势逻辑没用到性别,留着给以后大运/流年功能用)

# 地支六冲 / 六合表,用来判断"今日"和八字四柱的互动关系
ZHI_CHONG = {
    "子": "午", "午": "子", "丑": "未", "未": "丑", "寅": "申", "申": "寅",
    "卯": "酉", "酉": "卯", "辰": "戌", "戌": "辰", "巳": "亥", "亥": "巳",
}
ZHI_HE = {
    "子": "丑", "丑": "子", "寅": "亥", "亥": "寅", "卯": "戌", "戌": "卯",
    "辰": "酉", "酉": "辰", "巳": "申", "申": "巳", "午": "未", "未": "午",
}

WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def get_beijing_now() -> datetime:
    # GitHub Actions 跑在 UTC 时间,这里统一转换成北京时间(naive datetime,cnlunar 需要这个格式)
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)


def compute_fortune():
    now = get_beijing_now()

    # cnlunar 的 8char 模式已经内置了子时(23:00)换日的规则,birth 和 today 都直接传入即可
    birth = cnlunar.Lunar(BIRTH_DATETIME, godType="8char")
    today = cnlunar.Lunar(now, godType="8char")

    bazi = {
        "year": birth.year8Char,
        "month": birth.month8Char,
        "day": birth.day8Char,
        "hour": birth.twohour8Char,
    }

    # 今日地支和八字四柱地支的冲合关系(规则透明,不是套话堆砌)
    today_zhi = today.day8Char[1]
    pillar_zhis = {"年柱": bazi["year"][1], "月柱": bazi["month"][1],
                    "日柱": bazi["day"][1], "时柱": bazi["hour"][1]}

    interactions = []
    for label, zhi in pillar_zhis.items():
        if ZHI_CHONG.get(today_zhi) == zhi:
            interactions.append(f"今日{today_zhi}冲{label}{zhi}")
        elif ZHI_HE.get(today_zhi) == zhi:
            interactions.append(f"今日{today_zhi}合{label}{zhi}")

    if interactions:
        summary = "、".join(interactions) + "。传统命理认为「冲」易有变动波折,「合」偏向和缓顺遂,仅供参考,不是确定性的预测。"
    else:
        summary = "今日地支与你八字四柱之间没有明显的冲合关系,运势相对平稳。"

    result = {
        "update_date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "solar_date": now.strftime("%Y年%m月%d日") + " " + WEEKDAY_CN[now.weekday()],
        "lunar_date": f"农历{today.lunarMonthCn}{today.lunarDayCn}",
        "zodiac": today.chineseYearZodiac,
        "nayin": today.get_nayin(),
        "day_officer": today.today12DayOfficer,
        "star_28": today.today28Star,
        "good_things": today.goodThing,
        "bad_things": today.badThing,
        "bazi": bazi,
        "fortune_summary": summary,
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/fortune.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{result['lunar_date']},今日运势数据已写入 site/data/fortune.json")


if __name__ == "__main__":
    compute_fortune()
