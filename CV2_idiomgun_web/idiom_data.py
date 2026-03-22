# ============================================================
# idiom_data.py
# 成語資料庫 + 題目產生器
# ============================================================
import random

idioms = {

"怒髮衝冠":{
    0:{"easy":["努"],"medium":["弩"],"hard":["恕"]},
    1:{"medium":["髮"]},
    2:{"easy":["沖"],"medium":["充"]},
},

"四面楚歌":{
    0:{"easy":["西"],"medium":["死"],"hard":["匹"]},
    2:{"easy":["處"],"hard":["濋"]},
    3:{"easy":["哥"]}
},

"畫蛇添足":{
    0:{"easy":["劃"],"medium":["晝"],"hard":["書"]},
    1:{"easy":["它"]},
    2:{"easy":["填"],"medium":["婖"]}
},

"守株待兔":{
    0:{"easy":["首"],"medium":["宋"],"hard":["宇"]},
    1:{"easy":["珠"],"medium":["殊"],"hard":["朱"]},
    2:{"hard":["侍"]},
    3:{"medium":["免"]}
},

"自相矛盾":{
    0:{"easy":["白"],"medium":["目"],"hard":["由"]},
    1:{"medium":["湘"],"hard":["箱"]},
    2:{"easy":["予"],"medium":["茅"]},
    3:{"easy":["頓"],"medium":["鈍"],"hard":["遁"]}
},

"一鼓作氣":{
    1:{"easy":["股"]},
    2:{"easy":["做"]},
    3:{"easy":["汽"],"medium":["棄"]}
},

"亡羊補牢":{
    0:{"easy":["忘"],"medium":["芒"]},
    2:{"easy":["捕"]}
},

"破釜沉舟":{
    0:{"easy":["坡"],"medium":["波"],"hard":["披"]},
    1:{"easy":["斧"]},
    2:{"medium":["沈"]}
},

"對症下藥":{
    0:{"easy":["隊"],"hard":["對"]},
    1:{"easy":["証"],"medium":["政"]}
},

"刻苦耐勞":{
    0:{"easy":["克"],"medium":["剋"]},
    1:{"easy":["奈"]},
    2:{"easy":["牢"]}
},

"一目了然":{
    1:{"easy":["自"],"medium":["且"],"hard":["日"]},
    2:{"easy":["瞭"]},
    3:{"easy":["燃"],"hard":["染"]}
},

"一箭雙鵰":{
    1:{"easy":["剪"],"medium":["劍"]},
    2:{"hard":["爽"]},
    3:{"easy":["雕"],"medium":["凋"]}
},

"三思而行":{
    1:{"easy":["恩"]}
},

"五花八門":{
    1:{"medium":["化"]},
    2:{"easy":["人"],"medium":["入"]},
    3:{"hard":["們"]}
},

"虎視眈眈":{
    1:{"hard":["示"]},
    2:{"easy":["耽"]},
    3:{"easy":["耽"]}
},

"魚目混珠":{
    0:{"easy":["漁"]},
    1:{"easy":["自"],"medium":["且"],"hard":["日"]},
    2:{"hard":["渾"]},
    3:{"easy":["株"],"hard":["誅"]}
},

"青出於藍":{
    0:{"easy":["清"]},
    2:{"hard":["于"]},
    3:{"medium":["籃"]}
},

"百發百中":{
    0:{"easy":["白"]},
    1:{"medium":["廢"]},
    2:{"easy":["白"]},
    3:{"easy":["忠"],"medium":["仲"]}
},

"當機立斷":{
    0:{"easy":["檔"],"medium":["擋"]},
    1:{"easy":["基"],"hard":["奇"]},
    3:{"easy":["段"],"hard":["鍛"]}
},

"半途而廢":{
    0:{"easy":["牛"],"hard":["丰"]},
    1:{"easy":["圖"],"medium":["徒"],"hard":["徙"]},
    3:{"easy":["費"]}
},

"水落石出":{
    1:{"easy":["洛"],"medium":["絡"],"hard":["駱"]},
    2:{"easy":["右"]}
},

"手忙腳亂":{
    1:{"easy":["茫"],"medium":["盲"],"hard":["芒"]},
    3:{"medium":["辭"]}
},

"心驚膽跳":{
    0:{"easy":["必"]},
    2:{"easy":["擔"],"hard":["憚"]},
    3:{"medium":["挑"],"hard":["眺"]}
},

"名列前茅":{
    0:{"easy":["各"],"hard":["洛"]},
    1:{"medium":["烈"],"hard":["裂"]},
    2:{"medium":["煎"],"hard":["箭"]},
    3:{"medium":["矛"]}
},

"全神貫注":{
    0:{"easy":["金"]},
    1:{"easy":["伸"],"hard":["紳"]},
    2:{"easy":["串"],"medium":["慣"],"hard":["摜"]},
    3:{"easy":["住"],"medium":["註"],"hard":["柱"]}
},

"按部就班":{
    0:{"easy":["案"],"medium":["暗"],"hard":["黯"]},
    1:{"easy":["陪"],"medium":["步"],"hard":["倍"]},
    3:{"easy":["般"],"medium":["斑"],"hard":["搬"]}
},

"前功盡棄":{
    0:{"easy":["剪"],"hard":["煎"]},
    1:{"easy":["工"],"medium":["攻"],"hard":["公"]},
    2:{"easy":["儘"],"medium":["進"],"hard":["禁"]},
    3:{"easy":["氣"]}
},

"持之以恆":{
    0:{"hard":["待"]},
    1:{"easy":["支"]},
    2:{"easy":["已"],"medium":["己"],"hard":["乙"]},
    3:{"easy":["衡"]}
},

"如魚得水":{
    0:{"hard":["奴"]},
    1:{"easy":["漁"]},
    2:{"easy":["德"]}
},
}

options_pool = list("家國山水風雲花草人口手足心肝腦頭耳目鼻天地日月星空海河湖江田土木火金石")

DIFFICULTY_ORDER = ["easy", "medium", "hard"]

def _pick_wrong_char(pos_data: dict, difficulty: str) -> str | None:
    order = [difficulty] + [d for d in DIFFICULTY_ORDER if d != difficulty]
    for d in order:
        if d in pos_data and pos_data[d]:
            return random.choice(pos_data[d])
    return None


def make_wrong_question(idiom: str, difficulty: str = "easy") -> dict | None:
    pos_data = idioms.get(idiom, {})
    if not pos_data:
        return None

    valid_positions = list(pos_data.keys())
    if not valid_positions:
        return None

    pos = random.choice(valid_positions)
    wrong_char = _pick_wrong_char(pos_data[pos], difficulty)
    if wrong_char is None:
        return None

    correct_char = idiom[pos]
    display = list(idiom)
    display[pos] = wrong_char
    display_str = "".join(display)

    return {
        "type":         "wrong",
        "idiom":        idiom,
        "display":      display_str,
        "wrong_idx":    pos,
        "wrong_char":   wrong_char,
        "correct_char": correct_char,
        "hint":         "找出錯字並秒準 1.5 秒",
        "difficulty":   difficulty,
    }


def make_fill_question(idiom: str, difficulty: str = "easy") -> dict | None:
    pos_data = idioms.get(idiom, {})
    valid_positions = list(pos_data.keys())
    if not valid_positions:
        pos = random.randint(0, len(idiom)-1)
        distractors = []
    else:
        pos = random.choice(valid_positions)
        distractors = []
        for d in DIFFICULTY_ORDER:
            if d in pos_data[pos]:
                distractors.extend(pos_data[pos][d])
        if difficulty == "easy":
            preferred = pos_data[pos].get("easy", [])
            distractors = preferred if preferred else distractors
        elif difficulty == "medium":
            preferred = (pos_data[pos].get("easy", []) +
                         pos_data[pos].get("medium", []))
            distractors = preferred if preferred else distractors

    correct_char = idiom[pos]
    distractors  = [c for c in distractors if c != correct_char]

    options = [correct_char]
    for c in distractors:
        if c not in options:
            options.append(c)
        if len(options) == 4:
            break

    pool_copy = [c for c in options_pool if c not in options]
    random.shuffle(pool_copy)
    while len(options) < 4 and pool_copy:
        options.append(pool_copy.pop())

    random.shuffle(options)

    template = list(idiom)
    template[pos] = "＿"
    template_str = "".join(template)

    return {
        "type":       "fill",
        "idiom":      idiom,
        "template":   template_str,
        "blank_idx":  pos,
        "answer":     correct_char,
        "options":    options[:4],
        "hint":       "秒準正確的字 1.5 秒",
        "difficulty": difficulty,
    }


def generate_questions(
    n: int = 10,
    difficulty: str = "mixed",   # "mixed" 代表隨機混合三種難度
    wrong_ratio: float = 0.5,
) -> list[dict]:
    all_idioms  = list(idioms.keys())
    random.shuffle(all_idioms)

    n_wrong = round(n * wrong_ratio)
    n_fill  = n - n_wrong

    questions = []
    used      = set()

    def _random_diff():
        """每題隨機挑一個難度"""
        if difficulty == "mixed":
            return random.choice(["easy", "medium", "hard"])
        return difficulty

    def _pick(make_fn, count):
        pool = [i for i in all_idioms if i not in used]
        random.shuffle(pool)
        result = []
        for idiom in pool:
            if len(result) >= count:
                break
            q = make_fn(idiom, _random_diff())
            if q:
                result.append(q)
                used.add(idiom)
        return result

    questions += _pick(make_wrong_question, n_wrong)
    questions += _pick(make_fill_question,  n_fill)

    if len(questions) < n:
        extra_needed = n - len(questions)
        extra_pool = all_idioms[:]
        random.shuffle(extra_pool)
        for idiom in extra_pool:
            if len(questions) >= n:
                break
            if extra_needed % 2 == 0:
                q = make_fill_question(idiom, _random_diff())
            else:
                q = make_wrong_question(idiom, _random_diff())
            if q:
                questions.append(q)
                extra_needed -= 1

    random.shuffle(questions)
    return questions[:n]


if __name__ == "__main__":
    qs = generate_questions(n=10, difficulty="easy", wrong_ratio=0.5)
    for i, q in enumerate(qs, 1):
        print(f"[{i}] {q['type']:5s} | {q.get('display') or q.get('template'):6s} "
              f"| ans={q.get('wrong_char') or q.get('answer')} "
              f"| difficulty={q['difficulty']}")
        if q["type"] == "fill":
            print(f"       options={q['options']}")
