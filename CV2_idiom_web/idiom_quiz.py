import random
from sqlalchemy.sql.expression import func
from idioms_data import options_pool

# 引入剛剛建立的資料庫類別
from dataBase import IdiomDBManager, Idiom

# 在全域初始化 DB 連線，避免每次出題都重新啟動引擎
db = IdiomDBManager('sqlite:///CV2_idiom_web/idiom_game.db')

def generate_multiple_choice():
    with db.Session() as session:
        # 1. 取代 random.choice(list(idioms.keys()))：從資料庫隨機撈取一個成語
        idiom_obj = session.query(Idiom).order_by(func.random()).first()
        if not idiom_obj:
            return {"error": "資料庫無成語資料"}

        idiom_word = idiom_obj.word

        # 2. 取代 pos = random.choice(...)：找出該成語有設定干擾字的「位置」並隨機挑選
        available_positions = list(set([d.pos for d in idiom_obj.distractors]))
        if not available_positions:
            # 防呆：若該成語沒有設定任何干擾字，隨機挖空一個字
            pos = random.randint(0, len(idiom_word) - 1)
            wrong_opts = []
        else:
            pos = random.choice(available_positions)
            
            # 3. 取代原本的雙層 for 迴圈：抓出該位置的所有錯誤字 (合併所有難度並去重)
            distractors = [d.incorrect_char for d in idiom_obj.distractors if d.pos == pos]
            wrong_opts = list(set(distractors))

        # 建立題目與正確解答
        correct_char = idiom_word[pos]
        question = idiom_word[:pos] + "＿" + idiom_word[pos+1:]

        # 4. 不足 3 個錯誤字時，從 options_pool 補齊 (邏輯不變)
        while len(wrong_opts) < 3:
            c = random.choice(options_pool)
            if c != correct_char and c not in wrong_opts:
                wrong_opts.append(c)

        # 防呆：如果原本的錯誤字超過 3 個，隨機挑選 3 個
        if len(wrong_opts) > 3:
            wrong_opts = random.sample(wrong_opts, 3)

        choices = wrong_opts + [correct_char]
        random.shuffle(choices)

        return {
            "mode": "multiple",
            "idiom": idiom_word,
            "question": question,
            "choices": choices,
            "correct_char": correct_char,
            "correct_index": choices.index(correct_char) + 1   # 修正維持原樣
        }

def new_question():
    return generate_multiple_choice()