from dataBase import IdiomDBManager
from idioms_data import idioms

def run_migration():
    # 初始化資料庫管理器
    db = IdiomDBManager('sqlite:///CV2_idiom_web/idiom_game.db')
    
    # 定義難度映射：字串轉整數
    difficulty_map = {
        "easy": 1,
        "medium": 2,
        "hard": 3
    }

    success_count = 0
    fail_count = 0

    print("開始遷移資料...")
    
    for word, positions in idioms.items():
        # 轉換資料格式以符合 IdiomDBManager 的要求
        formatted_data = {}
        for pos, diff_dict in positions.items():
            formatted_data[pos] = {}
            for diff_str, chars in diff_dict.items():
                diff_int = difficulty_map[diff_str]
                formatted_data[pos][diff_int] = chars
        
        # 寫入資料庫
        if db.add_idiom(word, formatted_data):
            success_count += 1
        else:
            fail_count += 1
            print(f"略過：{word} (可能已存在)")

    print(f"遷移完成！成功寫入 {success_count} 筆，略過 {fail_count} 筆。")

if __name__ == "__main__":
    run_migration()