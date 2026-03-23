from dataBase import IdiomDBManager

# 0. 初始化 DB 管理器
db = IdiomDBManager('sqlite:///idiom_game.db')

print("=== 1. Create (新增) ===")
# 新增題目
new_data = {
    0: { 1: ["十", "八"], 3: ["丸"] },
    2: { 2: ["二"] }
}
is_created = db.add_idiom("九牛一毛", new_data)
print(f"新增 '九牛一毛': {'成功' if is_created else '失敗 (可能已存在)'}")


print("\n=== 2. Read (讀取) ===")
# 遊戲主程式需要抓取特定成語的資料來出題
idiom_data = db.get_idiom("九牛一毛")
print("讀取結果:", idiom_data)


print("\n=== 3. Update (更新/擴充選項) ===")
# 九牛一毛」的第 1 個字 (牛) 增加一個 Hard 難度的干擾字 (午)
is_updated = db.add_single_distractor("九牛一毛", pos=1, difficulty=3, char="午")
print(f"擴充干擾選項: {'成功' if is_updated else '成語不存在'}")

# 再次讀取確認
updated_data = db.get_idiom("九牛一毛")
print("更新後選項數量:", len(updated_data["distractors"]))


print("\n=== 4. Delete (刪除) ===")
# 刪除這題 (Cascade 設定會自動清空對應的干擾字)
is_deleted = db.delete_idiom("九牛一毛")
print(f"刪除 '九牛一毛': {'成功' if is_deleted else '失敗'}")