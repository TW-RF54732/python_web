from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

class Idiom(Base):
    __tablename__ = 'idioms'
    id = Column(Integer, primary_key=True)
    word = Column(String, unique=True, nullable=False, index=True)
    distractors = relationship("Distractor", back_populates="idiom", cascade="all, delete-orphan")

class Distractor(Base):
    __tablename__ = 'distractors'
    id = Column(Integer, primary_key=True)
    idiom_id = Column(Integer, ForeignKey('idioms.id'), nullable=False)
    pos = Column(Integer, nullable=False)
    difficulty = Column(Integer, nullable=False)  # 依你的設定改為 Integer
    incorrect_char = Column(String, nullable=False)
    idiom = relationship("Idiom", back_populates="distractors")

class IdiomDBManager:
    def __init__(self, db_url='sqlite:///idiom_game.db'):
        """初始化資料庫與 Session"""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    # --- Create ---
    def add_idiom(self, word: str, data_dict: dict) -> bool:
        """
        新增成語與其干擾項
        data_dict 格式: { 索引 : { 難度: [chars] } }
        範例: { 0: { 1: ["努", "怒"], 2: ["弩"] } }
        """
        with self.Session() as session:
            if session.query(Idiom).filter_by(word=word).first():
                return False  # 成語已存在
            
            new_idiom = Idiom(word=word)
            for pos, diff_dict in data_dict.items():
                for diff, chars in diff_dict.items():
                    for char in chars:
                        new_idiom.distractors.append(
                            Distractor(pos=pos, difficulty=diff, incorrect_char=char)
                        )
            session.add(new_idiom)
            session.commit()
            return True

    # --- Read ---
    def get_idiom(self, word: str) -> dict:
        """讀取特定成語及其所有干擾項"""
        with self.Session() as session:
            idiom = session.query(Idiom).filter_by(word=word).first()
            if not idiom:
                return None
            
            result = {"word": idiom.word, "distractors": []}
            for d in idiom.distractors:
                result["distractors"].append({
                    "pos": d.pos,
                    "difficulty": d.difficulty,
                    "char": d.incorrect_char
                })
            return result

    # --- Update ---
    def add_single_distractor(self, word: str, pos: int, difficulty: int, char: str) -> bool:
        """為已存在的成語增加單一干擾項"""
        with self.Session() as session:
            idiom = session.query(Idiom).filter_by(word=word).first()
            if not idiom:
                return False
            
            idiom.distractors.append(
                Distractor(pos=pos, difficulty=difficulty, incorrect_char=char)
            )
            session.commit()
            return True

    # --- Delete ---
    def delete_idiom(self, word: str) -> bool:
        """刪除成語 (Cascade 設定會連帶刪除所有干擾項)"""
        with self.Session() as session:
            idiom = session.query(Idiom).filter_by(word=word).first()
            if idiom:
                session.delete(idiom)
                session.commit()
                return True
            return False