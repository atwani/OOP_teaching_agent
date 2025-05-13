import sqlite3

class ChatDB:
    def __init__(self, db_name="chatbot.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_table()


    def create_connection(self):
        return sqlite3.connect("chatbot.db", check_same_thread=False)
    
    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_conversation(self, session_id, user_message, bot_response):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (session_id, user_message, bot_response) VALUES (?, ?, ?)",
            (session_id, user_message, bot_response)
        )
        self.conn.commit()

    def get_conversations(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT user_message, bot_response, timestamp FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        return cursor.fetchall()
