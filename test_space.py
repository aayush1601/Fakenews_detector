from app import get_connection

def q():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT table_name, tablespace_name FROM user_tables")
    for r in cur.fetchall():
        print(r)

if __name__ == "__main__":
    q()
