from app import get_connection

def fix():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("ALTER USER fakenews QUOTA UNLIMITED ON USERS")
        conn.commit()
        print("Quota updated")
        cur.close()
        conn.close()
    except Exception as e:
        print("Could not alter user (maybe lacking admin privs?):", e)

if __name__ == "__main__":
    fix()
