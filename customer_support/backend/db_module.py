import mysql.connector


db_config = {
    "host": "localhost",
    "user": "root",
    "password": "pass",
    "database": "support_db"
}

def get_agent_email():
    
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM agents ORDER BY RAND() LIMIT 1;")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None
