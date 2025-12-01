import sqlite3
import os

DB_PATH = 'database/gym_security.db'

def migrate_db():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("🔄 Starting migration...")
        
        # 1. Rename old table
        cursor.execute("ALTER TABLE members RENAME TO members_old")
        
        # 2. Create new table (without UNIQUE NOT NULL on email)
        cursor.execute("""
        CREATE TABLE members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firstName TEXT NOT NULL,
            lastName TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            membershipType TEXT DEFAULT 'standard',
            status TEXT DEFAULT 'active',
            photoPath TEXT NOT NULL,
            faceDescriptor TEXT NOT NULL,
            registeredAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            lastAccess DATETIME,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 3. Copy data
        # We need to list columns to be safe, but assuming structure matches except constraints
        cursor.execute("""
        INSERT INTO members (id, firstName, lastName, email, phone, membershipType, status, photoPath, faceDescriptor, registeredAt, lastAccess, createdAt, updatedAt)
        SELECT id, firstName, lastName, email, phone, membershipType, status, photoPath, faceDescriptor, registeredAt, lastAccess, createdAt, updatedAt
        FROM members_old
        """)
        
        # 4. Drop old table
        cursor.execute("DROP TABLE members_old")
        
        # 5. Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_email ON members(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_status ON members(status)")
        
        conn.commit()
        print("✅ Migration successful! Email is now optional.")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_db()
