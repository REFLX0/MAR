const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');
const logger = require('../utils/logger');

class Database {
  constructor() {
    const dbDir = path.join(__dirname, '../database');
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }

    this.dbPath = process.env.DB_PATH || path.join(dbDir, 'gym_security.db');
    this.db = null;
  }

  async initialize() {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database(this.dbPath, (err) => {
        if (err) {
          logger.error('Error opening database:', err);
          reject(err);
        } else {
          logger.info('Connected to SQLite database');
          this.createTables()
            .then(resolve)
            .catch(reject);
        }
      });
    });
  }

  async createTables() {
    const tables = [
      // Members table
      `CREATE TABLE IF NOT EXISTS members (
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
      )`,

      // Access logs table
      `CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memberId INTEGER,
        status TEXT NOT NULL,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (memberId) REFERENCES members(id) ON DELETE SET NULL
      )`,

      // Audit logs table
      `CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        entityType TEXT NOT NULL,
        entityId INTEGER,
        details TEXT,
        performedBy TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // Create indexes
      `CREATE INDEX IF NOT EXISTS idx_members_email ON members(email)`,
      `CREATE INDEX IF NOT EXISTS idx_members_status ON members(status)`,
      `CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON access_logs(timestamp)`,
      `CREATE INDEX IF NOT EXISTS idx_access_logs_status ON access_logs(status)`
    ];

    for (const sql of tables) {
      await this.run(sql);
    }

    logger.info('Database tables created successfully');
  }

  run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function (err) {
        if (err) {
          logger.error('Database run error:', err);
          reject(err);
        } else {
          resolve(this);
        }
      });
    });
  }

  get(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) {
          logger.error('Database get error:', err);
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  all(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) {
          logger.error('Database all error:', err);
          reject(err);
        } else {
          resolve(rows);
        }
      });
    });
  }

  // ==================== MEMBER OPERATIONS ====================

  async createMember(memberData) {
    const sql = `
      INSERT INTO members (firstName, lastName, email, phone, membershipType, photoPath, faceDescriptor)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    const result = await this.run(sql, [
      memberData.firstName,
      memberData.lastName,
      memberData.email,
      memberData.phone,
      memberData.membershipType,
      memberData.photoPath,
      memberData.faceDescriptor
    ]);

    await this.logAudit('CREATE', 'member', result.lastID,
      `New member registered: ${memberData.firstName} ${memberData.lastName}`, 'system');

    return result.lastID;
  }

  async getMemberById(id) {
    const sql = 'SELECT * FROM members WHERE id = ?';
    return await this.get(sql, [id]);
  }

  async getMemberByEmail(email) {
    const sql = 'SELECT * FROM members WHERE email = ? AND status = ?';
    return await this.get(sql, [email, 'active']);
  }

  async getAllMembers() {
    const sql = 'SELECT * FROM members WHERE status = ? ORDER BY lastName, firstName';
    return await this.all(sql, ['active']);
  }

  async updateMember(id, updates) {
    const fields = [];
    const values = [];

    for (const [key, value] of Object.entries(updates)) {
      if (value !== undefined) {
        fields.push(`${key} = ?`);
        values.push(value);
      }
    }

    if (fields.length === 0) {
      return false;
    }

    fields.push('updatedAt = CURRENT_TIMESTAMP');
    values.push(id);

    const sql = `UPDATE members SET ${fields.join(', ')} WHERE id = ?`;
    const result = await this.run(sql, values);

    if (result.changes > 0) {
      await this.logAudit('UPDATE', 'member', id,
        `Member updated: ${JSON.stringify(updates)}`, 'admin');
    }

    return result.changes > 0;
  }

  async deleteMember(id) {
    const member = await this.getMemberById(id);

    if (!member) {
      return false;
    }

    // Soft delete by setting status to inactive
    const sql = 'UPDATE members SET status = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?';
    const result = await this.run(sql, ['inactive', id]);

    if (result.changes > 0) {
      await this.logAudit('DELETE', 'member', id,
        `Member deactivated: ${member.firstName} ${member.lastName}`, 'admin');
    }

    return result.changes > 0;
  }

  async updateLastAccess(memberId) {
    const sql = 'UPDATE members SET lastAccess = CURRENT_TIMESTAMP WHERE id = ?';
    return await this.run(sql, [memberId]);
  }

  async updateMemberFaceDescriptor(memberId, faceDescriptor) {
    const sql = 'UPDATE members SET faceDescriptor = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?';
    return await this.run(sql, [faceDescriptor, memberId]);
  }

  // ==================== MEMBERSHIP OPERATIONS ====================

  async createMembership(membershipData) {
    // For SQLite, store membership info directly in members table
    // This is a simplified version - Supabase has a separate abbonnement table
    const sql = `
      UPDATE members 
      SET membershipType = ?, 
          updatedAt = CURRENT_TIMESTAMP 
      WHERE id = ?
    `;

    await this.run(sql, [
      membershipData.plan || 'standard',
      membershipData.memberId
    ]);

    logger.info(`Membership created for member ${membershipData.memberId}: ${membershipData.plan}`);
    return membershipData.memberId;
  }

  async getMembershipStatus(memberId) {
    // For SQLite, we don't track expiration in a separate table
    // Return a default active status
    const member = await this.getMemberById(memberId);

    if (!member) {
      return { status: 'unknown', daysLeft: 0, expired: false };
    }

    // Simplified - always return active for SQLite
    return {
      status: 'active',
      plan: member.membershipType || 'standard',
      endDate: null,
      daysLeft: 9999,
      expired: false
    };
  }

  async updateLastAccess(memberId) {
    const sql = 'UPDATE members SET lastAccess = CURRENT_TIMESTAMP WHERE id = ?';
    return await this.run(sql, [memberId]);
  }

  // ==================== ACCESS LOG OPERATIONS ====================

  async logAccessAttempt(memberId, status, message) {
    const sql = `
      INSERT INTO access_logs (memberId, status, message)
      VALUES (?, ?, ?)
    `;

    return await this.run(sql, [memberId, status, message]);
  }

  async getAccessLogs(limit = 50) {
    const sql = `
      SELECT 
        al.*,
        m.firstName,
        m.lastName,
        m.photoPath
      FROM access_logs al
      LEFT JOIN members m ON al.memberId = m.id
      ORDER BY al.timestamp DESC
      LIMIT ?
    `;

    return await this.all(sql, [limit]);
  }

  // ==================== STATISTICS ====================

  async getDashboardStats() {
    const totalMembers = await this.get(
      'SELECT COUNT(*) as count FROM members WHERE status = ?',
      ['active']
    );

    const todayAccess = await this.get(
      `SELECT COUNT(*) as count FROM access_logs 
       WHERE status = ? AND DATE(timestamp) = DATE('now')`,
      ['granted']
    );

    const todayDenied = await this.get(
      `SELECT COUNT(*) as count FROM access_logs 
       WHERE status = ? AND DATE(timestamp) = DATE('now')`,
      ['denied']
    );

    const recentMembers = await this.all(
      `SELECT * FROM members WHERE status = ? 
       ORDER BY registeredAt DESC LIMIT 5`,
      ['active']
    );

    const membershipBreakdown = await this.all(
      `SELECT membershipType, COUNT(*) as count 
       FROM members WHERE status = ? 
       GROUP BY membershipType`,
      ['active']
    );

    return {
      totalMembers: totalMembers.count,
      todayAccess: todayAccess.count,
      todayDenied: todayDenied.count,
      recentMembers: recentMembers,
      membershipBreakdown: membershipBreakdown
    };
  }

  // ==================== AUDIT LOG ====================

  async logAudit(action, entityType, entityId, details, performedBy) {
    const sql = `
      INSERT INTO audit_logs (action, entityType, entityId, details, performedBy)
      VALUES (?, ?, ?, ?, ?)
    `;

    return await this.run(sql, [action, entityType, entityId, details, performedBy]);
  }

  async getAuditLogs(limit = 100) {
    const sql = 'SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?';
    return await this.all(sql, [limit]);
  }

  // ==================== CLEANUP ====================

  close() {
    return new Promise((resolve, reject) => {
      this.db.close((err) => {
        if (err) {
          logger.error('Error closing database:', err);
          reject(err);
        } else {
          logger.info('Database connection closed');
          resolve();
        }
      });
    });
  }
}

module.exports = Database;
