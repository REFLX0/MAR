const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const logger = require('../utils/logger');

class AuthService {
  constructor() {
    this.jwtSecret = process.env.JWT_SECRET || 'default-secret-change-this';
    this.tokenExpiry = '24h';
    
    // In production, these should be stored in a database with hashed passwords
    this.adminUsername = process.env.ADMIN_USERNAME || 'admin';
    this.adminPasswordHash = bcrypt.hashSync(process.env.ADMIN_PASSWORD || 'admin123', 10);
  }

  login(username, password) {
    try {
      // Verify credentials
      if (username !== this.adminUsername) {
        return null;
      }

      const isValid = bcrypt.compareSync(password, this.adminPasswordHash);
      
      if (!isValid) {
        return null;
      }

      // Generate JWT token
      const token = jwt.sign(
        { 
          username: username,
          role: 'admin',
          timestamp: Date.now()
        },
        this.jwtSecret,
        { expiresIn: this.tokenExpiry }
      );

      return token;
    } catch (error) {
      logger.error('Login error:', error);
      return null;
    }
  }

  verifyToken(token) {
    try {
      const decoded = jwt.verify(token, this.jwtSecret);
      return decoded;
    } catch (error) {
      logger.error('Token verification error:', error);
      return null;
    }
  }

  hashPassword(password) {
    return bcrypt.hashSync(password, 10);
  }

  comparePassword(password, hash) {
    return bcrypt.compareSync(password, hash);
  }
}

module.exports = AuthService;
