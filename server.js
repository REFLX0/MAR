require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const Database = require('./database/db');
const SupabaseDatabase = require('./database/supabase');
const FaceRecognition = require('./services/faceRecognition');
const AuthService = require('./services/authService');
const logger = require('./utils/logger');

const app = express();
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '0.0.0.0'; // Listen on all network interfaces

// Initialize services - Use Supabase if credentials are available
let db;
if (process.env.SUPABASE_URL && process.env.SUPABASE_ANON_KEY) {
  logger.info('Using Supabase database');
  db = new SupabaseDatabase();
} else {
  logger.info('Using SQLite database');
  db = new Database();
}

const faceRecognition = new FaceRecognition();
const authService = new AuthService();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false // Allow inline scripts for demo purposes
}));
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Serve static files
app.use(express.static('public'));
app.use('/models', express.static('models'));
app.use('/uploads', express.static('uploads'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'face-' + uniqueSuffix + path.extname(file.originalname));
  }
});
const upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// ==================== AUTHENTICATION ROUTES ====================

// Admin login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    const token = authService.login(username, password);

    if (!token) {
      logger.warn(`Failed login attempt for username: ${username}`);
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    logger.info(`Admin logged in: ${username}`);
    res.json({ token, message: 'Login successful' });
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Verify token
app.post('/api/auth/verify', (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const decoded = authService.verifyToken(token);

    if (!decoded) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    res.json({ valid: true, user: decoded });
  } catch (error) {
    logger.error('Token verification error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// ==================== MEMBER MANAGEMENT ROUTES ====================

// Get all members
app.get('/api/members', authenticateToken, async (req, res) => {
  try {
    logger.info('Fetching all members...');
    const members = await db.getAllMembers();
    logger.info(`Successfully fetched ${members.length} members`);
    res.json({ members, count: members.length });
  } catch (error) {
    logger.error('Error fetching members:', error);
    logger.error('Error stack:', error.stack);
    logger.error('Error message:', error.message);
    res.status(500).json({ error: 'Failed to fetch members', details: error.message });
  }
});

// Get member by ID
app.get('/api/members/:id', authenticateToken, async (req, res) => {
  try {
    const member = await db.getMemberById(req.params.id);

    if (!member) {
      return res.status(404).json({ error: 'Member not found' });
    }

    res.json(member);
  } catch (error) {
    logger.error('Error fetching member:', error);
    res.status(500).json({ error: 'Failed to fetch member' });
  }
});

// Quick register (no authentication) - for multi-angle registration
app.post('/api/members/register-quick', upload.single('photo'), async (req, res) => {
  try {
    const { firstName, lastName, email, phone, membershipType } = req.body;

    // Validate required fields
    if (!firstName || !lastName || !email) {
      return res.status(400).json({ error: 'First name, last name, and email are required' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'Photo is required' });
    }

    // Save photo path
    const photoPath = req.file.path;

    // Create member in database immediately
    const memberId = await db.createMember({
      firstName,
      lastName,
      email,
      phone: phone || null,
      membershipType: membershipType || 'standard',
      photoPath: photoPath,
      faceDescriptor: JSON.stringify([1]) // Dummy descriptor
    });

    logger.info(`New member registered: ${firstName} ${lastName} (ID: ${memberId})`);

    res.status(201).json({
      message: 'Member registered successfully',
      memberId: memberId,
      member: {
        id: memberId,
        firstName,
        lastName,
        email,
        photoPath
      }
    });
  } catch (error) {
    logger.error('Error registering member:', error);
    res.status(500).json({ error: 'Failed to register member: ' + error.message });
  }
});

// Register new member (with authentication) - 5 PHOTO SUPPORT
app.post('/api/members/register', authenticateToken, upload.fields([
  { name: 'center', maxCount: 1 },
  { name: 'left', maxCount: 1 },
  { name: 'right', maxCount: 1 },
  { name: 'up', maxCount: 1 },
  { name: 'down', maxCount: 1 }
]), async (req, res) => {
  try {
    logger.info('=== REGISTRATION REQUEST RECEIVED ===');

    const { firstName, lastName, email, phone, membershipType } = req.body;

    // Validate required fields
    if (!firstName || !lastName || !email) {
      logger.error('Validation failed: Missing required fields');
      return res.status(400).json({ error: 'First name, last name, and email are required' });
    }

    // Check for photos
    if (!req.files || !req.files['center']) {
      logger.error('Validation failed: Center photo is required');
      return res.status(400).json({ error: 'Center photo is required' });
    }

    // Collect photo paths
    const photos = {};
    const angles = ['center', 'left', 'right', 'up', 'down'];
    let photoCount = 0;

    angles.forEach(angle => {
      if (req.files[angle] && req.files[angle][0]) {
        photos[angle] = req.files[angle][0].path;
        photoCount++;
      }
    });

    logger.info(`Received ${photoCount} photos`);

    // Use center photo as main profile photo
    const photoPath = photos['center'];
    logger.info('Main photo saved at:', photoPath);

    // Check for duplicate email
    const existingMember = await db.getMemberByEmail(email);
    if (existingMember) {
      return res.status(409).json({
        success: false,
        error: `Email "${email}" is already registered.`
      });
    }

    // Create member in database
    const memberId = await db.createMember({
      firstName,
      lastName,
      email,
      phone: phone || null,
      membershipType: membershipType || 'standard',
      photoPath: photoPath,
      faceDescriptor: JSON.stringify([1]) // Dummy descriptor
    });

    // Create membership
    const membershipDuration = parseInt(req.body.membershipDuration) || 30;
    const membershipPrice = parseFloat(req.body.membershipPrice) || 0;
    const endDate = new Date();
    endDate.setDate(endDate.getDate() + membershipDuration);

    try {
      await db.createMembership({
        memberId: memberId,
        plan: membershipType || 'standard',
        status: 'active',
        endDate: endDate,
        price: membershipPrice
      });
    } catch (membershipError) {
      logger.error('Failed to create membership:', membershipError);
    }

    logger.info(`✅ Member registered: ${firstName} ${lastName} (ID: ${memberId})`);

    // ========== TRAINING DATA INTEGRATION ==========
    try {
      const { spawn } = require('child_process');
      const pythonPath = 'C:\\Users\\Asus\\miniconda3\\python.exe';
      const registerScript = path.join(__dirname, 'register_member.py');

      logger.info('🎬 Starting registration pipeline with 5 photos...');

      // Prepare arguments: script, id, first, last, center, left, right, up, down
      const args = [
        registerScript,
        memberId.toString(),
        firstName,
        lastName,
        photos['center'] || '',
        photos['left'] || '',
        photos['right'] || '',
        photos['up'] || '',
        photos['down'] || ''
      ];

      const registerProcess = spawn(pythonPath, args, {
        detached: false,
        stdio: 'pipe'
      });

      registerProcess.stdout.on('data', (data) => {
        logger.info(`Pipeline: ${data.toString().trim()}`);
      });

      registerProcess.stderr.on('data', (data) => {
        const msg = data.toString();
        if (!msg.includes('WARNING')) {
          logger.warn(`Pipeline: ${msg.trim()}`);
        }
      });

    } catch (trainingError) {
      logger.error('⚠️ Training data integration failed:', trainingError);
    }

    res.status(201).json({
      success: true,
      message: 'Member registered successfully',
      memberId: memberId,
      trainingStatus: 'Registration pipeline in progress'
    });
  } catch (error) {
    logger.error('Registration error:', error);
    res.status(500).json({ error: 'Failed to register member' });
  }
});

// Update member
app.put('/api/members/:id', authenticateToken, async (req, res) => {
  try {
    const { firstName, lastName, email, phone, membershipType, status } = req.body;

    const updated = await db.updateMember(req.params.id, {
      firstName,
      lastName,
      email,
      phone,
      membershipType,
      status
    });

    if (!updated) {
      return res.status(404).json({ error: 'Member not found' });
    }

    res.json({ message: 'Member updated successfully' });
  } catch (error) {
    logger.error('Error updating member:', error);
    res.status(500).json({ error: 'Failed to update member' });
  }
});

// Delete member
app.delete('/api/members/:id', authenticateToken, async (req, res) => {
  try {
    const memberId = req.params.id;
    const deleted = await db.deleteMember(memberId);

    if (!deleted) {
      return res.status(404).json({ error: 'Member not found' });
    }

    logger.info(`Member deleted from DB: ID ${memberId}`);

    // ========== CLEANUP TRAINING DATA ==========
    try {
      const { spawn } = require('child_process');
      const pythonPath = 'C:\\Users\\Asus\\miniconda3\\python.exe';
      const deleteScript = path.join(__dirname, 'delete_member.py');

      logger.info('🗑️ Starting cleanup pipeline...');

      const deleteProcess = spawn(pythonPath, [deleteScript, memberId], {
        detached: false,
        stdio: 'pipe'
      });

      deleteProcess.stdout.on('data', (data) => {
        logger.info(`Cleanup: ${data.toString().trim()}`);
      });

      deleteProcess.stderr.on('data', (data) => {
        logger.warn(`Cleanup Error: ${data.toString().trim()}`);
      });

    } catch (cleanupError) {
      logger.error('⚠️ Cleanup pipeline failed:', cleanupError);
    }
    // ===========================================

    res.json({ message: 'Member deleted successfully' });
  } catch (error) {
    logger.error('Error deleting member:', error);
    res.status(500).json({ error: 'Failed to delete member' });
  }
});

// ==================== FACIAL RECOGNITION ROUTES ====================

// Verify face (scan for access) - Using YOLO + SFace with training_data
app.post('/api/face/verify', async (req, res) => {
  try {
    const { imageData } = req.body;

    if (!imageData) {
      return res.status(400).json({ error: 'Image data is required' });
    }

    // Verify face using YOLO + SFace API (training_data embeddings)
    logger.info('Verifying face with YOLO + SFace...');
    const result = await faceRecognition.extractFaceDescriptorFromBase64(imageData);

    if (result && result.match) {
      const memberId = result.member_id;
      const memberName = result.name;
      const confidence = (result.confidence * 100).toFixed(1);

      // Try to find member in database by ID
      let member = null;
      try {
        member = await db.getMemberById(memberId);
      } catch (err) {
        logger.warn(`Member lookup failed for ID ${memberId}: ${err.message}`);
      }

      if (member) {
        // Check membership expiration
        let membershipStatus;
        try {
          membershipStatus = await db.getMembershipStatus(member.id);
        } catch (err) {
          logger.error('Error checking membership:', err);
          membershipStatus = { status: 'unknown', daysLeft: 0, expired: false };
        }

        if (membershipStatus.expired) {
          logger.warn(`❌ Membership EXPIRED for ${member.firstName} ${member.lastName}`);
          return res.json({
            verified: false,
            expired: true,
            member: {
              id: member.id,
              firstName: member.firstName,
              lastName: member.lastName,
              photoPath: member.photoPath
            },
            message: `⚠️ MEMBERSHIP EXPIRED!\n${member.firstName} ${member.lastName}\nPlease renew your subscription.`,
            expiredDate: membershipStatus.endDate
          });
        }

        logger.info(`✓ Access granted: ${member.firstName} ${member.lastName} (Confidence: ${confidence}%)`);
        await db.logAccessAttempt(member.id, 'granted', 'Face recognized');
        await db.updateLastAccess(member.id);

        // Warning if expiring soon
        let expiryWarning = '';
        if (membershipStatus.daysLeft <= 7 && membershipStatus.daysLeft > 0) {
          expiryWarning = `\n⚠️ Membership expires in ${membershipStatus.daysLeft} day${membershipStatus.daysLeft > 1 ? 's' : ''}!`;
        }

        res.json({
          verified: true,
          member: {
            id: member.id,
            firstName: member.firstName,
            lastName: member.lastName,
            membershipType: member.membershipType,
            photoPath: member.photoPath
          },
          confidence: confidence,
          membershipDaysLeft: membershipStatus.daysLeft,
          message: `Welcome back, ${member.firstName}!${expiryWarning}`
        });
      } else {
        // Member found in training_data but not in database
        // Still allow access with warning - face is recognized!
        logger.warn(`⚠️ Face recognized as ${memberName} (ID: ${memberId}) but not in database`);

        res.json({
          verified: true,
          partialMatch: true,
          member: {
            id: memberId,
            firstName: memberName.split(' ')[0] || memberName,
            lastName: memberName.split(' ').slice(1).join(' ') || '',
            membershipType: 'Unknown'
          },
          confidence: confidence,
          message: `Welcome, ${memberName}!\n\n⚠️ Please visit the front desk to update your membership record.`
        });
      }
    } else {
      // No face match found
      const errorMsg = result?.error || 'Face not recognized';
      logger.warn(`✗ Face not recognized: ${errorMsg}`);
      await db.logAccessAttempt(null, 'denied', 'Face not recognized');

      res.json({
        verified: false,
        error: errorMsg,
        message: 'Face not recognized.\nPlease register at the front desk.'
      });
    }
  } catch (error) {
    logger.error('Face verification error:', error);
    res.status(500).json({
      error: 'Failed to verify face',
      details: error.message,
      message: 'System error. Please try again or contact admin.'
    });
  }
});

// ==================== STATISTICS ROUTES ====================

// Get dashboard statistics
app.get('/api/stats/dashboard', authenticateToken, async (req, res) => {
  try {
    const stats = await db.getDashboardStats();
    res.json(stats);
  } catch (error) {
    logger.error('Error fetching dashboard stats:', error);
    res.status(500).json({ error: 'Failed to fetch statistics' });
  }
});

// Get access logs
app.get('/api/logs/access', authenticateToken, async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 50;
    const logs = await db.getAccessLogs(limit);
    res.json(logs);
  } catch (error) {
    logger.error('Error fetching access logs:', error);
    res.status(500).json({ error: 'Failed to fetch access logs' });
  }
});

// ==================== MIDDLEWARE ====================

function authenticateToken(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  const decoded = authService.verifyToken(token);

  if (!decoded) {
    return res.status(403).json({ error: 'Invalid or expired token' });
  }

  req.user = decoded;
  next();
}

// ==================== ERROR HANDLING ====================

app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({ error: 'An unexpected error occurred' });
});

// ==================== SERVER INITIALIZATION ====================

async function initializeServer() {
  try {
    // Initialize database
    await db.initialize();
    logger.info('Database initialized successfully');

    // Initialize face recognition models
    await faceRecognition.initialize();
    logger.info('Face recognition models loaded successfully');

    // Start server on all network interfaces
    app.listen(PORT, HOST, () => {
      // Get network address for external access
      const os = require('os');
      const networkInterfaces = os.networkInterfaces();
      let localIP = 'localhost';

      Object.keys(networkInterfaces).forEach((interfaceName) => {
        networkInterfaces[interfaceName].forEach((iface) => {
          if (iface.family === 'IPv4' && !iface.internal) {
            localIP = iface.address;
          }
        });
      });

      logger.info(`🚀 Gym Security Dashboard running on:`);
      logger.info(`   Local:    http://localhost:${PORT}`);
      logger.info(`   Network:  http://${localIP}:${PORT}`);
      logger.info(`📊 Admin Dashboard: http://${localIP}:${PORT}/admin.html`);
      logger.info(`🎯 Access Scanner: http://${localIP}:${PORT}/scanner.html`);

      console.log(`\n🚀 Gym Security Dashboard running on:`);
      console.log(`   Local:    http://localhost:${PORT}`);
      console.log(`   Network:  http://${localIP}:${PORT}`);
      console.log(`\n📊 Admin Dashboard: http://${localIP}:${PORT}/admin.html`);
      console.log(`🎯 Access Scanner: http://${localIP}:${PORT}/scanner.html`);
      console.log(`\n⚠️  To access from other devices on your network, use the Network URL\n`);
    });
  } catch (error) {
    logger.error('Failed to initialize server:', error);
    process.exit(1);
  }
}

// Start the server
initializeServer();

module.exports = app;
