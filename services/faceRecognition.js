const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const logger = require('../utils/logger');
const axios = require('axios');

class FaceRecognition {
  constructor() {
    this.modelsLoaded = false;
    this.pythonScript = path.join(__dirname, '..', 'face_bridge.py');
    this.minConfidence = parseFloat(process.env.MIN_FACE_CONFIDENCE) || 0.5;
    this.matchThreshold = parseFloat(process.env.FACE_MATCH_THRESHOLD) || 0.70;
    this.apiUrl = 'http://localhost:5001'; // ArcFace API server
    this.apiProcess = null;
  }

  async initialize() {
    try {
      // Start the ArcFace Recognition API server
      logger.info('Starting ArcFace Recognition API server...');

      const pythonPath = 'C:\\Users\\Asus\\miniconda3\\python.exe';
      const apiScript = path.join(__dirname, '..', 'arcface_api.py');

      this.apiProcess = spawn(pythonPath, [apiScript], {
        detached: false,
        stdio: ['ignore', 'pipe', 'pipe']
      });

      this.apiProcess.stdout.on('data', (data) => {
        logger.info(`ArcFace API: ${data.toString().trim()}`);
      });

      this.apiProcess.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (!msg.includes('WARNING') && !msg.includes('Applied providers')) {
          logger.warn(`ArcFace API: ${msg}`);
        }
      });

      // Wait for API to start
      await this.waitForApi();

      this.modelsLoaded = true;
      logger.info('✅ ArcFace Recognition API ready!');
      logger.info('Using InsightFace ArcFace (buffalo_l) for maximum accuracy');
    } catch (error) {
      logger.error('Error initializing ArcFace Recognition API:', error);
      throw new Error('Failed to initialize face recognition.');
    }
  }

  async waitForApi(maxAttempts = 30) {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await axios.get(`${this.apiUrl}/health`, { timeout: 1000 });
        if (response.data.status === 'ok') {
          logger.info(`✅ Face Recognition API connected (attempt ${i + 1}/${maxAttempts})`);
          return true;
        }
      } catch (error) {
        if (i % 5 === 0) {
          logger.info(`⏳ Waiting for Face API... (attempt ${i + 1}/${maxAttempts})`);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    throw new Error('Face Recognition API failed to start after ' + maxAttempts + ' seconds');
  }

  async callApiWithRetry(endpoint, data, retries = 3) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await axios.post(`${this.apiUrl}${endpoint}`, data, {
          timeout: 30000, // Increased to 30s
          validateStatus: (status) => status < 500 // Don't throw on 4xx errors
        });
        return response;
      } catch (error) {
        if (attempt === retries) {
          logger.error(`❌ API call failed after ${retries} attempts: ${error.message}`);
          throw error;
        }
        logger.warn(`⚠️ API call attempt ${attempt} failed, retrying...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
      }
    }
  }

  async extractFaceDescriptor(imagePath) {
    if (!this.modelsLoaded) {
      throw new Error('Face recognition models not loaded. Please ensure the API server is running.');
    }

    try {
      // Check if image exists
      if (!fs.existsSync(imagePath)) {
        logger.warn(`❌ Image file not found: ${imagePath} `);
        return { success: false, error: 'Image file not found' };
      }

      // Validate file size
      const stats = fs.statSync(imagePath);
      if (stats.size < 1000) {
        logger.warn(`❌ Image file too small: ${imagePath} `);
        return { success: false, error: 'Image file is too small or corrupted' };
      }

      // Read image and convert to base64
      const imageBuffer = fs.readFileSync(imagePath);
      const base64Image = imageBuffer.toString('base64');

      // Use ArcFace API for embedding extraction with retry
      const response = await this.callApiWithRetry('/extract', {
        image: base64Image
      });

      if (response.data.success && response.data.embedding) {
        logger.info(`✅ Extracted ${response.data.dimensions} -dim embedding from ${path.basename(imagePath)} `);
        return response.data.embedding;
      } else {
        const errorMsg = response.data.error || 'Unknown error';
        logger.warn(`⚠️ Face extraction failed from ${imagePath}: ${errorMsg} `);
        return { success: false, error: errorMsg };
      }
    } catch (error) {
      logger.error(`❌ Error extracting face descriptor: ${error.message} `);
      return { success: false, error: 'Failed to process image: ' + error.message };
    }
  }

  async extractFaceDescriptorFromBase64(base64Data) {
    if (!this.modelsLoaded) {
      throw new Error('Face recognition models not loaded. Please restart the server.');
    }

    try {
      // Validate input
      if (!base64Data || base64Data.length < 100) {
        logger.error('❌ Invalid or empty image data received');
        return { match: false, error: 'Invalid image data' };
      }

      // Use ArcFace API for verification with retry logic
      const response = await this.callApiWithRetry('/verify', {
        image: base64Data
      });

      if (response.data.success && response.data.match) {
        const confidence = (response.data.confidence * 100).toFixed(1);
        logger.info(`✅ MATCH FOUND via ArcFace: ${response.data.name} (confidence: ${confidence}%)`);
        return {
          match: true,
          member_id: response.data.member_id,
          name: response.data.name,
          distance: response.data.distance,
          confidence: response.data.confidence
        };
      } else if (response.data.success && !response.data.match) {
        logger.info(`ℹ️ No matching face found in ArcFace database`);
        return { match: false, message: 'No matching face found' };
      } else {
        const errorMsg = response.data.error || 'Face verification failed';
        logger.warn(`⚠️ Verification failed: ${errorMsg} `);
        return { match: false, error: errorMsg };
      }
    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        logger.error('❌ ArcFace Recognition API is not running. Please start the Python API server.');
        return { match: false, error: 'Face recognition service unavailable. Please contact administrator.' };
      }
      logger.error(`❌ Error verifying face: ${error.message} `);
      return { match: false, error: 'Face verification failed: ' + error.message };
    }
  }

  calculateEuclideanDistance(descriptor1, descriptor2) {
    if (!descriptor1 || !descriptor2) {
      return Infinity;
    }

    if (descriptor1.length !== descriptor2.length) {
      logger.warn(`Descriptor length mismatch: ${descriptor1.length} vs ${descriptor2.length} `);
      return Infinity;
    }

    let sum = 0;
    for (let i = 0; i < descriptor1.length; i++) {
      const diff = descriptor1[i] - descriptor2[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  }

  async findMatch(descriptor, members) {
    if (!descriptor || members.length === 0) {
      return null;
    }

    let bestMatch = null;
    let bestDistance = Infinity;
    const distances = [];

    for (const member of members) {
      try {
        // Parse stored face descriptor
        const storedDescriptor = JSON.parse(member.faceDescriptor);

        // Calculate distance between descriptors
        const distance = this.calculateEuclideanDistance(descriptor, storedDescriptor);

        distances.push({
          id: member.id,
          name: `${member.firstName} ${member.lastName} `,
          distance: distance.toFixed(4)
        });

        // Check if this is the best match so far
        if (distance < bestDistance && distance < 1.0) {
          bestDistance = distance;
          bestMatch = member;
        }
      } catch (error) {
        logger.error(`Error comparing with member ${member.id}: `, error);
      }
    }

    // Log all distances for debugging
    logger.info(`Distance comparison: ${JSON.stringify(distances)} `);

    if (bestMatch) {
      // Convert distance to confidence percentage
      const confidence = Math.max(0, Math.min(100, (1 - bestDistance) * 100));

      return {
        ...bestMatch,
        confidence: confidence.toFixed(2),
        distance: bestDistance.toFixed(4)
      };
    }

    return null;
  }

  async detectMultipleFaces(imagePath) {
    // Simplified - single face detection only
    const descriptor = await this.extractFaceDescriptor(imagePath);
    if (descriptor) {
      return [{
        descriptor: descriptor,
        confidence: 0.9,
        box: { x: 0, y: 0, width: 100, height: 100 }
      }];
    }
    return [];
  }

  // Validate face quality before registration
  async validateFaceQuality(imagePath) {
    try {
      // Basic validation - check if file exists and is readable
      if (!fs.existsSync(imagePath)) {
        return {
          valid: false,
          reason: 'Image file not found'
        };
      }

      const stats = fs.statSync(imagePath);

      // Check file size (should be between 10KB and 10MB)
      if (stats.size < 10000) {
        return {
          valid: false,
          reason: 'Image file too small'
        };
      }

      if (stats.size > 10485760) {
        return {
          valid: false,
          reason: 'Image file too large'
        };
      }

      return {
        valid: true,
        confidence: 0.9
      };
    } catch (error) {
      logger.error('Error validating face quality:', error);
      return {
        valid: false,
        reason: 'Error processing image'
      };
    }
  }
}

module.exports = FaceRecognition;
