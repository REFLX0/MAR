const { createClient } = require('@supabase/supabase-js');
const logger = require('../utils/logger');

class SupabaseDatabase {
  constructor() {
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Supabase URL and Key must be provided in environment variables');
    }

    this.supabase = createClient(supabaseUrl, supabaseKey);
    logger.info('Supabase client initialized');
  }

  async initialize() {
    try {
      // Create tables if they don't exist
      await this.createTables();
      logger.info('Supabase database initialized successfully');
    } catch (error) {
      logger.error('Error initializing Supabase database:', error);
      throw error;
    }
  }

  async createTables() {
    // Note: Using existing Supabase tables (users, visite, abbonnement)
    // This is just a check to ensure tables exist
    logger.info('Checking existing Supabase tables...');
    
    // Check if user table exists (Supabase uses singular)
    const { data: users, error: usersError } = await this.supabase
      .from('user')
      .select('count')
      .limit(1);
    
    if (usersError && usersError.code === '42P01') {
      logger.warn('Tables not found in Supabase. Please run the SQL schema to add required columns.');
    } else {
      logger.info('Supabase tables verified (user, visite, abbonnement)');
    }
  }

  // Members operations (using 'users' table)
  async createMember(memberData) {
    try {
      // Use RPC or direct SQL to bypass schema cache issue
      const { data, error } = await this.supabase.rpc('create_user_member', {
        p_name: memberData.lastName,
        p_first_name: memberData.firstName,
        p_email: memberData.email,
        p_phone: memberData.phone,
        p_photo_url: memberData.photoPath,
        p_face_descriptor: memberData.faceDescriptor
      });

      if (error) {
        logger.warn('RPC create_user_member not available, using direct insert:', error.message);
        
        // Fallback: Try direct insert with schema specification
        const { data: insertData, error: insertError } = await this.supabase
          .schema('public')
          .from('user')
          .insert({
            name: memberData.lastName,
            'first name': memberData.firstName,
            email: memberData.email,
            phone: memberData.phone,
            user_photo_url: memberData.photoPath,
            face_descriptor: memberData.faceDescriptor,
            created_at: new Date().toISOString()
          })
          .select()
          .single();

        if (insertError) {
          logger.error('Error creating member in Supabase:', insertError);
          throw insertError;
        }

        logger.info(`✅ Member created in Supabase: ${memberData.firstName} ${memberData.lastName}`);
        return {
          id: insertData.id_user || insertData.id,
          firstName: insertData['first name'] || memberData.firstName,
          lastName: insertData.name || memberData.lastName,
          email: insertData.email,
          phone: insertData.phone,
          photoPath: insertData.user_photo_url || memberData.photoPath,
          membershipType: 'standard'
        };
      }

      logger.info(`✅ Member created via RPC: ${memberData.firstName} ${memberData.lastName}`);
      return {
        id: data,
        firstName: memberData.firstName,
        lastName: memberData.lastName,
        email: memberData.email,
        phone: memberData.phone,
        photoPath: memberData.photoPath,
        membershipType: 'standard'
      };
    } catch (err) {
      logger.error('Critical error creating member:', err);
      throw err;
    }
  }

  async getMemberById(id) {
    const { data, error } = await this.supabase
      .from('user')
      .select('*')
      .eq('id_user', id)
      .single();

    if (error && error.code !== 'PGRST116') {
      logger.error('Error getting member from Supabase:', error);
      throw error;
    }

    if (!data) return null;

    return {
      id: data.id_user,
      firstName: data['first name'],
      lastName: data.name,
      email: data.email,
      phone: data.phone,
      membershipType: 'standard',
      photoPath: data.user_photo_url,
      faceDescriptor: '[]',
      status: 'active',
      createdAt: data.created_at
    };
  }

  async getAllMembers() {
    try {
      logger.info('📥 Fetching all members from Supabase...');
      
      // Try direct query with explicit schema (excluding face_descriptor if it doesn't exist)
      const { data, error } = await this.supabase
        .schema('public')
        .from('user')
        .select('id_user, "first name", name, email, phone, user_photo_url, created_at')
        .order('created_at', { ascending: false });

      if (error) {
        logger.error('❌ Supabase query error:', {
          message: error.message,
          details: error.details,
          hint: error.hint,
          code: error.code
        });
        
        // Try alternative approach with RPC if available
        try {
          logger.info('🔄 Trying alternative fetch method...');
          const { data: rpcData, error: rpcError } = await this.supabase.rpc('get_all_users');
          
          if (!rpcError && rpcData) {
            logger.info(`✅ Successfully retrieved ${rpcData.length} members via RPC`);
            return rpcData.map(user => ({
              id: user.id_user,
              firstName: user.first_name || user['first name'] || '',
              lastName: user.name || '',
              email: user.email || '',
              phone: user.phone || '',
              membershipType: 'standard',
              photoPath: user.user_photo_url || '',
              faceDescriptor: user.face_descriptor || '',
              status: 'active',
              createdAt: user.created_at
            }));
          }
        } catch (rpcErr) {
          logger.warn('⚠️ RPC method also failed:', rpcErr.message);
        }
        
        // If schema cache error, log warning but continue
        if (error.message && error.message.includes('schema cache')) {
          logger.warn('⚠️ Schema cache issue detected');
          logger.warn('💡 This usually auto-resolves within minutes');
        }
        
        logger.warn('⚠️ Returning empty array - check Supabase console for data');
        return [];
      }

      if (!data || data.length === 0) {
        logger.info('ℹ️ No members found in database');
        return [];
      }

      logger.info(`✅ Successfully retrieved ${data.length} members`);
      
      return data.map(user => ({
        id: user.id_user,
        firstName: user['first name'] || '',
        lastName: user.name || '',
        email: user.email || '',
        phone: user.phone || '',
        membershipType: 'standard',
        photoPath: user.user_photo_url || '',
        faceDescriptor: '[]',
        status: 'active',
        createdAt: user.created_at
      }));
    } catch (err) {
      logger.error('💥 Critical error in getAllMembers:', err);
      logger.error('Stack:', err.stack);
      // Return empty array instead of throwing to prevent complete failure
      return [];
    }
  }

  async updateMember(id, updates) {
    const updateData = {};
    if (updates.firstName) updateData['first name'] = updates.firstName;
    if (updates.lastName) updateData.name = updates.lastName;
    if (updates.email) updateData.email = updates.email;
    if (updates.phone) updateData.phone = updates.phone;
    if (updates.photoPath) updateData.user_photo_url = updates.photoPath;
    // face_descriptor column doesn't exist in Supabase

    const { data, error } = await this.supabase
      .from('user')
      .update(updateData)
      .eq('id_user', id)
      .select()
      .single();

    if (error) {
      logger.error('Error updating member in Supabase:', error);
      throw error;
    }

    return {
      id: data.id_user,
      firstName: data['first name'],
      lastName: data.name,
      email: data.email,
      phone: data.phone,
      membershipType: 'standard',
      photoPath: data.user_photo_url,
      faceDescriptor: '[]',
      status: 'active'
    };
  }

  async deleteMember(id) {
    const { error } = await this.supabase
      .from('user')
      .delete()
      .eq('id_user', id);

    if (error) {
      logger.error('Error deleting member from Supabase:', error);
      throw error;
    }

    logger.info(`Member ${id} deleted from Supabase`);
  }

  async getMemberByEmail(email) {
    const { data, error } = await this.supabase
      .from('user')
      .select('*')
      .eq('email', email)
      .single();

    if (error && error.code !== 'PGRST116') {
      logger.error('Error getting member by email from Supabase:', error);
      throw error;
    }

    if (!data) return null;

    return {
      id: data.id_user,
      firstName: data['first name'],
      lastName: data.name,
      email: data.email,
      phone: data.phone,
      membershipType: 'standard',
      photoPath: data.user_photo_url,
      faceDescriptor: '[]',
      status: 'active'
    };
  }

  // Access logs operations (using 'visite' table)
  async createAccessLog(logData) {
    const { data, error } = await this.supabase
      .from('visite')
      .insert([{
        id_user: logData.memberId,
        visited_at: new Date().toISOString()
      }])
      .select()
      .single();

    if (error) {
      logger.error('Error creating access log in Supabase:', error);
      throw error;
    }

    return {
      id: data.id,
      memberId: data.id_user,
      status: logData.status,
      message: logData.message,
      timestamp: data.visited_at
    };
  }

  async getAccessLogs(limit = 100) {
    try {
      // First get all visite records
      const { data: visites, error: visiteError } = await this.supabase
        .from('visite')
        .select('*')
        .order('visited_at', { ascending: false })
        .limit(limit);

      if (visiteError) {
        logger.error('Error getting visite logs from Supabase:', visiteError);
        throw visiteError;
      }

      if (!visites || visites.length === 0) {
        return [];
      }

      // Get unique user IDs
      const userIds = [...new Set(visites.map(v => v.id_user).filter(id => id))];

      // Fetch user details
      const { data: users, error: usersError } = await this.supabase
        .from('user')
        .select('id_user, "first name", name, email')
        .in('id_user', userIds);

      if (usersError) {
        logger.error('Error getting users for logs:', usersError);
      }

      // Create a map of users by ID
      const userMap = {};
      (users || []).forEach(user => {
        userMap[user.id_user] = user;
      });

      // Transform data to match expected format
      return visites.map(log => {
        const user = userMap[log.id_user];
        return {
          id: log.id,
          memberId: log.id_user,
          firstName: user?.['first name'] || 'Unknown',
          lastName: user?.name || 'User',
          status: 'granted',
          message: 'Access granted',
          confidence: 0,
          timestamp: log.visited_at
        };
      });
    } catch (error) {
      logger.error('Error in getAccessLogs:', error);
      return []; // Return empty array instead of throwing
    }
  }

  async getAccessLogsByMemberId(memberId, limit = 50) {
    const { data, error } = await this.supabase
      .from('visite')
      .select('*')
      .eq('id_user', memberId)
      .order('visited_at', { ascending: false })
      .limit(limit);

    if (error) {
      logger.error('Error getting access logs by member ID from Supabase:', error);
      throw error;
    }

    return (data || []).map(log => ({
      id: log.id,
      memberId: log.id_user,
      status: 'granted',
      message: 'Access granted',
      timestamp: log.visited_at
    }));
  }

  // Dashboard stats
  async getDashboardStats() {
    // Get total members
    const { count: totalMembers } = await this.supabase
      .from('user')
      .select('*', { count: 'exact', head: true });

    // Get today's access logs
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const { data: todayLogs } = await this.supabase
      .from('visite')
      .select('*')
      .gte('visited_at', today.toISOString());

    const todayAccess = todayLogs?.length || 0;
    const grantedToday = todayAccess; // All visite records are granted access
    const deniedToday = 0;

    // Get recent access
    const { data: recentLogs } = await this.supabase
      .from('visite')
      .select(`
        *,
        user!inner (
          first name,
          name
        )
      `)
      .order('visited_at', { ascending: false })
      .limit(10);

    return {
      totalMembers: totalMembers || 0,
      activeMembers: totalMembers || 0,
      todayAccess,
      grantedToday,
      deniedToday,
      recentAccess: (recentLogs || []).map(log => ({
        memberId: log.id_user,
        firstName: log.user?.['first name'],
        lastName: log.user?.name,
        status: 'granted',
        timestamp: log.visited_at
      }))
    };
  }

  // Log access attempt (wrapper for createAccessLog)
  async logAccessAttempt(memberId, status, message) {
    if (status === 'granted' && memberId) {
      try {
        await this.createAccessLog({ memberId, status, message });
      } catch (error) {
        logger.error('Error logging access attempt:', error);
      }
    }
  }

  // Membership operations
  async createMembership(membershipData) {
    const { data, error } = await this.supabase
      .from('abbonnement')
      .insert([{
        id_user: membershipData.memberId,
        plan: membershipData.plan || 'standard',
        status: membershipData.status || 'active',
        start_date: new Date().toISOString(),
        end_date: membershipData.endDate ? membershipData.endDate.toISOString() : null,
        price: membershipData.price || 0
      }])
      .select()
      .single();

    if (error) {
      logger.error('Error creating membership in Supabase:', error);
      throw error;
    }

    return data.id;
  }

  async getMembershipStatus(memberId) {
    const { data, error } = await this.supabase
      .from('abbonnement')
      .select('*')
      .eq('id_user', memberId)
      .order('end_date', { ascending: false })
      .limit(1)
      .single();

    if (error && error.code !== 'PGRST116') {
      logger.error('Error getting membership status:', error);
      throw error;
    }

    if (!data) return { status: 'expired', daysLeft: 0 };

    const now = new Date();
    const endDate = new Date(data.end_date);
    const daysLeft = Math.ceil((endDate - now) / (1000 * 60 * 60 * 24));

    return {
      status: data.status === 'active' && daysLeft > 0 ? 'active' : 'expired',
      plan: data.plan,
      endDate: data.end_date,
      daysLeft: daysLeft > 0 ? daysLeft : 0,
      expired: daysLeft <= 0
    };
  }

  // Audit logs operations (optional - stored as comments in code, not in DB)
  async createAuditLog(logData) {
    // Audit logging - can be implemented later if needed
    logger.info(`Audit: ${logData.action} - ${logData.entityType} ${logData.entityId}`);
    return null;
  }
}

module.exports = SupabaseDatabase;
