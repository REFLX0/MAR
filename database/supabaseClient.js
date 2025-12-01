// Supabase database operations for admin dashboard
import { supabase } from '../config/supabase.js';

// ==================== USER MANAGEMENT ====================

/**
 * Get all active members with their subscription details
 */
export async function getActiveMembers() {
    const { data, error } = await supabase
        .from('active_members') // Using the view we created
        .select('*')
        .order('days_remaining', { ascending: true });

    if (error) {
        console.error('Error fetching active members:', error);
        return [];
    }

    return data;
}

/**
 * Get user with their subscription info
 */
export async function getUserWithSubscription(userId) {
    const { data, error } = await supabase
        .from('user_profile')
        .select(`
      *,
      abonnement (*)
    `)
        .eq('auth_id', userId)
        .single();

    if (error) {
        console.error('Error fetching user:', error);
        return null;
    }

    return data;
}

/**
 * Search members by name, email, or phone
 */
export async function searchMembers(query) {
    const { data, error } = await supabase
        .from('user_profile')
        .select(`
      *,
      abonnement!abonnement_auth_id_fkey (*)
    `)
        .or(`name.ilike.%${query}%,first_name.ilike.%${query}%,email.ilike.%${query}%,phone.ilike.%${query}%`);

    if (error) {
        console.error('Error searching members:', error);
        return [];
    }

    return data;
}

// ==================== REGISTRATION ====================

/**
 * Register a new member with subscription
 */
export async function registerMember(memberData) {
    try {
        // 1. Create auth user
        const { data: authData, error: authError } = await supabase.auth.signUp({
            email: memberData.email,
            password: memberData.password || Math.random().toString(36).slice(-8) // Random password if not provided
        });

        if (authError) throw authError;

        const userId = authData.user.id;

        // 2. Create user profile
        const { error: profileError } = await supabase
            .from('user_profile')
            .insert({
                auth_id: userId,
                email: memberData.email,
                name: memberData.name,
                first_name: memberData.first_name,
                phone: memberData.phone,
                user_photo_url: memberData.user_photo_url
            });

        if (profileError) throw profileError;

        // 3. Create subscription (abonnement)
        const endDate = new Date();
        endDate.setDate(endDate.getDate() + (memberData.duration_days || 30));

        const { data: abonnement, error: abonnementError } = await supabase
            .from('abonnement')
            .insert({
                auth_id: userId,
                plan: memberData.plan,
                status: 'active',
                start_date: new Date().toISOString().split('T')[0],
                end_date: endDate.toISOString().split('T')[0],
                price: memberData.price
            })
            .select()
            .single();

        if (abonnementError) throw abonnementError;

        // 4. Record payment if payment was made
        if (memberData.payment_method && memberData.price > 0) {
            const adminId = (await supabase.auth.getUser()).data.user.id;

            await supabase
                .from('payment')
                .insert({
                    abonnement_id: abonnement.id,
                    user_id: userId,
                    amount: memberData.price,
                    payment_method: memberData.payment_method,
                    status: 'completed',
                    processed_by: adminId
                });
        }

        return { success: true, userId, abonnementId: abonnement.id };
    } catch (error) {
        console.error('Error registering member:', error);
        return { success: false, error: error.message };
    }
}

// ==================== VISITS (CHECK-INS) ====================

/**
 * Get today's visits
 */
export async function getTodayVisits() {
    const { data, error } = await supabase
        .from('today_visits') // Using the view
        .select('*');

    if (error) {
        console.error('Error fetching today visits:', error);
        return [];
    }

    return data;
}

/**
 * Record a visit (check-in)
 */
export async function recordVisit(userId) {
    const { data, error } = await supabase
        .from('visite')
        .insert({
            auth_id: userId,
            visited_at: new Date().toISOString()
        })
        .select()
        .single();

    if (error) {
        console.error('Error recording visit:', error);
        return { success: false, error: error.message };
    }

    return { success: true, visit: data };
}

/**
 * Get visit history for a user
 */
export async function getUserVisits(userId, limit = 50) {
    const { data, error } = await supabase
        .from('visite')
        .select('*')
        .eq('auth_id', userId)
        .order('visited_at', { ascending: false })
        .limit(limit);

    if (error) {
        console.error('Error fetching user visits:', error);
        return [];
    }

    return data;
}

// ==================== SUBSCRIPTIONS ====================

/**
 * Get members with expiring memberships
 */
export async function getExpiringMemberships(days = 7) {
    const { data, error } = await supabase
        .from('expiring_memberships')
        .select('*');

    if (error) {
        console.error('Error fetching expiring memberships:', error);
        return [];
    }

    return data;
}

/**
 * Renew/extend a subscription
 */
export async function renewSubscription(abonnementId, durationDays, price, paymentMethod = 'cash') {
    try {
        // Get current subscription
        const { data: currentSub } = await supabase
            .from('abonnement')
            .select('*')
            .eq('id', abonnementId)
            .single();

        if (!currentSub) {
            throw new Error('Subscription not found');
        }

        // Calculate new end date
        const baseDate = new Date(currentSub.end_date) > new Date()
            ? new Date(currentSub.end_date)
            : new Date();
        const newEndDate = new Date(baseDate);
        newEndDate.setDate(newEndDate.getDate() + durationDays);

        // Update subscription
        const { error: updateError } = await supabase
            .from('abonnement')
            .update({
                end_date: newEndDate.toISOString().split('T')[0],
                status: 'active',
                updated_at: new Date().toISOString()
            })
            .eq('id', abonnementId);

        if (updateError) throw updateError;

        // Record payment
        const adminId = (await supabase.auth.getUser()).data.user.id;

        await supabase
            .from('payment')
            .insert({
                abonnement_id: abonnementId,
                user_id: currentSub.auth_id,
                amount: price,
                payment_method: paymentMethod,
                status: 'completed',
                processed_by: adminId,
                notes: `Renewal for ${durationDays} days`
            });

        return { success: true, newEndDate };
    } catch (error) {
        console.error('Error renewing subscription:', error);
        return { success: false, error: error.message };
    }
}

// ==================== PAYMENTS ====================

/**
 * Get payment history
 */
export async function getPayments(limit = 50) {
    const { data, error } = await supabase
        .from('payment')
        .select(`
      *,
      user_profile!payment_user_id_fkey (name, first_name, email),
      admin_profile!payment_processed_by_fkey (name)
    `)
        .order('created_at', { ascending: false })
        .limit(limit);

    if (error) {
        console.error('Error fetching payments:', error);
        return [];
    }

    return data;
}

/**
 * Get revenue summary
 */
export async function getRevenueSummary(days = 30) {
    const { data, error } = await supabase
        .from('revenue_summary')
        .select('*')
        .limit(days);

    if (error) {
        console.error('Error fetching revenue summary:', error);
        return [];
    }

    return data;
}

// ==================== FACE RECOGNITION ====================

/**
 * Save face embedding for a user
 */
export async function saveFaceEmbedding(userId, embeddingData, imageUrl, angle) {
    const { data, error } = await supabase
        .from('face_embedding')
        .insert({
            user_id: userId,
            embedding_data: JSON.stringify(embeddingData),
            image_url: imageUrl,
            angle: angle // 'center', 'left', 'right', 'up', 'down'
        })
        .select()
        .single();

    if (error) {
        console.error('Error saving face embedding:', error);
        return { success: false, error: error.message };
    }

    return { success: true, embedding: data };
}

/**
 * Get all face embeddings
 */
export async function getAllFaceEmbeddings() {
    const { data, error } = await supabase
        .from('face_embedding')
        .select(`
      *,
      user_profile!face_embedding_user_id_fkey (
        name,
        first_name,
        email,
        user_photo_url
      )
    `)
        .eq('is_active', true);

    if (error) {
        console.error('Error fetching embeddings:', error);
        return [];
    }

    return data;
}

// ==================== DASHBOARD STATS ====================

/**
 * Get dashboard statistics
 */
export async function getDashboardStats() {
    const today = new Date().toISOString().split('T')[0];

    // Today's visits count
    const { count: todayVisits } = await supabase
        .from('visite')
        .select('*', { count: 'exact', head: true })
        .gte('visited_at', today);

    // Active memberships
    const { count: activeMembers } = await supabase
        .from('abonnement')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'active')
        .gte('end_date', today);

    // Expiring soon (next 7 days)
    const sevenDaysFromNow = new Date();
    sevenDaysFromNow.setDate(sevenDaysFromNow.getDate() + 7);

    const { count: expiringSoon } = await supabase
        .from('abonnement')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'active')
        .gte('end_date', today)
        .lte('end_date', sevenDaysFromNow.toISOString().split('T')[0]);

    // Today's revenue
    const { data: todayPayments } = await supabase
        .from('payment')
        .select('amount')
        .eq('status', 'completed')
        .gte('created_at', today);

    const todayRevenue = todayPayments?.reduce((sum, p) => sum + (p.amount || 0), 0) || 0;

    return {
        todayVisits: todayVisits || 0,
        activeMembers: activeMembers || 0,
        expiringSoon: expiringSoon || 0,
        todayRevenue
    };
}

// ==================== REALTIME SUBSCRIPTIONS ====================

/**
 * Subscribe to new visits in real-time
 */
export function subscribeToVisits(callback) {
    const channel = supabase
        .channel('visits-channel')
        .on(
            'postgres_changes',
            {
                event: 'INSERT',
                schema: 'public',
                table: 'visite'
            },
            async (payload) => {
                console.log('New visit detected!', payload);

                // Get user details
                const { data: user } = await supabase
                    .from('user_profile')
                    .select('*')
                    .eq('auth_id', payload.new.auth_id)
                    .single();

                // Call the callback with visit + user data
                callback({
                    visit: payload.new,
                    user: user
                });
            }
        )
        .subscribe();

    return channel;
}

/**
 * Subscribe to payment changes
 */
export function subscribeToPayments(callback) {
    const channel = supabase
        .channel('payments-channel')
        .on(
            'postgres_changes',
            {
                event: '*', // All events
                schema: 'public',
                table: 'payment'
            },
            (payload) => {
                console.log('Payment change:', payload);
                callback(payload);
            }
        )
        .subscribe();

    return channel;
}

/**
 * Subscribe to subscription changes
 */
export function subscribeToAbonnements(callback) {
    const channel = supabase
        .channel('abonnement-channel')
        .on(
            'postgres_changes',
            {
                event: 'UPDATE',
                schema: 'public',
                table: 'abonnement'
            },
            async (payload) => {
                const endDate = new Date(payload.new.end_date);
                const daysRemaining = Math.floor((endDate - new Date()) / (1000 * 60 * 60 * 24));

                // Alert if expiring soon
                if (daysRemaining <= 7 && daysRemaining > 0) {
                    const { data: user } = await supabase
                        .from('user_profile')
                        .select('*')
                        .eq('auth_id', payload.new.auth_id)
                        .single();

                    callback({
                        type: 'expiring',
                        daysRemaining,
                        user,
                        abonnement: payload.new
                    });
                }
            }
        )
        .subscribe();

    return channel;
}

/**
 * Unsubscribe from a channel
 */
export function unsubscribeChannel(channel) {
    supabase.removeChannel(channel);
}
