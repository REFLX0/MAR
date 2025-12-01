import { createClient } from '@supabase/supabase-js';

// Get these from your Supabase dashboard -> Settings -> API
const supabaseUrl = process.env.SUPABASE_URL || 'YOUR_SUPABASE_PROJECT_URL'; // e.g., https://xxxxx.supabase.co
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY || 'YOUR_SUPABASE_ANON_KEY';

// Create and export the Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
    },
    db: {
        schema: 'public'
    },
    realtime: {
        params: {
            eventsPerSecond: 10
        }
    }
});

// Helper function to check if user is authenticated
export async function getCurrentUser() {
    const { data: { user }, error } = await supabase.auth.getUser();
    if (error) {
        console.error('Error getting user:', error);
        return null;
    }
    return user;
}

// Helper function to check if current user is admin
export async function isAdmin() {
    const user = await getCurrentUser();
    if (!user) return false;

    const { data, error } = await supabase
        .from('admin_profile')
        .select('*')
        .eq('auth_id', user.id)
        .eq('is_active', true)
        .single();

    if (error) {
        console.error('Error checking admin status:', error);
        return false;
    }

    return !!data;
}
