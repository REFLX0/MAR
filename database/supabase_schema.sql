-- Gym Security System - Supabase Database Schema
-- Creating tables from scratch

-- Create users table (gym members)
CREATE TABLE IF NOT EXISTS users (
  id_user SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  "first name" TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  phone TEXT,
  user_photo_url TEXT,
  face_descriptor TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create visite table (access logs)
CREATE TABLE IF NOT EXISTS visite (
  id SERIAL PRIMARY KEY,
  id_user INTEGER REFERENCES users(id_user) ON DELETE CASCADE,
  visited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create abbonnement table (memberships)
CREATE TABLE IF NOT EXISTS abbonnement (
  id SERIAL PRIMARY KEY,
  id_user INTEGER REFERENCES users(id_user) ON DELETE CASCADE,
  plan TEXT DEFAULT 'standard',
  status TEXT DEFAULT 'active',
  start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  end_date TIMESTAMP WITH TIME ZONE,
  price DECIMAL(10,2),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_visite_id_user ON visite(id_user);
CREATE INDEX IF NOT EXISTS idx_visite_visited_at ON visite(visited_at);
CREATE INDEX IF NOT EXISTS idx_abbonnement_id_user ON abbonnement(id_user);

-- Enable Row Level Security (RLS) if not already enabled
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE visite ENABLE ROW LEVEL SECURITY;
ALTER TABLE abbonnement ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (to avoid conflicts)
DROP POLICY IF EXISTS "Allow anonymous read access to users" ON users;
DROP POLICY IF EXISTS "Allow anonymous insert to users" ON users;
DROP POLICY IF EXISTS "Allow anonymous update to users" ON users;
DROP POLICY IF EXISTS "Allow anonymous delete from users" ON users;
DROP POLICY IF EXISTS "Allow anonymous read access to visite" ON visite;
DROP POLICY IF EXISTS "Allow anonymous insert to visite" ON visite;
DROP POLICY IF EXISTS "Allow anonymous read access to abbonnement" ON abbonnement;

-- Create policies for anon access (adjust as needed for your security requirements)
-- Allow anonymous users to read users (for face recognition)
CREATE POLICY "Allow anonymous read access to users" ON users
  FOR SELECT TO anon USING (true);

-- Allow anonymous users to insert users (for registration)
CREATE POLICY "Allow anonymous insert to users" ON users
  FOR INSERT TO anon WITH CHECK (true);

-- Allow anonymous users to update users
CREATE POLICY "Allow anonymous update to users" ON users
  FOR UPDATE TO anon USING (true);

-- Allow anonymous users to delete users
CREATE POLICY "Allow anonymous delete from users" ON users
  FOR DELETE TO anon USING (true);

-- Allow anonymous access to visite (access logs)
CREATE POLICY "Allow anonymous read access to visite" ON visite
  FOR SELECT TO anon USING (true);

CREATE POLICY "Allow anonymous insert to visite" ON visite
  FOR INSERT TO anon WITH CHECK (true);

-- Allow anonymous access to abbonnement (memberships)
CREATE POLICY "Allow anonymous read access to abbonnement" ON abbonnement
  FOR SELECT TO anon USING (true);

-- Create a function to update the updated_at timestamp (if needed)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for users table (if updated_at column exists)
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Create helper function to insert users (bypasses schema cache issues)
CREATE OR REPLACE FUNCTION create_user_member(
  p_name TEXT,
  p_first_name TEXT,
  p_email TEXT,
  p_phone TEXT DEFAULT NULL,
  p_photo_url TEXT DEFAULT NULL,
  p_face_descriptor TEXT DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
  new_id INTEGER;
BEGIN
  INSERT INTO users (name, "first name", email, phone, user_photo_url, face_descriptor, created_at)
  VALUES (p_name, p_first_name, p_email, p_phone, p_photo_url, p_face_descriptor, NOW())
  RETURNING id_user INTO new_id;
  
  RETURN new_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to anon
GRANT EXECUTE ON FUNCTION create_user_member TO anon;

-- Table Mapping for Gym Security System:
-- users table = members (gym members)
-- visite table = access_logs (entry/exit logs)
-- abbonnement table = membership subscriptions
