import * as dotenv from 'dotenv';

dotenv.config();

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
export const SUPABASE_URL = process.env.SUPABASE_URL || '';
export const SUPABASE_KEY = process.env.SUPABASE_KEY || '';
