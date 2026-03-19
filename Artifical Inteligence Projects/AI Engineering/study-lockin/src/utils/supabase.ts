import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function generatePersonalizedPlan(subject: string, duration: string, goal: string) {
  const apiUrl = `${supabaseUrl}/functions/v1/personalize-study-plan`;

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${supabaseAnonKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ subject, duration, goal }),
  });

  if (!response.ok) {
    throw new Error('Failed to generate personalized plan');
  }

  return await response.json();
}

export async function saveStudyPlan(
  subject: string,
  duration: string,
  goal: string,
  personalizedPlan: unknown
) {
  const { data, error } = await supabase
    .from('study_plans')
    .insert([
      {
        subject,
        duration,
        goal,
        personalized_plan: personalizedPlan,
      },
    ])
    .select();

  if (error) throw error;
  return data;
}

export async function getStudyPlans() {
  const { data, error } = await supabase
    .from('study_plans')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data;
}

export async function deleteStudyPlan(id: string) {
  const { error } = await supabase
    .from('study_plans')
    .delete()
    .eq('id', id);

  if (error) throw error;
}
