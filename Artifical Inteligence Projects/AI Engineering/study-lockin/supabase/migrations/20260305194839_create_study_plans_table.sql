/*
  # Create study_plans table

  1. New Tables
    - `study_plans`
      - `id` (uuid, primary key)
      - `subject` (text) - The subject to study
      - `duration` (text) - Study duration (e.g., "4 weeks")
      - `goal` (text) - Learning goal
      - `personalized_plan` (jsonb) - AI-generated personalized study plan
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on `study_plans` table
    - No policies needed (for demo purposes, publicly readable)
*/

CREATE TABLE IF NOT EXISTS study_plans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject text NOT NULL,
  duration text NOT NULL,
  goal text NOT NULL,
  personalized_plan jsonb,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE study_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Study plans are publicly readable"
  ON study_plans
  FOR SELECT
  USING (true);

CREATE POLICY "Anyone can insert study plans"
  ON study_plans
  FOR INSERT
  WITH CHECK (true);
