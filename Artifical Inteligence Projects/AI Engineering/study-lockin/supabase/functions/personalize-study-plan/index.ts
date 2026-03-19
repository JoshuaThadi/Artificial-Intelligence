import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

interface StudyPlanRequest {
  subject: string;
  duration: string;
  goal: string;
}

interface PersonalizedPlan {
  overview: string;
  weeklyBreakdown: {
    week: number;
    focus: string;
    tasks: string[];
    estimatedHours: number;
  }[];
  resources: string[];
  studyTips: string[];
  milestones: string[];
}

function generatePersonalizedPlan(data: StudyPlanRequest): PersonalizedPlan {
  const { subject, duration, goal } = data;

  const durationValue = parseInt(duration);
  const weeks = durationValue > 1 ? durationValue : 4;

  const weeklyBreakdown = [];
  const baseHours = Math.ceil(30 / weeks);

  for (let week = 1; week <= Math.min(weeks, 6); week++) {
    const tasks: string[] = [];

    if (week === 1) {
      tasks.push(`Fundamentals: Review core concepts of ${subject}`);
      tasks.push("Create study notes and concept map");
      tasks.push("Identify knowledge gaps");
    } else if (week <= Math.ceil(weeks / 2)) {
      tasks.push(`Dive deeper: Explore advanced topics in ${subject}`);
      tasks.push("Solve practice problems");
      tasks.push("Join study groups or forums");
    } else {
      tasks.push(`Mastery phase: Complex applications of ${subject}`);
      tasks.push("Complete full-length practice tests");
      tasks.push("Review and reinforce weak areas");
    }

    weeklyBreakdown.push({
      week,
      focus: week === 1 ? "Foundation" : week <= Math.ceil(weeks / 2) ? "Depth" : "Mastery",
      tasks,
      estimatedHours: baseHours,
    });
  }

  const resources = [
    `Online courses on ${subject}`,
    "YouTube tutorials and explanations",
    "Practice problem sets and quizzes",
    "Study groups and peer discussion",
    "Official textbooks and reference materials",
  ];

  const studyTips = [
    "Use the Pomodoro Technique: 25 minutes focus, 5 minutes break",
    "Review notes within 24 hours to enhance memory retention",
    "Teach concepts to others to deepen understanding",
    "Mix passive reading with active problem-solving",
    "Track progress with a visual study calendar",
    "Take mock tests to assess readiness",
  ];

  const milestones = [
    "Understand fundamental concepts (Week 1-2)",
    "Complete 50% of practice problems (Week 2-3)",
    "Take first full practice test (Week 3-4)",
    "Achieve 80%+ accuracy on practice tests (Week 4-5)",
    "Final review and refinement (Week 5-6)",
  ];

  return {
    overview: `A personalized ${weeks}-week study plan for ${subject}. This plan focuses on: ${goal}. You'll progress through foundation, depth, and mastery phases with structured weekly goals and resources.`,
    weeklyBreakdown: weeklyBreakdown.slice(0, weeks),
    resources,
    studyTips,
    milestones,
  };
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    if (req.method !== "POST") {
      return new Response(
        JSON.stringify({ error: "Method not allowed" }),
        {
          status: 405,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const requestData: StudyPlanRequest = await req.json();

    if (!requestData.subject || !requestData.duration || !requestData.goal) {
      return new Response(
        JSON.stringify({ error: "Missing required fields" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const personalizedPlan = generatePersonalizedPlan(requestData);

    return new Response(
      JSON.stringify({
        success: true,
        personalizedPlan,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error:", error);
    return new Response(
      JSON.stringify({
        error: "Internal server error",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
