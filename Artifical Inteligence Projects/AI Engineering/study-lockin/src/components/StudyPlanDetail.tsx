import { ChevronLeft, BookOpen, Clock, Target, Lightbulb, Zap, Flag, ExternalLink, Calendar } from 'lucide-react';

interface WeeklyBreakdown {
  week: number;
  focus: string;
  tasks: string[];
  estimatedHours: number;
}

interface PersonalizedPlan {
  overview: string;
  weeklyBreakdown: WeeklyBreakdown[];
  resources: string[];
  studyTips: string[];
  milestones: string[];
}

interface StudyPlanDetailProps {
  subject: string;
  duration: string;
  goal: string;
  personalizedPlan: PersonalizedPlan;
  onBack: () => void;
}

const resourceLinks: Record<string, string> = {
  'Online courses on': 'https://www.coursera.org',
  'YouTube tutorials': 'https://www.youtube.com',
  'Practice problem sets': 'https://www.khanacademy.org',
  'Study groups': 'https://www.discord.com',
  'Official textbooks': 'https://www.wikipedia.org',
  'Video lectures': 'https://ocw.mit.edu',
};

function getResourceLink(resource: string): string {
  for (const [key, link] of Object.entries(resourceLinks)) {
    if (resource.toLowerCase().includes(key.toLowerCase())) {
      return link;
    }
  }
  return 'https://www.google.com/search?q=' + encodeURIComponent(resource);
}

export function StudyPlanDetail({
  subject,
  duration,
  goal,
  personalizedPlan,
  onBack,
}: StudyPlanDetailProps) {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm overflow-y-auto z-50">
      <div className="min-h-screen bg-white dark:bg-zinc-950 p-4 md:p-8">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white mb-8 transition-colors font-medium"
          >
            <ChevronLeft className="w-5 h-5" />
            Back to Plans
          </button>

          <div className="bg-zinc-50 dark:bg-zinc-900 rounded-2xl shadow-lg p-8 mb-8 border border-zinc-200 dark:border-zinc-800">
            <h1 className="text-4xl font-bold text-zinc-950 dark:text-white mb-2">
              {subject}
            </h1>
            <div className="flex flex-wrap gap-4 mb-6">
              <div className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400">
                <Clock className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
                <span>{duration}</span>
              </div>
              <div className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400">
                <Target className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
                <span>{goal}</span>
              </div>
            </div>
            <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">
              {personalizedPlan.overview}
            </p>
          </div>

          <div className="bg-zinc-50 dark:bg-zinc-900 rounded-2xl shadow-lg p-8 mb-8 border border-zinc-200 dark:border-zinc-800">
            <h2 className="text-2xl font-bold text-zinc-950 dark:text-white mb-6 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
              Weekly Breakdown
            </h2>
            <div className="space-y-4">
              {personalizedPlan.weeklyBreakdown.map((week) => (
                <div
                  key={week.week}
                  className="border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 hover:shadow-md transition-all bg-white dark:bg-zinc-800"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="text-lg font-semibold text-zinc-950 dark:text-white">
                        Week {week.week}: {week.focus}
                      </h3>
                      <p className="text-sm text-zinc-500 dark:text-zinc-400">
                        {week.estimatedHours} hours per week
                      </p>
                    </div>
                  </div>
                  <ul className="space-y-2">
                    {week.tasks.map((task, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2 text-zinc-600 dark:text-zinc-400"
                      >
                        <span className="text-zinc-700 dark:text-zinc-300 font-semibold mt-0.5">
                          •
                        </span>
                        <span>{task}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <div className="bg-zinc-50 dark:bg-zinc-900 rounded-2xl shadow-lg p-8 border border-zinc-200 dark:border-zinc-800">
              <h2 className="text-xl font-bold text-zinc-950 dark:text-white mb-6 flex items-center gap-2">
                <Zap className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
                Study Tips
              </h2>
              <ul className="space-y-3">
                {personalizedPlan.studyTips.map((tip, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-3 text-zinc-600 dark:text-zinc-400"
                  >
                    <span className="text-zinc-700 dark:text-zinc-300 font-bold">
                      ✓
                    </span>
                    <span className="text-sm">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-zinc-50 dark:bg-zinc-900 rounded-2xl shadow-lg p-8 border border-zinc-200 dark:border-zinc-800">
              <h2 className="text-xl font-bold text-zinc-950 dark:text-white mb-6 flex items-center gap-2">
                <Flag className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
                Milestones
              </h2>
              <ul className="space-y-3">
                {personalizedPlan.milestones.map((milestone, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-3 text-zinc-600 dark:text-zinc-400"
                  >
                    <span className="text-zinc-700 dark:text-zinc-300 font-bold">
                      ★
                    </span>
                    <span className="text-sm">{milestone}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="bg-zinc-50 dark:bg-zinc-900 rounded-2xl shadow-lg p-8 border border-zinc-200 dark:border-zinc-800">
            <h2 className="text-xl font-bold text-zinc-950 dark:text-white mb-6 flex items-center gap-2">
              <Lightbulb className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
              Recommended Resources
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              {personalizedPlan.resources.map((resource, idx) => {
                const resourceLink = getResourceLink(resource);
                return (
                  <a
                    key={idx}
                    href={resourceLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group bg-white dark:bg-zinc-800 p-4 rounded-lg border border-zinc-200 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600 hover:shadow-md transition-all"
                  >
                    <div className="flex items-start gap-3 justify-between">
                      <div className="flex items-start gap-3">
                        <BookOpen className="w-5 h-5 text-zinc-700 dark:text-zinc-300 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-zinc-700 dark:text-zinc-300 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors">
                          {resource}
                        </span>
                      </div>
                      <ExternalLink className="w-4 h-4 text-zinc-400 dark:text-zinc-600 group-hover:text-zinc-700 dark:group-hover:text-zinc-300 flex-shrink-0 mt-0.5 transition-colors" />
                    </div>
                  </a>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
