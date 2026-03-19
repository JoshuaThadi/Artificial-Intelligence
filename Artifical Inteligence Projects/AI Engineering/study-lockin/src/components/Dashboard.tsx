import { useEffect, useState } from 'react';
import { Brain, Moon, Sun, Plus, BookOpen, Clock, Target, Trash2, Loader, Home } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { generatePersonalizedPlan, saveStudyPlan, getStudyPlans, deleteStudyPlan } from '../utils/supabase';
import { StudyPlanDetail } from './StudyPlanDetail';

interface PersonalizedPlan {
  overview: string;
  weeklyBreakdown: Array<{
    week: number;
    focus: string;
    tasks: string[];
    estimatedHours: number;
  }>;
  resources: string[];
  studyTips: string[];
  milestones: string[];
}

interface StudyPlan {
  id: string;
  subject: string;
  duration: string;
  goal: string;
  personalized_plan?: PersonalizedPlan;
  created_at?: string;
}

interface DashboardProps {
  onHome?: () => void;
}

export function Dashboard({ onHome }: DashboardProps) {
  const { theme, toggleTheme } = useTheme();
  const [studyPlans, setStudyPlans] = useState<StudyPlan[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<StudyPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [formData, setFormData] = useState({
    subject: '',
    duration: '',
    goal: '',
  });

  useEffect(() => {
    loadPlans();
  }, []);

  const loadPlans = async () => {
    try {
      setLoading(true);
      const plans = await getStudyPlans();
      setStudyPlans(plans || []);
    } catch (error) {
      console.error('Failed to load plans:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);

    try {
      const response = await generatePersonalizedPlan(formData.subject, formData.duration, formData.goal);

      if (response.success && response.personalizedPlan) {
        const savedPlans = await saveStudyPlan(
          formData.subject,
          formData.duration,
          formData.goal,
          response.personalizedPlan
        );

        if (savedPlans && savedPlans.length > 0) {
          const newPlan = savedPlans[0];
          setStudyPlans([newPlan, ...studyPlans]);
          setFormData({ subject: '', duration: '', goal: '' });
          setShowForm(false);
        }
      }
    } catch (error) {
      console.error('Failed to create plan:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDeletePlan = async (id: string) => {
    try {
      await deleteStudyPlan(id);
      setStudyPlans(studyPlans.filter((plan) => plan.id !== id));
    } catch (error) {
      console.error('Failed to delete plan:', error);
    }
  };

  const handleViewPlan = (plan: StudyPlan) => {
    if (plan.personalized_plan) {
      setSelectedPlan(plan);
    }
  };

  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 transition-colors duration-300">
      <nav className="px-6 py-5 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-zinc-950 dark:bg-white rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white dark:text-zinc-950" />
            </div>
            <span className="text-2xl font-bold text-zinc-950 dark:text-white">StudyAI</span>
          </div>
          <div className="flex items-center gap-3">
            {onHome && (
              <button
                onClick={onHome}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 font-medium transition-colors"
              >
                <Home className="w-5 h-5" />
                Home
              </button>
            )}
            <button
              onClick={toggleTheme}
              className="p-2.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors duration-200"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
              ) : (
                <Sun className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
              )}
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-4xl font-bold text-zinc-950 dark:text-white mb-2">
              My Study Plans
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400">
              Create and manage your personalized study plans
            </p>
          </div>
          <button
            onClick={() => setShowForm(true)}
            className="flex items-center gap-2 px-6 py-3 bg-zinc-950 dark:bg-white text-white dark:text-zinc-950 font-semibold rounded-xl hover:bg-zinc-800 dark:hover:bg-zinc-100 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            <Plus className="w-5 h-5" />
            Create Plan
          </button>
        </div>

        {showForm && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl max-w-md w-full p-8 border border-zinc-200 dark:border-zinc-800">
              <h2 className="text-2xl font-bold text-zinc-950 dark:text-white mb-6">
                Create Study Plan
              </h2>
              <form onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                    Subject
                  </label>
                  <input
                    type="text"
                    value={formData.subject}
                    onChange={(e) =>
                      setFormData({ ...formData, subject: e.target.value })
                    }
                    className="w-full px-4 py-3 bg-zinc-50 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded-lg focus:ring-2 focus:ring-zinc-950 dark:focus:ring-white focus:border-transparent text-zinc-900 dark:text-white transition-all"
                    placeholder="e.g., Mathematics, Physics"
                    required
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                    Duration
                  </label>
                  <input
                    type="text"
                    value={formData.duration}
                    onChange={(e) =>
                      setFormData({ ...formData, duration: e.target.value })
                    }
                    className="w-full px-4 py-3 bg-zinc-50 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded-lg focus:ring-2 focus:ring-zinc-950 dark:focus:ring-white focus:border-transparent text-zinc-900 dark:text-white transition-all"
                    placeholder="e.g., 4 weeks, 2 months"
                    required
                  />
                </div>
                <div className="mb-6">
                  <label className="block text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                    Goal
                  </label>
                  <textarea
                    value={formData.goal}
                    onChange={(e) =>
                      setFormData({ ...formData, goal: e.target.value })
                    }
                    className="w-full px-4 py-3 bg-zinc-50 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded-lg focus:ring-2 focus:ring-zinc-950 dark:focus:ring-white focus:border-transparent text-zinc-900 dark:text-white transition-all resize-none"
                    placeholder="What do you want to achieve?"
                    rows={3}
                    required
                  />
                </div>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => setShowForm(false)}
                    disabled={isGenerating}
                    className="flex-1 px-4 py-3 bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 font-medium rounded-lg hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-all disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isGenerating}
                    className="flex-1 px-4 py-3 bg-zinc-950 dark:bg-white text-white dark:text-zinc-950 font-medium rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {isGenerating && <Loader className="w-4 h-4 animate-spin" />}
                    {isGenerating ? 'Generating...' : 'Create'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {selectedPlan && selectedPlan.personalized_plan && (
          <StudyPlanDetail
            subject={selectedPlan.subject}
            duration={selectedPlan.duration}
            goal={selectedPlan.goal}
            personalizedPlan={selectedPlan.personalized_plan}
            onBack={() => setSelectedPlan(null)}
          />
        )}

        {loading ? (
          <div className="text-center py-20">
            <Loader className="w-10 h-10 animate-spin text-zinc-950 dark:text-white mx-auto" />
          </div>
        ) : studyPlans.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-20 h-20 bg-zinc-200 dark:bg-zinc-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <BookOpen className="w-10 h-10 text-zinc-400 dark:text-zinc-600" />
            </div>
            <h3 className="text-xl font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
              No study plans yet
            </h3>
            <p className="text-zinc-500 dark:text-zinc-500">
              Create your first study plan to get started
            </p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {studyPlans.map((plan) => (
              <div
                key={plan.id}
                onClick={() => handleViewPlan(plan)}
                className="bg-zinc-50 dark:bg-zinc-900 p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 border border-zinc-200 dark:border-zinc-800 cursor-pointer"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="w-12 h-12 bg-zinc-200 dark:bg-zinc-800 rounded-lg flex items-center justify-center">
                    <BookOpen className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeletePlan(plan.id);
                    }}
                    className="p-2 text-zinc-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
                <h3 className="text-lg font-semibold text-zinc-950 dark:text-white mb-3">
                  {plan.subject}
                </h3>
                <div className="space-y-2 mb-4">
                  <div className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400">
                    <Clock className="w-4 h-4" />
                    <span className="text-sm">{plan.duration}</span>
                  </div>
                  <div className="flex items-start gap-2 text-zinc-600 dark:text-zinc-400">
                    <Target className="w-4 h-4 mt-0.5" />
                    <span className="text-sm line-clamp-2">{plan.goal}</span>
                  </div>
                </div>
                <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                  <p className="text-xs text-zinc-500 dark:text-zinc-500">
                    {plan.created_at ? new Date(plan.created_at).toLocaleDateString() : 'Just now'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
