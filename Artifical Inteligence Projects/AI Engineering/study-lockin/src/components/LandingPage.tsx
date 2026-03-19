import { Brain, Calendar, Target, TrendingUp, Moon, Sun, ArrowRight } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 transition-colors duration-300">
      <nav className="px-6 py-5 flex justify-between items-center max-w-7xl mx-auto border-b border-zinc-200 dark:border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-zinc-950 dark:bg-white rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white dark:text-zinc-950" />
          </div>
          <span className="text-2xl font-bold text-zinc-950 dark:text-white">StudyAI</span>
        </div>
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
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-24">
        <div className="text-center mb-20">
          <h1 className="text-7xl font-bold text-zinc-950 dark:text-white mb-6 leading-tight tracking-tight">
            Intelligent Study
            <br />
            <span className="text-zinc-500 dark:text-zinc-400">Planning Redefined</span>
          </h1>
          <p className="text-xl text-zinc-600 dark:text-zinc-400 mb-10 max-w-2xl mx-auto leading-relaxed">
            AI-powered personalized study plans that adapt to your learning style, schedule, and goals. Achieve mastery with intelligent planning.
          </p>
          <button
            onClick={onGetStarted}
            className="inline-flex items-center gap-2 px-8 py-4 bg-zinc-950 dark:bg-white text-white dark:text-zinc-950 text-lg font-semibold rounded-xl hover:bg-zinc-800 dark:hover:bg-zinc-100 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            Get Started
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mt-20">
          <div className="group p-8 rounded-2xl bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 hover:shadow-lg transition-all duration-300">
            <div className="w-12 h-12 bg-zinc-200 dark:bg-zinc-800 rounded-lg flex items-center justify-center mb-5 group-hover:bg-zinc-300 dark:group-hover:bg-zinc-700 transition-colors">
              <Target className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
            </div>
            <h3 className="text-xl font-semibold text-zinc-950 dark:text-white mb-3">
              Smart Goals
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Set achievable study goals with AI-powered insights, progress tracking, and personalized recommendations.
            </p>
          </div>

          <div className="group p-8 rounded-2xl bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 hover:shadow-lg transition-all duration-300">
            <div className="w-12 h-12 bg-zinc-200 dark:bg-zinc-800 rounded-lg flex items-center justify-center mb-5 group-hover:bg-zinc-300 dark:group-hover:bg-zinc-700 transition-colors">
              <Calendar className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
            </div>
            <h3 className="text-xl font-semibold text-zinc-950 dark:text-white mb-3">
              Smart Scheduling
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Automatically generate optimized study schedules that adapt to your lifestyle and changing needs.
            </p>
          </div>

          <div className="group p-8 rounded-2xl bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 hover:shadow-lg transition-all duration-300">
            <div className="w-12 h-12 bg-zinc-200 dark:bg-zinc-800 rounded-lg flex items-center justify-center mb-5 group-hover:bg-zinc-300 dark:group-hover:bg-zinc-700 transition-colors">
              <TrendingUp className="w-6 h-6 text-zinc-700 dark:text-zinc-300" />
            </div>
            <h3 className="text-xl font-semibold text-zinc-950 dark:text-white mb-3">
              Track Progress
            </h3>
            <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Monitor your learning journey with detailed analytics and performance metrics at every stage.
            </p>
          </div>
        </div>
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-12 text-center text-zinc-600 dark:text-zinc-400 border-t border-zinc-200 dark:border-zinc-800 mt-24">
        <p>© 2024 StudyAI. Empowering learners with intelligent planning.</p>
      </footer>
    </div>
  );
}
