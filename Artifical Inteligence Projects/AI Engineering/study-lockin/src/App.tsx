import { useState } from 'react';
import { LandingPage } from './components/LandingPage';
import { Dashboard } from './components/Dashboard';

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'dashboard'>('landing');

  return (
    <>
      {currentPage === 'landing' ? (
        <LandingPage onGetStarted={() => setCurrentPage('dashboard')} />
      ) : (
        <Dashboard onHome={() => setCurrentPage('landing')} />
      )}
    </>
  );
}

export default App;
