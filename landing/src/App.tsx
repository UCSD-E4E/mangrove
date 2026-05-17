import { Routes, Route, useNavigate } from 'react-router-dom'
import { Nav } from './components/Nav'
import { Hero } from './components/Hero'
import { IntroHero } from './components/IntroHero'
import { Team } from './components/Team'
import { JoinCTA } from './components/JoinCTA'
import { Footer } from './components/Footer'
import { MangroveVideoSection } from './components/story/MangroveVideoSection'
import { CrisisSection } from './components/story/CrisisSection'
import { SolutionSection } from './components/story/SolutionSection'
import { SuperResSection } from './components/story/SuperResSection'
import { useScrollAnimations } from './hooks/useScrollAnimations'
import { RegionPage } from './pages/RegionPage'
import { BlogPage } from './pages/BlogPage'
import { TeamPage } from './pages/TeamPage'
import { VisualizerPage } from './pages/VisualizerPage'
import { CollaboratePage } from './pages/CollaboratePage'
import { BlogPostPage } from './pages/BlogPostPage'

function MainPage() {
  const navigate = useNavigate()
  useScrollAnimations()
  return (
    <>
      <Nav />
      <main>
        <IntroHero />
        <MangroveVideoSection videoSrc="/mangrove.mp4" />
        <CrisisSection />
        <SolutionSection />
        <Hero onRegionClick={(id) => navigate(`/region/${id}`)} />
        <SuperResSection />
        <Team />
        <JoinCTA />
      </main>
      <Footer />
    </>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route path="/region/:id" element={<RegionPage />} />
      <Route path="/visualizer" element={<VisualizerPage />} />
      <Route path="/blog" element={<BlogPage />} />
      <Route path="/blog/:slug" element={<BlogPostPage />} />
      <Route path="/team" element={<TeamPage />} />
      <Route path="/collaborate" element={<CollaboratePage />} />
    </Routes>
  )
}

export default App
