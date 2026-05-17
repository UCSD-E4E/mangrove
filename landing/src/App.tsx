import { Routes, Route, useNavigate } from 'react-router-dom'
import { Nav } from './components/Nav'
import { Hero } from './components/Hero'
import { About } from './components/About'
import { SplitCompare } from './components/SplitCompare'
import { Pipeline } from './components/Pipeline'
import { Team } from './components/Team'
import { JoinCTA } from './components/JoinCTA'
import { Footer } from './components/Footer'
import { useScrollAnimations } from './hooks/useScrollAnimations'
import { RegionPage } from './pages/RegionPage'

function MainPage() {
  const navigate = useNavigate()
  useScrollAnimations()
  return (
    <>
      <Nav />
      <main>
        <Hero onRegionClick={(id) => navigate(`/region/${id}`)} />
        <About />
        <SplitCompare />
        <Pipeline />
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
    </Routes>
  )
}

export default App
