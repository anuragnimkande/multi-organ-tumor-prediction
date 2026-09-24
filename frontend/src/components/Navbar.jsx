import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'

const NAV_LINKS = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/history',   label: 'History' },
  { to: '/models',    label: 'Models' },
  { to: '/about',     label: 'About' },
]

export default function Navbar() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  return (
    <nav
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-void-950/90 backdrop-blur-md border-b border-quantum-500/15 shadow-quantum'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <Link
            to="/dashboard"
            className="flex items-center gap-2 group"
            id="navbar-logo"
          >
            <div className="w-9 h-9 rounded-xl flex items-center justify-center text-lg
                            bg-gradient-to-br from-quantum-purple to-quantum-500
                            group-hover:shadow-glow transition-all duration-300">
              ⚛️
            </div>
            <span className="font-display font-bold text-lg text-white">
              Quantum<span className="text-quantum-500">Scan</span>
            </span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map(link => (
              <Link
                key={link.to}
                id={`nav-link-${link.label.toLowerCase()}`}
                to={link.to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${location.pathname === link.to
                    ? 'bg-quantum-500/15 text-quantum-500'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                  }`}
              >
                {link.label}
              </Link>
            ))}

            <Link
              to="/analysis"
              id="nav-cta-button"
              className="ml-3 btn-quantum text-sm py-2 px-5"
            >
              New Analysis →
            </Link>
          </div>

          {/* Mobile hamburger */}
          <button
            id="navbar-mobile-toggle"
            className="md:hidden p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5"
            onClick={() => setMenuOpen(v => !v)}
            aria-label="Toggle menu"
          >
            <div className={`w-5 h-0.5 bg-current transition-all ${menuOpen ? 'rotate-45 translate-y-1.5' : ''}`} />
            <div className={`w-5 h-0.5 bg-current my-1 transition-all ${menuOpen ? 'opacity-0' : ''}`} />
            <div className={`w-5 h-0.5 bg-current transition-all ${menuOpen ? '-rotate-45 -translate-y-1.5' : ''}`} />
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden border-t border-quantum-500/10 bg-void-950/95 backdrop-blur-md px-4 py-3 space-y-1 animate-fade-in">
          {NAV_LINKS.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-all
                ${location.pathname === link.to
                  ? 'bg-quantum-500/15 text-quantum-500'
                  : 'text-slate-400 hover:text-white'
                }`}
            >
              {link.label}
            </Link>
          ))}
          <Link to="/analysis" className="btn-quantum block text-center mt-2 text-sm py-2.5">
            New Analysis →
          </Link>
        </div>
      )}
    </nav>
  )
}
