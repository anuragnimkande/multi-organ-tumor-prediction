import { Link } from "react-router-dom";
import { useEffect, useRef } from "react";

// ──────────────────────────────────────────────
// Animated quantum particle background (Canvas)
// ──────────────────────────────────────────────
function QuantumCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let raf;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Particles
    const particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.5 + 0.5,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.1,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 212, 255, ${p.alpha})`;
        ctx.fill();
      });

      // Draw connecting lines
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(0, 212, 255, ${0.06 * (1 - dist / 100)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}

// ──────────────────────────────────────────────
// Pipeline step card
// ──────────────────────────────────────────────
function PipelineStep({ icon, title, desc, delay }) {
  return (
    <div
      className="glass-card p-5 flex gap-4 hover:shadow-quantum transition-all duration-300
                 hover:border-quantum-500/40 hover:-translate-y-0.5 animate-slide-up"
      style={{ animationDelay: delay, animationFillMode: "both", opacity: 0 }}
    >
      <div className="text-2xl flex-shrink-0 mt-0.5">{icon}</div>
      <div>
        <h3 className="font-display font-semibold text-slate-200 mb-1">
          {title}
        </h3>
        <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────
// Stat badge
// ──────────────────────────────────────────────
function StatBadge({ label, value }) {
  return (
    <div className="text-center px-6 py-4 glass-card">
      <div className="text-2xl font-bold font-display text-quantum-500">
        {value}
      </div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  );
}

// ──────────────────────────────────────────────
// HomePage
// ──────────────────────────────────────────────
export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* ── Hero section ── */}
      <section
        className="relative min-h-[90vh] flex flex-col items-center justify-center
                           px-4 text-center"
      >
        <QuantumCanvas />

        {/* Gradient orbs */}
        <div
          className="absolute top-20 left-1/4 w-96 h-96 rounded-full opacity-10
                        bg-quantum-purple blur-3xl pointer-events-none"
        />
        <div
          className="absolute bottom-20 right-1/4 w-96 h-96 rounded-full opacity-8
                        bg-quantum-500 blur-3xl pointer-events-none"
        />

        <div className="relative z-10 max-w-4xl mx-auto">
          {/* Badge */}
          <div
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full
                          border border-quantum-500/30 bg-quantum-500/5
                          text-quantum-500 text-sm font-medium mb-8 animate-fade-in"
          >
            <span className="w-2 h-2 rounded-full bg-quantum-500 animate-pulse" />
            Hybrid Quantum-Classical AI System
          </div>

          {/* Main heading */}
          <h1
            className="font-display font-bold text-5xl md:text-7xl text-white mb-6
                         leading-tight animate-slide-up"
          >
            Brain Tumor
            <span
              className="block bg-clip-text text-transparent
                             bg-gradient-to-r from-quantum-purple to-quantum-500"
            >
              Detection AI
            </span>
          </h1>

          <p
            className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed
                        animate-slide-up"
            style={{ animationDelay: "0.1s" }}
          >
            Upload an MRI brain scan and our{" "}
            <span className="text-quantum-500 font-medium">ResNet18 CNN</span> +{" "}
            <span className="text-quantum-purple font-medium">
              4-qubit PennyLane circuit
            </span>{" "}
            hybrid model will analyze it for tumor indicators in seconds.
          </p>

          {/* CTA button */}
          <div
            className="flex flex-wrap justify-center gap-4 animate-slide-up"
            style={{ animationDelay: "0.2s" }}
          >
            <Link
              to="/upload"
              id="hero-cta-primary"
              className="btn-quantum text-base py-4 px-8"
            >
              ⚛️ Analyze MRI Scan
            </Link>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce opacity-40">
          <div className="w-px h-12 bg-gradient-to-b from-quantum-500 to-transparent mx-auto" />
        </div>
      </section>

      {/* ── Stats row ── */}
      <section className="py-12 px-4">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatBadge value="4" label="Quantum Qubits" />
          <StatBadge value="128" label="CNN Feature Dims" />
          <StatBadge value="2" label="VQC Layers" />
          <StatBadge value="224²" label="Input Resolution" />
        </div>
      </section>
    </div>
  );
}
