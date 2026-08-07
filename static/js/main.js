/* main.js — CineScope micro-animations and UX helpers */

document.addEventListener('DOMContentLoaded', () => {

  // ── Navbar scroll shadow ──────────────────────────────
  const navbar = document.getElementById('navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });
  }

  // ── Auto-dismiss flash messages after 5s ─────────────
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      el.style.opacity = '0';
      el.style.transform = 'translateX(120%)';
      setTimeout(() => el.remove(), 500);
    }, 5000);
  });

  // ── Submit button loading state ───────────────────────
  const reviewForm   = document.getElementById('reviews-form');
  const submitBtn    = document.getElementById('submit-reviews');

  if (reviewForm && submitBtn) {
    reviewForm.addEventListener('submit', () => {
      const text = submitBtn.querySelector('.btn-submit-text');
      const icon = submitBtn.querySelector('.btn-submit-icon');
      if (text) text.textContent = 'Analysing…';
      if (icon) icon.textContent = '⏳';
      submitBtn.disabled = true;
      submitBtn.style.opacity = '0.8';
    });
  }

  // ── Auth form submit loading state ───────────────────
  ['login-form', 'register-form'].forEach(id => {
    const form = document.getElementById(id);
    if (!form) return;
    form.addEventListener('submit', () => {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) {
        const t = btn.querySelector('.btn-text');
        if (t) t.textContent = 'Please wait…';
        btn.disabled = true;
        btn.style.opacity = '0.7';
      }
    });
  });

  // ── Animate progress bars on results page ────────────
  // Bars start at 0 in CSS and animate to their target width on load
  document.querySelectorAll('.progress-bar').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = target; }, 100);
  });

  // ── Stagger animation for movie cards & result cards ─
  const animatedCards = document.querySelectorAll('.movie-card, .result-card');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });
    animatedCards.forEach(card => {
      card.style.opacity = '0';
      observer.observe(card);
    });
  }

  // ── Movie card hover tilt effect ─────────────────────
  document.querySelectorAll('.movie-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect   = card.getBoundingClientRect();
      const x      = (e.clientX - rect.left) / rect.width  - 0.5;
      const y      = (e.clientY - rect.top)  / rect.height - 0.5;
      card.style.transform = `translateY(-6px) rotateX(${-y * 5}deg) rotateY(${x * 5}deg)`;
      card.style.transition = 'transform 0.1s ease';
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
      card.style.transition = 'transform 0.4s ease, box-shadow 0.3s ease, border-color 0.3s ease';
    });
  });

  // ── Input focus highlight ─────────────────────────────
  document.querySelectorAll('.form-input, .review-textarea').forEach(el => {
    el.addEventListener('focus', () => {
      el.closest('.form-group, .review-input-group')?.classList.add('focused');
    });
    el.addEventListener('blur', () => {
      el.closest('.form-group, .review-input-group')?.classList.remove('focused');
    });
  });

  console.log('%c🎬 CineScope loaded', 'color:#f5c518;font-size:14px;font-weight:bold;');
});
