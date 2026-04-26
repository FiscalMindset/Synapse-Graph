// Lightweight guided tour / slideshow for static demo pages
(function () {
  const DEFAULT_AUTO_ADVANCE = false;
  const DEFAULT_ADVANCE_MS = 3500;

  function injectStyles() {
    if (document.getElementById('tour-styles')) return;
    const css = `
      .tour-highlight{position:relative;z-index:9999;box-shadow:0 8px 30px rgba(3,10,25,0.7), 0 0 0 6px rgba(139,211,255,0.06);transition:box-shadow .25s,transform .18s;border-radius:8px}
      .tour-tooltip{position:absolute;z-index:10000;background:rgba(3,10,25,0.96);color:#e6eef8;padding:10px;border-radius:8px;max-width:380px;box-shadow:0 8px 30px rgba(2,6,23,0.6);font-size:13px}
      .tour-controls{position:fixed;right:18px;bottom:18px;z-index:10001;display:flex;gap:8px}
      .tour-btn{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.04);color:#e6eef8;padding:8px 10px;border-radius:8px;cursor:pointer}
      .tour-close{position:fixed;right:16px;top:12px;z-index:10002;background:transparent;border:none;color:#cfe3ff;font-size:18px}
      .tour-step{font-weight:700;margin-bottom:6px}
    `;
    const s = document.createElement('style');
    s.id = 'tour-styles';
    s.appendChild(document.createTextNode(css));
    document.head.appendChild(s);
  }

  function buildUI() {
    const tooltip = document.createElement('div');
    tooltip.className = 'tour-tooltip';
    tooltip.style.display = 'none';
    document.body.appendChild(tooltip);

    const close = document.createElement('button');
    close.className = 'tour-close';
    close.innerText = '✕';
    close.title = 'Close tour';
    document.body.appendChild(close);

    return { tooltip, close };
  }

  function startTour() {
    injectStyles();
    const rawItems = Array.from(document.querySelectorAll('[data-tour]'));
    if (!rawItems.length) return;

    // Sort by explicit order when provided, otherwise preserve DOM order
    const items = rawItems.slice().sort((a, b) => {
      const ao = a.getAttribute('data-tour-order');
      const bo = b.getAttribute('data-tour-order');
      if (ao != null && bo != null) return (parseInt(ao, 10) || 0) - (parseInt(bo, 10) || 0);
      if (ao != null) return -1;
      if (bo != null) return 1;
      return 0;
    });

    const ui = buildUI();
    let idx = 0;
    const ADVANCE_MS = parseInt(document.body.getAttribute('data-tour-interval') || DEFAULT_ADVANCE_MS, 10);

    function positionTooltip(el) {
      ui.tooltip.style.display = 'block';
      const rect = el.getBoundingClientRect();
      ui.tooltip.style.left = '0px';
      ui.tooltip.style.top = '0px';
      requestAnimationFrame(() => {
        const tRect = ui.tooltip.getBoundingClientRect();
        const spaceAbove = rect.top;
        const preferAbove = spaceAbove > tRect.height + 24;
        const top = preferAbove ? window.scrollY + rect.top - tRect.height - 12 : window.scrollY + rect.bottom + 12;
        let left = window.scrollX + rect.left;
        if (left + tRect.width > window.innerWidth - 16) left = window.innerWidth - tRect.width - 16;
        if (left < 8) left = 8;
        ui.tooltip.style.left = left + 'px';
        ui.tooltip.style.top = Math.max(8, top) + 'px';
      });
    }

    function show(i) {
      items.forEach((el) => el.classList.remove('tour-highlight'));
      const el = items[i];
      if (!el) return;
      el.classList.add('tour-highlight');
      const text = el.getAttribute('data-tour') || '';
      ui.tooltip.innerHTML = `<div class="tour-step">Step ${i + 1}/${items.length}</div><div>${text}</div>`;
      positionTooltip(el);
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function close() {
      ui.tooltip.remove(); ui.close.remove(); items.forEach((el) => el.classList.remove('tour-highlight'));
      document.removeEventListener('keydown', keyHandler);
    }

    ui.close.addEventListener('click', close);

    function keyHandler(e) {
      if (e.key === 'Escape') close();
    }

    document.addEventListener('keydown', keyHandler);

    // start showing first step only; controls/navigation removed per user request
    show(0);
    window.__tour = { items };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startTour);
  } else startTour();
})();
