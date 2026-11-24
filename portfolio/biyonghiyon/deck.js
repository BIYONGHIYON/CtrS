// 발표 화면을 스크롤로 넘깁니다: 장마다 클릭 단계(.fragment)를 스크롤 위치에 맞춰 켭니다.
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const scenes = [...document.querySelectorAll(".pg-scene")];
  const bar = document.querySelector(".pg-progress");

  // x-connector → SVG 화살표
  document.querySelectorAll("x-connector").forEach((c) => {
    const n = (k) => parseFloat(c.getAttribute(k) || "0");
    const [x1, y1, x2, y2] = [n("x1"), n("y1"), n("x2"), n("y2")];
    const cs = getComputedStyle(c);
    const w = parseFloat(cs.borderTopWidth) || 3;
    const col = cs.color;
    c.style.border = "0";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "1920"); svg.setAttribute("height", "1080");
    const ang = Math.atan2(y2 - y1, x2 - x1), hl = 6 + w * 3.2, hw = 4 + w * 2;
    const head = c.getAttribute("head") === "end";
    const ex = head ? x2 - Math.cos(ang) * hl * 0.8 : x2, ey = head ? y2 - Math.sin(ang) * hl * 0.8 : y2;
    const line = document.createElementNS(NS, "line");
    Object.entries({ x1, y1, x2: ex, y2: ey, stroke: col, "stroke-width": w, "stroke-linecap": "round" }).forEach(([k, v]) => line.setAttribute(k, v));
    svg.appendChild(line);
    if (head) {
      const p = (d, s) => [x2 - Math.cos(ang) * d - Math.sin(ang) * s, y2 - Math.sin(ang) * d + Math.cos(ang) * s].join(",");
      const tri = document.createElementNS(NS, "polygon");
      tri.setAttribute("points", [x2 + "," + y2, p(hl, hw), p(hl, -hw)].join(" "));
      tri.setAttribute("fill", col);
      svg.appendChild(tri);
    }
    c.appendChild(svg);
  });

  // 장마다 단계 묶음 (reveal.js와 같은 순서: 번호 있는 것 → 번호 없는 것 문서 순)
  scenes.forEach((s) => {
    const ordered = [], unordered = [];
    s.querySelectorAll(".slide .fragment").forEach((f) => {
      if (f.hasAttribute("data-fragment-index")) (ordered[+f.dataset.fragmentIndex] ||= []).push(f);
      else unordered.push([f]);
    });
    s._groups = ordered.filter(Boolean).concat(unordered);
    s._slide = s.querySelector(".slide");
    s._frame = s.querySelector(".pg-frame");
    s._says = [...s.querySelectorAll(".pg-say")];
    s._dots = [...s.querySelectorAll(".pg-dot")];
    s._step = -1;
  });

  const fit = () => scenes.forEach((s) => s._frame.style.setProperty("--s", s._frame.clientWidth / 1920));

  const setStep = (s, step) => {
    if (s._step === step) return;
    s._step = step;
    s._groups.forEach((g, i) => g.forEach((f) => {
      f.classList.toggle("visible", i < step);
      f.classList.toggle("current-fragment", i === step - 1);
    }));
    s._says.forEach((p, i) => p.classList.toggle("on", i === step));
    s._dots.forEach((d, i) => { d.classList.toggle("on", i === step); d.classList.toggle("done", i < step); });
  };

  const geom = (s) => {
    const hdr = 56, vh = innerHeight;
    const r = s.getBoundingClientRect();
    const span = s.offsetHeight - (vh - hdr);
    return { r, vh, hdr, span, p: Math.min(1, Math.max(0, (hdr - r.top) / Math.max(1, span))) };
  };

  const update = () => {
    scenes.forEach((s) => {
      const { r, vh, p } = geom(s);
      const n = s._groups.length;
      if (r.bottom < -vh || r.top > 2 * vh) return;
      setStep(s, Math.min(n, Math.floor(p * (n + 1))));
      if (r.top < vh * 0.8 && r.bottom > vh * 0.3) s._slide.classList.add("present");
      else if (r.top >= vh || r.bottom <= 0) s._slide.classList.remove("present");
    });
    const d = document.documentElement;
    if (bar) bar.style.setProperty("--p", scrollY / Math.max(1, d.scrollHeight - innerHeight));
  };

  // 단계 점·화면 클릭으로 이동
  const goTo = (s, step) => {
    const { r, hdr, span } = geom(s);
    const n = s._groups.length;
    const y = scrollY + r.top - hdr + span * ((step + 0.5) / (n + 1));
    scrollTo({ top: y, behavior: "smooth" });
  };
  scenes.forEach((s, k) => {
    s._dots.forEach((d) => d.addEventListener("click", () => goTo(s, +d.dataset.go)));
    s._frame.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const n = s._groups.length;
      if (s._step < n) goTo(s, s._step + 1);
      else {
        const next = scenes[k + 1];
        if (next) scrollTo({ top: scrollY + next.getBoundingClientRect().top - 56, behavior: "smooth" });
      }
    });
  });

  let raf = 0;
  const onScroll = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(); }); };
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", () => { fit(); update(); });
  fit(); update();
  document.fonts && document.fonts.ready.then(fit);
})();
