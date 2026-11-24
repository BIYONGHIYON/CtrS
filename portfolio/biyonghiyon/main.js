// 스크롤에 맞춰 나타나는 효과와 몇 가지 움직이는 도표
(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // 들어올 때 한 번 나타남
  const onIn = new Map();
  let pending = [...document.querySelectorAll("[data-r]")];
  const reveal = (el) => {
    if (el.classList.contains("in")) return;
    el.classList.add("in");
    io.unobserve(el);
    (onIn.get(el) || []).forEach((f) => f());
  };
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) reveal(e.target); }),
    { rootMargin: "0px 0px -12% 0px", threshold: 0.12 });
  const when = (el, f) => { if (!onIn.has(el)) onIn.set(el, []); onIn.get(el).push(f); };
  pending.forEach((el) => io.observe(el));
  // 화면 안에 이미 있는 것은 바로, 그 밖은 스크롤할 때도 한 번 더 확인
  const sweep = () => {
    const h = innerHeight * 0.88;
    pending = pending.filter((el) => {
      const r = el.getBoundingClientRect();
      if (r.top < h && r.bottom > 0) { reveal(el); return false; }
      return !el.classList.contains("in");
    });
  };

  // 숫자 세기
  const fmt = (v, s) => v.toFixed((s.split(".")[1] || "").length);
  document.querySelectorAll("[data-count]").forEach((el) => {
    const host = el.closest("[data-r]");
    if (!host || reduce) return;
    const to = parseFloat(el.dataset.count), pre = el.dataset.pre || "", suf = el.dataset.suf || "";
    el.textContent = pre + fmt(0, el.dataset.count) + suf;
    when(host, () => {
      const t0 = performance.now(), dur = 1300;
      const step = (t) => {
        const p = Math.min(1, (t - t0) / dur), k = 1 - Math.pow(1 - p, 3);
        el.textContent = pre + fmt(to * k, el.dataset.count) + suf;
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  });

  // 첫 화면 비교 슬라이더
  const cmp = document.querySelector(".cmp");
  if (cmp) {
    const input = cmp.querySelector("input");
    let touched = false;
    const set = (v) => cmp.style.setProperty("--x", v + "%");
    input.addEventListener("input", () => { touched = true; set(input.value); });
    if (!reduce) {
      setTimeout(() => {
        const t0 = performance.now();
        const wig = (t) => {
          if (touched) return;
          const p = Math.min(1, (t - t0) / 1800);
          const v = 50 + 22 * Math.sin(p * Math.PI * 2) * (1 - p);
          set(v); input.value = v;
          if (p < 1) requestAnimationFrame(wig);
        };
        requestAnimationFrame(wig);
      }, 900);
    }
  }

  // 생성형 비교: 들어오면 잠시 뒤 확대
  const gen = document.querySelector(".gen");
  if (gen) {
    const btns = gen.querySelectorAll("[data-zoom]");
    let user = false;
    const zoom = (on) => {
      gen.classList.toggle("zoomed", on);
      btns.forEach((b) => b.classList.toggle("on", (b.dataset.zoom === "1") === on));
    };
    btns.forEach((b) => b.addEventListener("click", () => { user = true; zoom(b.dataset.zoom === "1"); }));
    when(gen, () => setTimeout(() => { if (!user) zoom(true); }, 1600));
  }

  // 밴드별 밝기 보정: 단계별 재생
  const cal = document.querySelector(".calib");
  if (cal) {
    const steps = cal.querySelectorAll(".cal-steps li[data-s]");
    let timer = 0;
    const show = (p) => {
      cal.dataset.phase = p;
      steps.forEach((li) => {
        const s = +li.dataset.s;
        li.classList.toggle("on", s === p);
        li.classList.toggle("done", s < p);
      });
    };
    const play = () => {
      clearTimeout(timer);
      show(0);
      if (reduce) return show(3);
      let p = 0;
      const next = () => { if (++p <= 3) { show(p); timer = setTimeout(next, 2000); } };
      timer = setTimeout(next, 2200);
    };
    cal.querySelector(".replay button").addEventListener("click", play);
    steps.forEach((li) => li.addEventListener("click", () => { clearTimeout(timer); show(+li.dataset.s); }));
    when(cal, play);
  }

  // 머리말: 진행 막대와 현재 장
  const bar = document.querySelector(".prog");
  const links = [...document.querySelectorAll(".hd nav a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
  let raf = 0;
  const tick = () => {
    raf = 0;
    if (pending.length) sweep();
    const d = document.documentElement;
    bar.style.setProperty("--p", scrollY / Math.max(1, d.scrollHeight - innerHeight));
    let cur = -1;
    secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top < innerHeight * 0.4) cur = i; });
    links.forEach((a, i) => a.classList.toggle("on", i === cur));
  };
  addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(tick); }, { passive: true });
  tick();
})();
