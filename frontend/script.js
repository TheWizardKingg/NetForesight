/* ================= UX UTILITIES ================= */
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const _animFrames = new WeakMap();

/** Smoothly counts a number element from its current value to `to`. */
function animateValue(el, to, { decimals = 0, duration = 550, suffix = '' } = {}){
    if(!el) return;
    const from = parseFloat((el.textContent || '0').replace(/[^0-9.\-]/g, '')) || 0;
    if(prefersReducedMotion || Math.abs(to - from) < (decimals ? 0.05 : 1)){
        el.textContent = to.toFixed(decimals) + suffix;
        return;
    }
    if(_animFrames.has(el)) cancelAnimationFrame(_animFrames.get(el));
    const start = performance.now();
    const ease = t => 1 - Math.pow(1 - t, 3); // cubic ease-out
    function tick(now){
        const p = Math.min(1, (now - start) / duration);
        const val = from + (to - from) * ease(p);
        el.textContent = (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
        if(p < 1){
            _animFrames.set(el, requestAnimationFrame(tick));
        } else {
            el.textContent = (decimals ? to.toFixed(decimals) : to.toLocaleString()) + suffix;
        }
    }
    _animFrames.set(el, requestAnimationFrame(tick));
}

/** Briefly flashes an element to draw the eye to a changed value. */
function flashValue(el){
    if(!el || prefersReducedMotion) return;
    el.classList.remove('value-flash');
    void el.offsetWidth; // restart animation
    el.classList.add('value-flash');
}

/** Pushes a live alert into the toast feed, auto-dismissing after a few seconds. */
function showToast(title, body, color = '#a855f7'){
    const stack = document.getElementById('toastStack');
    if(!stack) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <span class="toast-dot" style="background:${color}; box-shadow:0 0 8px ${color};"></span>
        <div>
            <div class="toast-title" style="color:${color};">${title}</div>
            <div class="toast-body">${body}</div>
        </div>`;
    stack.appendChild(toast);
    while(stack.children.length > 4) stack.removeChild(stack.firstChild);
    setTimeout(() => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    }, 5000);
}

/* ================= ATTACK FORECAST TREE ENGINE ================= */
const TREE_DATA = [
    { id:0,  parent:null, name:"Suspicious Network Activity", prob:100, conf:96, desc:"Anomalous behaviour deviating from baseline network traffic patterns." },

    { id:1,  parent:0, name:"Initial Access",       prob:64, conf:82, desc:"Adversary attempts to gain an initial foothold into the network." },
    { id:2,  parent:0, name:"Discovery",             prob:23, desc:"Attacker enumerates internal hosts, services and network topology." },
    { id:3,  parent:0, name:"Credential Access",     prob:13, desc:"Attempt to steal account credentials for further access." },

    { id:4,  parent:1, name:"Execution",             prob:71, desc:"Malicious code is executed on the compromised host." },
    { id:5,  parent:1, name:"Persistence",           prob:29, desc:"Attacker maintains footholds across system restarts or credential changes." },

    { id:6,  parent:2, name:"Lateral Movement",      prob:55, desc:"Pivoting to other hosts using discovered network information." },
    { id:7,  parent:2, name:"Collection",            prob:45, desc:"Gathering data of interest from discovered internal sources." },

    { id:8,  parent:3, name:"Privilege Escalation",  prob:67, desc:"Using stolen credentials to gain higher-level permissions." },
    { id:9,  parent:3, name:"Lateral Movement",      prob:33, desc:"Moving across the network using compromised credentials." },

    { id:10, parent:4, name:"Privilege Escalation",  prob:58, desc:"Exploiting executed code context to gain elevated privileges." },
    { id:11, parent:4, name:"Defense Evasion",       prob:42, desc:"Techniques used to avoid detection during execution." },

    { id:12, parent:5, name:"Defense Evasion",       prob:50, desc:"Hiding persistence mechanisms from security tooling." },
    { id:13, parent:5, name:"Credential Access",     prob:50, desc:"Harvesting credentials from the persistent foothold." },

    { id:14, parent:6, name:"Command and Control",   prob:62, desc:"Establishing outbound channel for remote operator control." },
    { id:15, parent:6, name:"Exfiltration",          prob:38, desc:"Extracting collected data from the compromised host directly." },

    { id:16, parent:7, name:"Exfiltration",          prob:80, desc:"Bulk transfer of collected sensitive data out of the network." },

    { id:17, parent:8, name:"Lateral Movement",      prob:70, desc:"Elevated access used to pivot deeper into the network." },
    { id:18, parent:8, name:"Defense Evasion",       prob:30, desc:"Elevated privileges used to disable security controls." },

    { id:19, parent:9, name:"Command and Control",   prob:55, desc:"Compromised host establishes covert C2 communication." },
    { id:20, parent:9, name:"Exfiltration",          prob:45, desc:"Sensitive data extracted using existing lateral access." },

    { id:21, parent:10, name:"Credential Access",    prob:64, desc:"Privileged access leveraged to dump further credentials." },
    { id:22, parent:10, name:"Lateral Movement",     prob:36, desc:"Escalated privileges enable movement to critical systems." },

    { id:23, parent:14, name:"Exfiltration",         prob:75, desc:"Data is exfiltrated through the established C2 channel." },
    { id:24, parent:14, name:"Impact",               prob:25, desc:"Operator uses C2 access to disrupt or destroy systems/data." },

    { id:25, parent:17, name:"Command and Control",  prob:58, desc:"Newly reached host establishes a secondary C2 channel." },
    { id:26, parent:17, name:"Collection",           prob:42, desc:"Sensitive data gathered from newly accessed internal systems." },
];

const BOX_W = 200, BOX_H = 62, COL_GAP = 92, ROW_HEIGHT = 76, PAD = 40;

function buildTreeAndRender(){
    const nodeMap = {};
    TREE_DATA.forEach(n => nodeMap[n.id] = { ...n, children: [] });
    const childrenOf = {};
    TREE_DATA.forEach(n => {
        if(n.parent !== null){
            childrenOf[n.parent] = childrenOf[n.parent] || [];
            childrenOf[n.parent].push(nodeMap[n.id]);
            nodeMap[n.parent].children.push(nodeMap[n.id]);
        }
    });
    const root = nodeMap[0];

    // Assign depth (BFS)
    (function assignDepth(node, depth){
        node.depth = depth;
        node.children.forEach(c => assignDepth(c, depth+1));
    })(root, 0);

    // Assign row via recursive centering
    let leafCounter = 0;
    (function assignRow(node){
        if(node.children.length === 0){
            node.row = leafCounter;
            leafCounter += 1;
        } else {
            node.children.forEach(assignRow);
            const rows = node.children.map(c => c.row);
            node.row = (Math.min(...rows) + Math.max(...rows)) / 2;
        }
    })(root);

    // Mark top-probability branch at every fork
    Object.values(childrenOf).forEach(childArr => {
        let top = childArr[0];
        childArr.forEach(c => { if(c.prob > top.prob) top = c; });
        top.isTop = true;
    });

    // Compute pixel positions
    const flat = Object.values(nodeMap);
    let maxDepth = 0, maxRow = 0;
    flat.forEach(n => {
        n.x = PAD + n.depth * (BOX_W + COL_GAP);
        n.y = PAD + n.row * ROW_HEIGHT;
        if(n.depth > maxDepth) maxDepth = n.depth;
        if(n.row > maxRow) maxRow = n.row;
    });

    const contentW = PAD*2 + BOX_W + maxDepth*(BOX_W+COL_GAP);
    const contentH = PAD*2 + BOX_H + maxRow*ROW_HEIGHT;

    const canvas = document.getElementById('forecastTreeCanvas');
    const svg = document.getElementById('forecastTreeSvg');
    canvas.style.width = contentW + 'px';
    canvas.style.height = contentH + 'px';
    svg.setAttribute('width', contentW);
    svg.setAttribute('height', contentH);

    // Draw connector lines first (so boxes render on top)
    flat.forEach(n => {
        if(n.parent === null) return;
        const p = nodeMap[n.parent];
        const x1 = p.x + BOX_W, y1 = p.y + BOX_H/2;
        const x2 = n.x, y2 = n.y + BOX_H/2;
        const midX = x1 + (x2-x1)/2;
        const d = `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;

        const path = document.createElementNS('http://www.w3.org/2000/svg','path');
        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.setAttribute('class', 'tree-line');
        path.dataset.child = n.id;

        if(n.isTop){
            path.setAttribute('stroke', 'rgba(129, 140, 248,0.55)');
            path.setAttribute('stroke-width', '2.4');
        } else {
            path.setAttribute('stroke', 'rgba(168,85,247,0.28)');
            path.setAttribute('stroke-width', '1.4');
        }
        svg.appendChild(path);
    });

    // Draw boxes
    flat.forEach(n => {
        const box = document.createElement('div');
        box.className = 'tree-box';
        box.style.left = n.x + 'px';
        box.style.top = n.y + 'px';
        box.style.width = BOX_W + 'px';
        box.style.height = BOX_H + 'px';
        box.dataset.id = n.id;

        if(n.parent === null) box.classList.add('tree-box-root');
        else if(n.isTop) box.classList.add('tree-box-top');

        let inner = '';
        if(n.parent === null) inner += `<div class="root-pill">Current State</div>`;
        inner += `<div class="stage-name">${n.name}</div>`;
        inner += `<div class="stage-prob">Probability: ${n.prob}%</div>`;
        if(n.conf) inner += `<div class="stage-conf">Confidence: ${n.conf}%</div>`;
        box.innerHTML = inner;

        box.addEventListener('mouseenter', () => highlightPath(n.id, nodeMap));
        box.addEventListener('mouseleave', clearHighlight);

        canvas.appendChild(box);
    });
}

function highlightPath(id, nodeMap){
    const ancestors = new Set();
    let cur = nodeMap[id];
    while(cur){
        ancestors.add(cur.id);
        cur = cur.parent !== null ? nodeMap[cur.parent] : null;
    }

    document.querySelectorAll('.tree-box').forEach(box => {
        const bid = parseInt(box.dataset.id);
        box.classList.toggle('highlight', ancestors.has(bid));
        box.classList.toggle('dim', !ancestors.has(bid));
    });

    document.querySelectorAll('.tree-line').forEach(line => {
        const cid = parseInt(line.dataset.child);
        line.classList.toggle('highlight', ancestors.has(cid));
        line.classList.toggle('dim', !ancestors.has(cid));
    });

    const node = nodeMap[id];
    document.getElementById('treeInfoText').innerHTML =
        `<span style="color:#c084fc; font-weight:700;">${node.name}</span> — Probability: <span style="color:#818cf8; font-weight:700;">${node.prob}%</span>${node.conf ? ` · Confidence: <span style="color:#818cf8;">${node.conf}%</span>` : ''} <br><span style="color:#948ba8;">${node.desc}</span>`;
}

function clearHighlight(){
    document.querySelectorAll('.tree-box').forEach(b => { b.classList.remove('highlight','dim'); });
    document.querySelectorAll('.tree-line').forEach(l => { l.classList.remove('highlight','dim'); });
    document.getElementById('treeInfoText').textContent = 'Hover over any predicted state above to trace its attack path and view details.';
}

/* ================= REST OF DASHBOARD ================= */
const ATTACK_CHAIN = [
    { id:0, name:"Reconnaissance", fullName:"Network Reconnaissance", risk:22, prob:0, conf:0, next:null,
      mitreTactic:"Reconnaissance",
      description:"Passive network enumeration detected", events:["🔍 Port scanning detected","📡 DNS probing activity"],
      alert:"Monitoring reconnaissance phase", anomaly:0.15 },

    { id:1, name:"Network Scan", fullName:"Active Scanning & Probing", risk:38, prob:0.45, conf:0, next:"Initial Access",
      mitreTactic:"Reconnaissance",
      description:"Intensified network scanning detected", events:["🎯 Multiple port connections","⚠️ Service fingerprinting"],
      alert:"Escalating reconnaissance activity", anomaly:0.38 },

    { id:2, name:"Initial Access", fullName:"Exploitation & Access Gained", risk:55, prob:0.60, conf:0.72, next:"Lateral Movement",
      mitreTactic:"Initial Access",
      description:"System compromise confirmed", events:["🔓 Unauthorized access","💥 Payload execution"],
      alert:"CRITICAL: Initial access compromised", anomaly:0.55 },

    { id:3, name:"Lateral Movement", fullName:"Internal System Propagation", risk:68, prob:0.72, conf:0.81, next:"Command and Control",
      mitreTactic:"Lateral Movement",
      description:"Lateral movement in progress", events:["🔀 Internal lateral connections","👤 Privilege escalation attempt"],
      alert:"ALERT: Attacker spreading through network", anomaly:0.68 },

    { id:4, name:"Command & Control", fullName:"Malicious C2 Communication", risk:78, prob:0.78, conf:0.89, next:"Data Exfiltration",
      mitreTactic:"Command and Control",
      description:"C2 communication established", events:["📤 Suspicious outbound traffic","🎯 C2 beacon detected"],
      alert:"CRITICAL: C2 communication confirmed", anomaly:0.78 },

    { id:5, name:"Data Exfiltration", fullName:"Sensitive Data Extraction", risk:74, prob:0.61, conf:0.75, next:null,
      mitreTactic:"Exfiltration",
      description:"Data breach in progress", events:["🗂️ Database query surge","📦 Large data transfer"],
      alert:"CRITICAL: Data exfiltration underway", anomaly:0.74 },
];

let STATE = { step:0, flows:82, packets:2140, risk:5, anomaly:0, trafficHistory:[] };
for (let i = 0; i < 20; i++) {
    STATE.trafficHistory.push({ incoming: 2100 + Math.random()*100, outgoing: 1600 + Math.random()*100 });
}
let CHARTS = {};
const rand = (min,max) => Math.random()*(max-min)+min;
const formatTime = () => new Date().toLocaleTimeString("en-US",{hour12:false});

function initCharts(){
    const ctx = document.getElementById('trafficReportChart');
    if(!ctx) return;
    CHARTS.traffic = new Chart(ctx, {
        type: 'line',
        data: {
            labels: STATE.trafficHistory.map((_,i)=>i),
            datasets: [
                { label: 'Incoming Traffic', data: STATE.trafficHistory.map(d=>d.incoming),
                  borderColor: '#818cf8', backgroundColor: 'rgba(129, 140, 248, 0.10)',
                  tension: 0.4, fill: true, pointRadius: 0, borderWidth: 2.5 },
                { label: 'Outgoing Traffic', data: STATE.trafficHistory.map(d=>d.outgoing),
                  borderColor: '#a855f7', backgroundColor: 'rgba(168, 85, 247, 0.10)',
                  tension: 0.4, fill: true, pointRadius: 0, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13,8,21,0.9)',
                    borderColor: 'rgba(168,85,247,0.4)', borderWidth: 1,
                    titleFont: { family: 'Rajdhani' }, bodyFont: { family: 'Rajdhani' }
                }
            },
            scales: {
                y: { beginAtZero: false, grid: { color: 'rgba(168,85,247,0.08)' },
                     ticks: { color: '#948ba8', font: { family:'Rajdhani' } } },
                x: { grid: { display: false },
                     ticks: { color: '#948ba8', font: { family:'Rajdhani' } } }
            }
        }
    });
}

function updateTrafficChart(){
    if(!CHARTS.traffic) return;
    CHARTS.traffic.data.labels = STATE.trafficHistory.map((_,i)=>i);
    CHARTS.traffic.data.datasets[0].data = STATE.trafficHistory.map(d=>d.incoming);
    CHARTS.traffic.data.datasets[1].data = STATE.trafficHistory.map(d=>d.outgoing);
    CHARTS.traffic.update('none');
}

function updateUI(){
    const stage = ATTACK_CHAIN[Math.min(STATE.step, ATTACK_CHAIN.length-1)];

    animateValue(document.getElementById("flowsPerSec"), STATE.flows, { duration: 650 });
    animateValue(document.getElementById("packetsPerSec"), STATE.packets, { duration: 650 });

    animateValue(document.getElementById("eventCount"), Math.round(STATE.risk*0.3), { duration: 400 });
    let eventHTML = "";
    if(stage.events?.length){
        stage.events.forEach((evt,idx)=>{
            const color = idx===0 ? "#fb7185" : "#fbbf24";
            eventHTML += `<div class="slide-up text-xs flex justify-between gap-2 font-semibold" style="color:${color}"><span>${evt}</span><span class="mono-num text-purple-300/30" style="font-size:10px;">${formatTime()}</span></div>`;
        });
    }
    document.getElementById("eventsList").innerHTML = eventHTML || '<div class="text-xs text-purple-300/40">✓ No incoming anomalies</div>';

    let status, statusColor, statusDesc, glowClass, threatLabel;
    if(STATE.risk < 40){
        status="NORMAL"; statusColor="#34d399"; statusDesc="✓ All systems operating normally";
        glowClass="glow-green"; threatLabel="LOW";
    } else if(STATE.risk <= 70){
        status="AT RISK"; statusColor="#fbbf24"; statusDesc="⚠ Suspicious activity — heightened alert";
        glowClass="glow-amber"; threatLabel="MEDIUM";
    } else {
        status="UNDER ATTACK"; statusColor="#fb7185"; statusDesc="🚨 CRITICAL threat — immediate action needed";
        glowClass="glow-red"; threatLabel="CRITICAL";
    }

    document.getElementById("statusIndicator").style.background = statusColor;
    document.getElementById("statusIndicator").style.boxShadow = `0 0 10px ${statusColor}`;
    const statusLabelEl = document.getElementById("statusLabel");
    statusLabelEl.style.color = statusColor;
    if(statusLabelEl.textContent !== status){
        statusLabelEl.textContent = status;
        flashValue(statusLabelEl);
    }
    document.getElementById("statusDesc").textContent = statusDesc;
    animateValue(document.getElementById("riskScoreVal"), Math.round(STATE.risk), { duration: 500 });

    const statusCard = document.getElementById("statusCardGlow");
    statusCard.classList.remove("glow-green","glow-amber","glow-red");
    statusCard.classList.add(glowClass);

    const threatLabelEl = document.getElementById("threatLevelLabel");
    if(threatLabelEl.textContent !== threatLabel){
        threatLabelEl.textContent = threatLabel;
        flashValue(threatLabelEl);
    }
    threatLabelEl.style.color = statusColor;
    const litCount = Math.min(5, Math.max(1, Math.ceil(STATE.risk / 20)));
    const segments = document.querySelectorAll("#threatSegments .segment");
    segments.forEach((seg, i)=>{
        if(i < litCount){
            seg.style.background = statusColor;
            seg.style.borderColor = statusColor;
        } else {
            seg.style.background = "rgba(168,85,247,0.08)";
            seg.style.borderColor = "rgba(168,85,247,0.15)";
        }
    });
    document.getElementById("confidence").textContent = stage.conf ? (stage.conf*100).toFixed(0)+"%" : "—";

    document.getElementById("currentStage").textContent = stage.fullName;
    document.getElementById("stageDescription").textContent = stage.description;
    document.getElementById("predictedStage").textContent = stage.next || "—";
    document.getElementById("predictedTime").textContent = stage.next ? `${Math.round(stage.prob*100)}% likely` : "—";

    if(stage.next && stage.prob > 0){
        document.getElementById("probabilitySection").style.display = "block";
        animateValue(document.getElementById("probValue"), Math.round(stage.prob*100), { suffix: "%", duration: 500 });
        animateValue(document.getElementById("probPercentage"), Math.round(stage.prob*100), { suffix: "%", duration: 500 });
        document.getElementById("probBar").style.width = (stage.prob*100)+"%";
    } else {
        document.getElementById("probabilitySection").style.display = "none";
    }

    document.getElementById("forecastAlert").style.display = stage.next ? "block" : "none";
    document.getElementById("forecastText").textContent = stage.alert;

    document.querySelectorAll("#mitreTableBody tr").forEach(row=>{
        row.classList.toggle("active-row", row.dataset.tactic === stage.mitreTactic);
    });

    animateValue(document.getElementById("anomalyScore"), STATE.anomaly*10, { decimals: 1, duration: 500 });
    animateValue(document.getElementById("detectionConf"), stage.conf ? Math.round(stage.conf*100) : 0, { duration: 500 });
    document.getElementById("stageProgress").textContent = `${STATE.step+1}/${ATTACK_CHAIN.length}`;
    document.getElementById("lastUpdate").textContent = formatTime();

    updateTrafficChart();
}

function jitterTraffic(){
    const risk = STATE.risk;
    STATE.flows = Math.max(10, Math.round(80 + risk*3 + rand(-15,15)));
    STATE.packets = Math.max(200, Math.round(2100 + risk*60 + rand(-200,200)));

    STATE.trafficHistory.push({
        incoming: Math.max(200, 2100 + risk*55 + rand(-150,150)),
        outgoing: Math.max(150, 1600 + risk*40 + rand(-120,120)),
    });
    if(STATE.trafficHistory.length > 24) STATE.trafficHistory.shift();

    for(let i = 0; i < 5; i++){
        const bar = document.getElementById(`bar${i}`);
        if(bar) bar.style.height = Math.max(15, Math.min(95, 45 + risk*0.5 + rand(-18,18))) + "%";
    }

    updateUI();
}

function advanceStep(){
    if(STATE.step < ATTACK_CHAIN.length - 1){
        STATE.step += 1;
    } else {
        setTimeout(()=>{
            STATE.step = 0; STATE.risk = 5; STATE.anomaly = 0; updateUI();
            showToast("MONITORING RESUMED", "Attack simulation reset — network back to baseline.", "#34d399");
        }, 2500);
        return;
    }
    const stage = ATTACK_CHAIN[STATE.step];
    STATE.risk = stage.risk;
    STATE.anomaly = stage.anomaly;
    updateUI();

    const toastColor = stage.risk < 40 ? "#34d399" : stage.risk <= 70 ? "#fbbf24" : "#fb7185";
    showToast(stage.fullName, stage.alert, toastColor);
}

buildTreeAndRender();
initCharts();
updateUI();
setInterval(jitterTraffic, 700);
setInterval(advanceStep, 4200);

/* ================= NAV SCROLLSPY ================= */
(function initScrollspy(){
    const targets = [
        { observeId: "dashboardAnchor", href: "#dashboard" },
        { observeId: "prediction",      href: "#prediction" },
        { observeId: "mitre",           href: "#mitre" },
        { observeId: "traffic-report",  href: "#traffic-report" },
    ];
    const navLinks = Array.from(document.querySelectorAll('nav a.nav-link'));
    if(!navLinks.length) return;

    const setActive = (href) => {
        navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === href));
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if(entry.isIntersecting){
                const match = targets.find(t => t.observeId === entry.target.id);
                if(match) setActive(match.href);
            }
        });
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });

    targets.forEach(t => {
        const el = document.getElementById(t.observeId);
        if(el) observer.observe(el);
    });
})();

/* ================= HEADER SHRINK ON SCROLL ================= */
(function initHeaderShrink(){
    const header = document.querySelector('header.site-header');
    if(!header) return;
    const onScroll = () => header.classList.toggle('scrolled', window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
})();

/* ================= STAGGERED REVEAL ON SCROLL ================= */
(function initScrollReveal(){
    if(prefersReducedMotion) return;
    const groups = document.querySelectorAll('main > .grid, #prediction');
    groups.forEach(group => {
        Array.from(group.children).forEach((child, i) => {
            child.classList.add('reveal');
            child.style.transitionDelay = `${Math.min(i, 4) * 70}ms`;
        });
    });
    const soloCards = document.querySelectorAll('main > .glass-strong, main > .glass');
    soloCards.forEach(card => card.classList.add('reveal'));

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if(entry.isIntersecting){
                entry.target.classList.add('revealed');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });

    document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));
})();
