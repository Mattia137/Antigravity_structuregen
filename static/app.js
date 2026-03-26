import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// ═══════════════════════════════════════════
// SCENE SETUP
// ═══════════════════════════════════════════
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(1200, 800, 1200);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(400, 400, 400);
controls.update();

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(500, 1000, 500);
scene.add(dirLight);

// Grid helper
const grid = new THREE.GridHelper(2000, 20, 0x111111, 0x080808);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// ═══════════════════════════════════════════
// GROUPS
// ═══════════════════════════════════════════
const meshGroup = new THREE.Group();
const structureGroup = new THREE.Group();
scene.add(meshGroup);
scene.add(structureGroup);

// ═══════════════════════════════════════════
// OBJ MESH LOADER
// ═══════════════════════════════════════════
let currentMeshUrl = '/mesh/mass-DEF.obj';

function loadMesh(url) {
    while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);

    const loader = new OBJLoader();
    loader.load(url, (obj) => {
        obj.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshPhongMaterial({
                    color: 0x4488cc,
                    transparent: true,
                    opacity: 0.08,
                    wireframe: false,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const wire = new THREE.WireframeGeometry(child.geometry);
                const wireLines = new THREE.LineSegments(wire,
                    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06 })
                );
                meshGroup.add(wireLines);
            }
        });
        meshGroup.add(obj);

        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        controls.target.copy(center);
        camera.position.set(center.x + maxDim, center.y + maxDim * 0.6, center.z + maxDim);
        controls.update();

        setStatus('MESH LOADED');
    }, undefined, (err) => {
        console.error('OBJ load error:', err);
        setStatus('MESH LOAD FAILED');
    });
}

loadMesh(currentMeshUrl);

document.getElementById('mesh-upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('upload-text').textContent = file.name;
    const url = URL.createObjectURL(file);
    loadMesh(url);
});

// ═══════════════════════════════════════════
// STRUCTURE RENDERING
// ═══════════════════════════════════════════

function turboColormap(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.max(0, Math.min(1, 0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.6770 - t * 58.1544))))));
    const g = Math.max(0, Math.min(1, 0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 + t * 2.7747))))));
    const b = Math.max(0, Math.min(1, 0.1067 + t * (12.5925 - t * (60.1097 - t * (109.0745 - t * (88.5066 - t * 26.8183))))));
    return new THREE.Color(r, g, b);
}

function renderStructure(data, useGradient) {
    while (structureGroup.children.length) structureGroup.remove(structureGroup.children[0]);

    const nodes = data.nodes;
    const members = data.members;

    members.forEach((m) => {
        const n1 = nodes[m.from];
        const n2 = nodes[m.to];
        if (!n1 || !n2) return;

        // Y-up: swap Y and Z for Three.js (Three uses Y-up natively, OBJ Y-up maps directly)
        const points = [
            new THREE.Vector3(n1.x, n1.y, n1.z),
            new THREE.Vector3(n2.x, n2.y, n2.z)
        ];

        if (useGradient) {
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const colors = new Float32Array(6);
            const c1 = turboColormap(m.disp_i);
            const c2 = turboColormap(m.disp_j);
            colors[0] = c1.r; colors[1] = c1.g; colors[2] = c1.b;
            colors[3] = c2.r; colors[4] = c2.g; colors[5] = c2.b;
            geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
            structureGroup.add(new THREE.LineSegments(geo, mat));
        } else {
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const color = m.role === 'primary_crease' ? 0xffffff
                        : m.role === 'secondary_lattice' ? 0x66aaff : 0x888888;
            const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 });
            structureGroup.add(new THREE.Line(geo, mat));
        }
    });

    const nodePositions = [];
    Object.values(nodes).forEach(n => {
        nodePositions.push(n.x, n.y, n.z);
    });
    const nodeGeo = new THREE.BufferGeometry();
    nodeGeo.setAttribute('position', new THREE.Float32BufferAttribute(nodePositions, 3));
    const nodeMat = new THREE.PointsMaterial({ color: 0xffffff, size: 2, transparent: true, opacity: 0.3 });
    structureGroup.add(new THREE.Points(nodeGeo, nodeMat));
}

// ═══════════════════════════════════════════
// VARIANT STATE
// ═══════════════════════════════════════════
let allRuns = [];         // [{name, variants: [{name, data, metrics}]}]
let activeRunIdx = 0;
let activeVariantIdx = 1; // default: BALANCED

function getActiveVariant() {
    if (!allRuns[activeRunIdx]) return null;
    return allRuns[activeRunIdx].variants[activeVariantIdx];
}

// ═══════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════
async function evaluate() {
    setStatus('<span class="spinner"></span>ANALYZING...');
    const btn = document.getElementById('btn-evaluate');
    btn.disabled = true;

    const unit = document.getElementById('unit-system').value;

    const payload = {
        material: document.getElementById('material-preset').value,
        num_floors: parseInt(document.getElementById('num-floors').value),
        floor_height: parseFloat(document.getElementById('floor-height').value),
        unit: unit,
        custom_props: {
            E: parseFloat(document.getElementById('mat-E').value),
            nu: parseFloat(document.getElementById('mat-nu').value),
            rho: parseFloat(document.getElementById('mat-rho').value),
            Strength: parseFloat(document.getElementById('mat-str').value)
        }
    };

    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            cache: 'no-store',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.error) {
            setStatus('ANALYSIS FAILED: ' + data.error);
            btn.disabled = false;
            return;
        }

        // Store run with all 3 variants
        const runName = 'RUN_' + (allRuns.length + 1);
        const variants = (data.variants || []).map(v => ({
            name: v.name,
            data: { nodes: v.nodes, members: v.members, max_disp: v.max_disp },
            metrics: v.metrics
        }));

        // Fallback: if server doesn't return variants array, wrap the legacy flat response
        if (variants.length === 0 && data.nodes) {
            variants.push({ name: 'BALANCED', data: { nodes: data.nodes, members: data.members, max_disp: data.max_disp }, metrics: data.metrics });
        }

        activeRunIdx = allRuns.length;
        activeVariantIdx = Math.min(1, variants.length - 1); // default BALANCED
        allRuns.push({ name: runName, variants });

        showVariant(activeRunIdx, activeVariantIdx);
        updateRunList();
        drawChart();
        setStatus(runName + ' COMPLETE — ' + variants.length + ' VARIANTS');

    } catch (err) {
        console.error(err);
        setStatus('ERROR — CHECK CONSOLE');
    }
    btn.disabled = false;
}

function showVariant(runIdx, varIdx) {
    activeRunIdx = runIdx;
    activeVariantIdx = varIdx;
    const v = allRuns[runIdx]?.variants[varIdx];
    if (!v) return;

    // Update metrics panel
    document.getElementById('m-carbon').textContent = Number(v.metrics.Carbon_kgCO2e).toLocaleString();
    document.getElementById('m-cost').textContent = '$' + Number(v.metrics.Cost_USD).toLocaleString();
    document.getElementById('m-volume').textContent = Number(v.metrics.Volume).toLocaleString();
    document.getElementById('m-drift').textContent = Number(v.metrics.Max_Disp).toFixed(4);

    const showGradient = document.getElementById('show-gradient').checked;
    renderStructure(v.data, showGradient);

    // Auto-fit camera to structure
    const structureBbox = new THREE.Box3().setFromObject(structureGroup);
    if (!structureBbox.isEmpty()) {
        const center = structureBbox.getCenter(new THREE.Vector3());
        const size = structureBbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        controls.target.copy(center);
        camera.position.set(
            center.x + maxDim * 1.2,
            center.y + maxDim * 0.8,
            center.z + maxDim * 1.2
        );
        controls.update();
    }

    // Highlight active variant tabs
    document.querySelectorAll('.variant-tab').forEach((el, i) => {
        el.classList.toggle('active', i === varIdx);
    });
}

function updateRunList() {
    const list = document.getElementById('iteration-list');
    list.innerHTML = '';

    allRuns.forEach((run, runIdx) => {
        // Run header
        const header = document.createElement('div');
        header.className = 'iter-run-header';
        header.textContent = run.name;
        list.appendChild(header);

        // Variant tabs for this run
        const tabRow = document.createElement('div');
        tabRow.className = 'variant-tabs';
        run.variants.forEach((v, varIdx) => {
            const tab = document.createElement('div');
            tab.className = 'variant-tab' + (runIdx === activeRunIdx && varIdx === activeVariantIdx ? ' active' : '');
            tab.innerHTML = `<span class="vname">${v.name}</span><br>
                <span class="vmeta">C: ${Number(v.metrics.Carbon_kgCO2e).toLocaleString()} kg</span><br>
                <span class="vmeta">$${Number(v.metrics.Cost_USD).toLocaleString()}</span><br>
                <span class="vmeta">δ: ${Number(v.metrics.Max_Disp).toFixed(4)} m</span>`;
            tab.addEventListener('click', () => showVariant(runIdx, varIdx));
            tabRow.appendChild(tab);
        });
        list.appendChild(tabRow);
    });
}

// ═══════════════════════════════════════════
// UNIT TOGGLE
// ═══════════════════════════════════════════
function updateUnitLabels() {
    const unit = document.getElementById('unit-system').value;
    const isM = unit === 'm';
    document.getElementById('lbl-floor-height').textContent = isM ? 'FLOOR_HEIGHT (M)' : 'FLOOR_HEIGHT (FT)';
    document.getElementById('lbl-volume').textContent = isM ? 'VOLUME (M³)' : 'VOLUME (FT³)';
    document.getElementById('lbl-drift').textContent = isM ? 'MAX DRIFT (M)' : 'MAX DRIFT (FT)';
}

document.getElementById('unit-system').addEventListener('change', () => {
    const unit = document.getElementById('unit-system').value;
    const floorInput = document.getElementById('floor-height');
    if (unit === 'm') {
        floorInput.value = 3.66; floorInput.step = 0.1;
    } else {
        floorInput.value = 12; floorInput.step = 1;
    }
    updateUnitLabels();
});

// ═══════════════════════════════════════════
// MINI CHART (Canvas 2D) — compares all 3 variants of latest run
// ═══════════════════════════════════════════
function drawChart() {
    const cv = document.getElementById('chart-canvas');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    const run = allRuns[activeRunIdx];
    if (!run) return;

    const w = cv.width;
    const h = cv.height;
    const variants = run.variants;
    const barW = Math.min(28, (w - 20) / (variants.length * 3 + variants.length));
    const pad = 10;

    let maxC = 1, maxCost = 1, maxV = 1;
    variants.forEach(v => {
        maxC    = Math.max(maxC,    v.metrics.Carbon_kgCO2e);
        maxCost = Math.max(maxCost, v.metrics.Cost_USD);
        maxV    = Math.max(maxV,    v.metrics.Volume);
    });

    variants.forEach((v, i) => {
        const x = pad + i * (barW * 3 + 8);
        const hC  = (v.metrics.Carbon_kgCO2e / maxC)   * (h - 30);
        const hCo = (v.metrics.Cost_USD       / maxCost) * (h - 30);
        const hV  = (v.metrics.Volume         / maxV)    * (h - 30);

        ctx.fillStyle = 'rgba(100,200,100,0.5)';
        ctx.fillRect(x, h - 15 - hC, barW, hC);

        ctx.fillStyle = 'rgba(100,150,255,0.5)';
        ctx.fillRect(x + barW, h - 15 - hCo, barW, hCo);

        ctx.fillStyle = 'rgba(255,160,60,0.5)';
        ctx.fillRect(x + barW * 2, h - 15 - hV, barW, hV);

        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '6px Fragment Mono';
        ctx.fillText(v.name, x, h - 4);
    });
}

// ═══════════════════════════════════════════
// EVENT BINDINGS
// ═══════════════════════════════════════════
document.getElementById('btn-evaluate').addEventListener('click', evaluate);
document.getElementById('btn-clear').addEventListener('click', () => {
    allRuns = [];
    activeRunIdx = 0;
    activeVariantIdx = 1;
    while (structureGroup.children.length) structureGroup.remove(structureGroup.children[0]);
    document.getElementById('m-carbon').textContent = '—';
    document.getElementById('m-cost').textContent = '—';
    document.getElementById('m-volume').textContent = '—';
    document.getElementById('m-drift').textContent = '—';
    document.getElementById('iteration-list').innerHTML = '';
    drawChart();
    setStatus('MEMORY CLEARED');
});

document.getElementById('num-floors').addEventListener('input', (e) => {
    document.getElementById('num-floors-val').textContent = e.target.value;
});

document.getElementById('material-preset').addEventListener('change', async () => {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    const mat = document.getElementById('material-preset').value;
    const props = cfg.materials[mat];
    document.getElementById('mat-E').value = props.E;
    document.getElementById('mat-nu').value = props.nu;
    document.getElementById('mat-rho').value = props.rho;
    document.getElementById('mat-str').value = props.Strength;
});

document.getElementById('show-mesh').addEventListener('change', (e) => {
    meshGroup.visible = e.target.checked;
});
document.getElementById('show-structure').addEventListener('change', (e) => {
    structureGroup.visible = e.target.checked;
});
document.getElementById('show-gradient').addEventListener('change', () => {
    const v = getActiveVariant();
    if (v) renderStructure(v.data, document.getElementById('show-gradient').checked);
});

// ═══════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════
function setStatus(text) {
    document.getElementById('hud-status').innerHTML = text;
}

// ═══════════════════════════════════════════
// RESIZE & RENDER LOOP
// ═══════════════════════════════════════════
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
