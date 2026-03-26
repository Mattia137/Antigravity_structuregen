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
let currentMeshUrl = 'mass-DEF.obj';

function loadMesh(url) {
    // Remove old mesh
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
                // Also add wireframe overlay
                const wire = new THREE.WireframeGeometry(child.geometry);
                const wireLines = new THREE.LineSegments(wire,
                    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06 })
                );
                meshGroup.add(wireLines);
            }
        });
        meshGroup.add(obj);

        // Auto-fit camera to mesh
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

// File upload
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
let iterations = [];

function turboColormap(t) {
    // Attempt to match the Turbo colormap
    t = Math.max(0, Math.min(1, t));
    const r = Math.max(0, Math.min(1, 0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.6770 - t * 58.1544))))));
    const g = Math.max(0, Math.min(1, 0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 + t * 2.7747))))));
    const b = Math.max(0, Math.min(1, 0.1067 + t * (12.5925 - t * (60.1097 - t * (109.0745 - t * (88.5066 - t * 26.8183))))));
    return new THREE.Color(r, g, b);
}

function renderStructure(data, useGradient) {
    // Clear previous
    while (structureGroup.children.length) structureGroup.remove(structureGroup.children[0]);

    const nodes = data.nodes;
    const members = data.members;

    // Draw members
    members.forEach((m) => {
        const n1 = nodes[m.from];
        const n2 = nodes[m.to];
        if (!n1 || !n2) return;

        const points = [
            new THREE.Vector3(n1.x, n1.z, n1.y),  // swap Y/Z for Three.js
            new THREE.Vector3(n2.x, n2.z, n2.y)
        ];

        if (useGradient) {
            // MIDAS-style: color by displacement
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
            const color = m.role === 'Primary' ? 0xffffff : m.role === 'Diagonal' ? 0x66aaff : 0x888888;
            const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 });
            structureGroup.add(new THREE.Line(geo, mat));
        }
    });

    // Draw nodes
    const nodePositions = [];
    Object.values(nodes).forEach(n => {
        nodePositions.push(n.x, n.z, n.y);  // swap Y/Z
    });
    const nodeGeo = new THREE.BufferGeometry();
    nodeGeo.setAttribute('position', new THREE.Float32BufferAttribute(nodePositions, 3));
    const nodeMat = new THREE.PointsMaterial({ color: 0xffffff, size: 2, transparent: true, opacity: 0.3 });
    structureGroup.add(new THREE.Points(nodeGeo, nodeMat));
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.error) {
            setStatus('ANALYSIS FAILED');
            btn.disabled = false;
            return;
        }

        // Update metrics (use new keys from server)
        document.getElementById('m-carbon').textContent = Number(data.metrics.Carbon_kgCO2e).toLocaleString();
        document.getElementById('m-cost').textContent = '$' + Number(data.metrics.Cost_USD).toLocaleString();
        document.getElementById('m-volume').textContent = Number(data.metrics.Volume).toLocaleString();
        document.getElementById('m-drift').textContent = data.metrics.Max_Disp.toFixed(4);

        // Store iteration
        const iterName = 'OPT_' + (iterations.length + 1);
        iterations.push({ name: iterName, data, metrics: data.metrics });

        // Render
        const showGradient = document.getElementById('show-gradient').checked;
        renderStructure(data, showGradient);

        updateIterationList();
        drawChart();

        setStatus(iterName + ' COMPLETE (' + unit.toUpperCase() + ')');
    } catch (err) {
        console.error(err);
        setStatus('ERROR — CHECK CONSOLE');
    }
    btn.disabled = false;
}

function updateIterationList() {
    const list = document.getElementById('iteration-list');
    list.innerHTML = '';
    iterations.forEach((it) => {
        const el = document.createElement('div');
        el.className = 'iter-item';
        el.innerHTML = `<span>${it.name}</span><span>${Number(it.metrics.Carbon_kgCO2e).toLocaleString()} kg</span>`;
        list.appendChild(el);
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
// MINI CHART (Canvas 2D)
// ═══════════════════════════════════════════
function drawChart() {
    const cv = document.getElementById('chart-canvas');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    if (iterations.length === 0) return;

    const w = cv.width;
    const h = cv.height;
    const barW = Math.min(30, (w - 20) / (iterations.length * 3 + iterations.length));
    const pad = 10;

    // Find max values
    let maxC = 1, maxCost = 1, maxV = 1;
    iterations.forEach(it => {
        maxC = Math.max(maxC, it.metrics.Carbon_kgCO2e);
        maxCost = Math.max(maxCost, it.metrics.Cost_USD);
        maxV = Math.max(maxV, it.metrics.Volume);
    });

    iterations.forEach((it, i) => {
        const x = pad + i * (barW * 3 + 8);
        const hC = (it.metrics.Carbon_kgCO2e / maxC) * (h - 30);
        const hCo = (it.metrics.Cost_USD / maxCost) * (h - 30);
        const hV = (it.metrics.Volume / maxV) * (h - 30);

        ctx.fillStyle = 'rgba(100,200,100,0.5)';
        ctx.fillRect(x, h - 15 - hC, barW, hC);

        ctx.fillStyle = 'rgba(100,150,255,0.5)';
        ctx.fillRect(x + barW, h - 15 - hCo, barW, hCo);

        ctx.fillStyle = 'rgba(255,160,60,0.5)';
        ctx.fillRect(x + barW * 2, h - 15 - hV, barW, hV);

        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '6px Fragment Mono';
        ctx.fillText(it.name, x, h - 4);
    });
}

// ═══════════════════════════════════════════
// EVENT BINDINGS
// ═══════════════════════════════════════════
document.getElementById('btn-evaluate').addEventListener('click', evaluate);
document.getElementById('btn-clear').addEventListener('click', () => {
    iterations = [];
    while (structureGroup.children.length) structureGroup.remove(structureGroup.children[0]);
    document.getElementById('m-carbon').textContent = '—';
    document.getElementById('m-cost').textContent = '—';
    document.getElementById('m-volume').textContent = '—';
    document.getElementById('m-drift').textContent = '—';
    updateIterationList();
    drawChart();
    setStatus('MEMORY CLEARED');
});

// Slider label
document.getElementById('num-floors').addEventListener('input', (e) => {
    document.getElementById('num-floors-val').textContent = e.target.value;
});

// Material preset auto-fill
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

// Visibility toggles
document.getElementById('show-mesh').addEventListener('change', (e) => {
    meshGroup.visible = e.target.checked;
});
document.getElementById('show-structure').addEventListener('change', (e) => {
    structureGroup.visible = e.target.checked;
});
document.getElementById('show-gradient').addEventListener('change', (e) => {
    if (iterations.length > 0) {
        const latest = iterations[iterations.length - 1];
        renderStructure(latest.data, e.target.checked);
    }
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
