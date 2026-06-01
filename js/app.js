/**
 * app.js — 수화 번역기
 * TF.js 없이 순수 JS로 신경망 추론
 * 구조: Dense(256,relu) → BN → Dense(128,relu) → BN → Dense(64,relu) → Dense(28,softmax)
 */

// ─── 상태 ─────────────────────────────────────────────────────────
let W = null;          // 가중치
let LABELS = [];       // 라벨
let outText = '';
let prevSign = '';
let prevTime = 0;
let stableCount = 0;
const STABLE = 20;
const COOLDOWN = 1200;

// ─── DOM ──────────────────────────────────────────────────────────
const video    = document.getElementById('webcam');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const elSign   = document.getElementById('current-sign');
const elText   = document.getElementById('recognized-text');
const elStatus = document.getElementById('status');
const elBar    = document.getElementById('progress-bar');

// ─── 순수 JS 신경망 ───────────────────────────────────────────────

function dot(x, kernel, bias) {
  // x: Float32Array (n,)  kernel: Array (n, m)  bias: Array (m,)
  const m = kernel[0].length;
  const out = new Float32Array(m);
  for (let j = 0; j < m; j++) {
    let s = bias[j];
    for (let i = 0; i < x.length; i++) s += x[i] * kernel[i][j];
    out[j] = s;
  }
  return out;
}

function relu(x) { return x.map(v => v > 0 ? v : 0); }

function softmax(x) {
  const max = Math.max(...x);
  const e = x.map(v => Math.exp(v - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function batchNorm(x, gamma, beta, mean, variance) {
  const eps = 0.001;
  return x.map((v, i) => gamma[i] * (v - mean[i]) / Math.sqrt(variance[i] + eps) + beta[i]);
}

function forward(input) {
  let x = new Float32Array(input);

  // Dense 256 + relu
  x = relu(dot(x, W.dense[0], W.dense[1]));
  // BatchNorm
  x = batchNorm(x, W.batch_normalization[0], W.batch_normalization[1],
                    W.batch_normalization[2], W.batch_normalization[3]);
  // Dense 128 + relu
  x = relu(dot(x, W.dense_1[0], W.dense_1[1]));
  // BatchNorm
  x = batchNorm(x, W.batch_normalization_1[0], W.batch_normalization_1[1],
                    W.batch_normalization_1[2], W.batch_normalization_1[3]);
  // Dense 64 + relu
  x = relu(dot(x, W.dense_2[0], W.dense_2[1]));
  // Dense 28 + softmax
  x = softmax(dot(x, W.dense_3[0], W.dense_3[1]));

  return x;
}

// ─── 랜드마크 정규화 ──────────────────────────────────────────────
function normalize(lms) {
  const wx = lms[0].x, wy = lms[0].y, wz = lms[0].z;
  let rel = lms.map(p => ({ x: p.x - wx, y: p.y - wy, z: p.z - wz }));
  const scale = Math.sqrt(rel[9].x**2 + rel[9].y**2 + rel[9].z**2) + 1e-8;
  rel = rel.map(p => ({ x: p.x/scale, y: p.y/scale, z: p.z/scale }));
  return rel.flatMap(p => [p.x, p.y, p.z]);
}

// ─── 분류 ─────────────────────────────────────────────────────────
function classify(lms) {
  if (!W || !LABELS.length) return '?';
  const probs = forward(normalize(lms));
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  return probs[maxIdx] >= 0.6 ? LABELS[maxIdx] : '?';
}

// ─── 캔버스 그리기 ────────────────────────────────────────────────
function draw(lms) {
  const C = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
             [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
             [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
  ctx.strokeStyle = '#00C2FF';
  ctx.lineWidth = 2;
  C.forEach(([a,b]) => {
    ctx.beginPath();
    ctx.moveTo(lms[a].x * canvas.width, lms[a].y * canvas.height);
    ctx.lineTo(lms[b].x * canvas.width, lms[b].y * canvas.height);
    ctx.stroke();
  });
  lms.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, i === 0 ? 6 : 4, 0, Math.PI*2);
    ctx.fillStyle = i === 0 ? '#FF6B6B' : '#00C2FF';
    ctx.fill();
  });
}

// ─── 텍스트 ───────────────────────────────────────────────────────
function append(sign) {
  const now = Date.now();
  if (sign === prevSign && now - prevTime < COOLDOWN) return;
  if (sign === '?') return;
  outText += sign;
  elText.textContent = outText;
  elText.classList.remove('placeholder');
  prevSign = sign; prevTime = now;
}

function reset() {
  outText = ''; prevSign = ''; stableCount = 0;
  elText.textContent = '인식된 글자가 여기에 표시됩니다';
  elText.classList.add('placeholder');
  elSign.textContent = '—';
  elBar.style.width = '0%';
}

function space() {
  outText += ' ';
  elText.textContent = outText;
}

// ─── MediaPipe ────────────────────────────────────────────────────
function initMP() {
  const hands = new Hands({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
  });
  hands.setOptions({
    maxNumHands: 1, modelComplexity: 1,
    minDetectionConfidence: 0.7, minTrackingConfidence: 0.5
  });

  hands.onResults(res => {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (res.multiHandLandmarks && res.multiHandLandmarks.length > 0) {
      const lms = res.multiHandLandmarks[0].map(p => ({ ...p, x: 1 - p.x }));
      draw(lms);

      const sign = classify(lms);
      elSign.textContent = sign !== '?' ? sign : '—';

      if (sign !== '?' && sign === prevSign) {
        stableCount++;
        elBar.style.width = `${(stableCount / STABLE) * 100}%`;
        if (stableCount >= STABLE) { append(sign); stableCount = 0; elBar.style.width = '0%'; }
      } else {
        stableCount = 1; prevSign = sign; elBar.style.width = '0%';
      }
      elStatus.textContent = '손 감지됨 — 지문자를 보여주세요';
      elStatus.classList.add('active');
    } else {
      elSign.textContent = '—'; stableCount = 0; elBar.style.width = '0%';
      elStatus.textContent = '손을 카메라에 보여주세요';
      elStatus.classList.remove('active');
    }
  });

  const cam = new Camera(video, {
    onFrame: async () => hands.send({ image: video }),
    width: 640, height: 480
  });
  cam.start();
}

// ─── 초기화 ───────────────────────────────────────────────────────
async function init() {
  // 웹캠
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    elStatus.textContent = '웹캠 오류: ' + e.message; return;
  }

  // 가중치 + 라벨 로드
  elStatus.textContent = '모델 로딩 중...';
  try {
    const [wRes, lRes] = await Promise.all([
      fetch('./model/weights.json'),
      fetch('./model/labels.json')
    ]);
    W = await wRes.json();
    LABELS = await lRes.json();
    elStatus.textContent = `✅ 모델 로드 완료 — ${LABELS.length}개 지문자`;
    elStatus.classList.add('active');
  } catch (e) {
    elStatus.textContent = '모델 로드 실패: ' + e.message; return;
  }

  initMP();

  document.getElementById('btn-reset').addEventListener('click', reset);
  document.getElementById('btn-space').addEventListener('click', space);
}

window.addEventListener('load', init);
