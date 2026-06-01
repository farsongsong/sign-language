/**
 * app.js
 * 수화 번역기 — 가중치 직접 주입 방식 (TF.js 모델 로드 없이)
 * 웹캠 → MediaPipe Hands → 수동 신경망 추론 → 텍스트 출력
 */

// ─── 전역 상태 ────────────────────────────────────────────────────
let weights = null;
let labels = [];
let recognizedText = '';
let lastSign = '';
let lastSignTime = 0;
let stableCount = 0;
const STABLE_THRESHOLD = 20;
const SIGN_COOLDOWN = 1200;

// ─── DOM ─────────────────────────────────────────────────────────
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('canvas');
const ctx = canvasEl.getContext('2d');
const currentSignEl = document.getElementById('current-sign');
const recognizedTextEl = document.getElementById('recognized-text');
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('progress-bar');

// ─── 수동 신경망 추론 ─────────────────────────────────────────────

/** 행렬 곱셈: (1 x inDim) x (inDim x outDim) → (1 x outDim) */
function matMul(input, kernel) {
  const out = new Float32Array(kernel[0].length);
  for (let j = 0; j < kernel[0].length; j++) {
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * kernel[i][j];
    out[j] = sum;
  }
  return out;
}

/** bias 더하기 */
function addBias(x, bias) {
  return x.map((v, i) => v + bias[i]);
}

/** ReLU */
function relu(x) {
  return x.map(v => Math.max(0, v));
}

/** Softmax */
function softmax(x) {
  const max = Math.max(...x);
  const exp = x.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

/** BatchNorm 추론 (gamma, beta, mean, variance) */
function batchNorm(x, gamma, beta, mean, variance, epsilon = 0.001) {
  return x.map((v, i) => {
    const normalized = (v - mean[i]) / Math.sqrt(variance[i] + epsilon);
    return gamma[i] * normalized + beta[i];
  });
}

/** Dense 레이어 */
function denseLayer(x, layerName, activation) {
  const w = weights[layerName];
  let out = matMul(x, w[0]);
  out = addBias(out, w[1]);
  if (activation === 'relu') return relu(out);
  if (activation === 'softmax') return softmax(out);
  return out;
}

/** BatchNorm 레이어 */
function bnLayer(x, layerName) {
  const w = weights[layerName];
  // [gamma, beta, moving_mean, moving_variance]
  return batchNorm(x, w[0], w[1], w[2], w[3]);
}

/**
 * 전체 모델 추론
 * Dense(256,relu) → BN → Dense(128,relu) → BN → Dense(64,relu) → Dense(28,softmax)
 */
function predict(input) {
  let x = new Float32Array(input);
  x = denseLayer(x, 'dense', 'relu');
  x = bnLayer(x, 'batch_normalization');
  x = denseLayer(x, 'dense_1', 'relu');
  x = bnLayer(x, 'batch_normalization_1');
  x = denseLayer(x, 'dense_2', 'relu');
  x = denseLayer(x, 'dense_3', 'softmax');
  return x;
}

// ─── 모델 로드 ────────────────────────────────────────────────────
async function loadModel() {
  statusEl.textContent = '모델 로딩 중...';
  try {
    const [wRes, lRes] = await Promise.all([
      fetch('./model/weights.json'),
      fetch('./model/labels.json'),
    ]);
    weights = await wRes.json();
    labels = await lRes.json();
    statusEl.textContent = `모델 로드 완료 — ${labels.length}개 지문자 인식 준비됨`;
    statusEl.classList.add('active');
  } catch (e) {
    statusEl.textContent = '모델 로드 실패: ' + e.message;
    console.error(e);
  }
}

// ─── 랜드마크 정규화 ──────────────────────────────────────────────
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  let lm = landmarks.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  const scale = Math.sqrt(lm[9].x**2 + lm[9].y**2 + lm[9].z**2) + 1e-8;
  lm = lm.map(p => ({ x: p.x/scale, y: p.y/scale, z: p.z/scale }));
  return lm.flatMap(p => [p.x, p.y, p.z]);
}

// ─── 분류 ────────────────────────────────────────────────────────
function classifySign(landmarks) {
  if (!weights || labels.length === 0) return '?';
  const input = normalizeLandmarks(landmarks);
  const probs = predict(input);
  const maxIdx = probs.indexOf(Math.max(...probs));
  const confidence = probs[maxIdx];
  if (confidence < 0.6) return '?';
  return labels[maxIdx];
}

// ─── 캔버스 랜드마크 그리기 ───────────────────────────────────────
function drawLandmarks(landmarks) {
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];
  ctx.strokeStyle = '#00C2FF';
  ctx.lineWidth = 2;
  connections.forEach(([a, b]) => {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * canvasEl.width, landmarks[a].y * canvasEl.height);
    ctx.lineTo(landmarks[b].x * canvasEl.width, landmarks[b].y * canvasEl.height);
    ctx.stroke();
  });
  landmarks.forEach((lm, i) => {
    ctx.beginPath();
    ctx.arc(lm.x * canvasEl.width, lm.y * canvasEl.height, i === 0 ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? '#FF6B6B' : '#00C2FF';
    ctx.fill();
  });
}

// ─── 텍스트 처리 ─────────────────────────────────────────────────
function addToText(sign) {
  const now = Date.now();
  if (sign === lastSign && now - lastSignTime < SIGN_COOLDOWN) return;
  if (sign === '?') return;
  recognizedText += sign;
  recognizedTextEl.textContent = recognizedText;
  recognizedTextEl.classList.remove('placeholder');
  lastSign = sign;
  lastSignTime = now;
}

function resetText() {
  recognizedText = '';
  lastSign = '';
  stableCount = 0;
  recognizedTextEl.textContent = '인식된 글자가 여기에 표시됩니다';
  recognizedTextEl.classList.add('placeholder');
  currentSignEl.textContent = '—';
  progressEl.style.width = '0%';
}

function addSpace() {
  recognizedText += ' ';
  recognizedTextEl.textContent = recognizedText;
}

// ─── MediaPipe 초기화 ─────────────────────────────────────────────
function initMediaPipe() {
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    ctx.save();
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.translate(canvasEl.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
    ctx.restore();

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const mirrored = landmarks.map(lm => ({ ...lm, x: 1 - lm.x }));
      drawLandmarks(mirrored);

      const sign = classifySign(mirrored);
      currentSignEl.textContent = sign !== '?' ? sign : '—';

      if (sign !== '?' && sign === lastSign) {
        stableCount++;
        progressEl.style.width = `${(stableCount / STABLE_THRESHOLD) * 100}%`;
        if (stableCount >= STABLE_THRESHOLD) {
          addToText(sign);
          stableCount = 0;
          progressEl.style.width = '0%';
        }
      } else {
        stableCount = 1;
        lastSign = sign;
        progressEl.style.width = '0%';
      }

      statusEl.textContent = '손 감지됨 — 지문자를 보여주세요';
      statusEl.classList.add('active');
    } else {
      currentSignEl.textContent = '—';
      stableCount = 0;
      progressEl.style.width = '0%';
      statusEl.textContent = '손을 카메라에 보여주세요';
    }
  });

  const camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width: 640, height: 480,
  });
  camera.start();
}

// ─── 앱 시작 ─────────────────────────────────────────────────────
async function init() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    videoEl.srcObject = stream;
  } catch (e) {
    statusEl.textContent = '웹캠 오류: ' + e.message;
  }

  await loadModel();
  initMediaPipe();

  document.getElementById('btn-reset').addEventListener('click', resetText);
  document.getElementById('btn-space').addEventListener('click', addSpace);
}

window.addEventListener('load', init);
