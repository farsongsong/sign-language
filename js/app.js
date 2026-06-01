/**
 * app.js
 * 수화 번역기 메인 로직
 * 웹캠 → MediaPipe Hands → TensorFlow.js 모델 → 텍스트 출력
 */

// ─── 전역 상태 ───────────────────────────────────────────────────
let recognizedText = '';
let lastSign = '';
let lastSignTime = 0;
let stableCount = 0;
let tfjsModel = null;
let labels = [];
const STABLE_THRESHOLD = 20;
const SIGN_COOLDOWN = 1200;

// ─── DOM 요소 ────────────────────────────────────────────────────
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('canvas');
const ctx = canvasEl.getContext('2d');
const currentSignEl = document.getElementById('current-sign');
const recognizedTextEl = document.getElementById('recognized-text');
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('progress-bar');

/**
 * TensorFlow.js 모델 로드
 */
async function loadModel() {
  statusEl.textContent = '모델 로딩 중...';
  try {
    // 라벨 로드
    const labelsRes = await fetch('./model/labels.json');
    labels = await labelsRes.json();

    // 모델 로드
    tfjsModel = await tf.loadLayersModel('./model/model.json');
    statusEl.textContent = `모델 로드 완료 (${labels.length}개 지문자)`;
    statusEl.classList.add('active');
    console.log('✅ 모델 로드 완료, 라벨:', labels);
  } catch (e) {
    statusEl.textContent = '모델 로드 실패: ' + e.message;
    console.error(e);
  }
}

/**
 * 랜드마크 정규화 (학습 시와 동일한 방법)
 */
function normalizeLandmarks(landmarks) {
  // 손목 기준 상대 좌표
  const wrist = landmarks[0];
  let lm = landmarks.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  // 중지 MCP(9번)까지 거리로 스케일 정규화
  const scale = Math.sqrt(lm[9].x**2 + lm[9].y**2 + lm[9].z**2) + 1e-8;
  lm = lm.map(p => ({ x: p.x/scale, y: p.y/scale, z: p.z/scale }));

  // 63차원 벡터로 평탄화
  return lm.flatMap(p => [p.x, p.y, p.z]);
}

/**
 * TF.js 모델로 지문자 분류
 */
async function classifyWithModel(landmarks) {
  if (!tfjsModel || labels.length === 0) return '?';
  const input = normalizeLandmarks(landmarks);
  const tensor = tf.tensor2d([input]);
  const prediction = tfjsModel.predict(tensor);
  const probs = await prediction.data();
  tensor.dispose();
  prediction.dispose();

  const maxIdx = probs.indexOf(Math.max(...probs));
  const confidence = probs[maxIdx];

  // 신뢰도 60% 미만이면 불확실
  if (confidence < 0.6) return '?';
  return labels[maxIdx];
}

/**
 * 캔버스에 랜드마크 그리기
 */
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

/**
 * 텍스트 추가
 */
function addToText(sign) {
  const now = Date.now();
  if (sign === lastSign && now - lastSignTime < SIGN_COOLDOWN) return;
  if (sign === '?') return;
  recognizedText += sign;
  recognizedTextEl.textContent = recognizedText || '인식된 글자가 여기에 표시됩니다';
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

/**
 * MediaPipe 초기화
 */
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

  hands.onResults(async (results) => {
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

      const sign = await classifyWithModel(mirrored);
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
        progressEl.style.width = '0%';
        lastSign = sign;
      }

      statusEl.textContent = `손 감지됨 — 지문자를 보여주세요`;
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

/**
 * 앱 초기화
 */
async function init() {
  // 웹캠 시작
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    videoEl.srcObject = stream;
  } catch (e) {
    statusEl.textContent = '웹캠 오류: ' + e.message;
  }

  // 모델 로드
  await loadModel();

  // MediaPipe 초기화
  initMediaPipe();

  // 버튼 이벤트
  document.getElementById('btn-reset').addEventListener('click', resetText);
  document.getElementById('btn-space').addEventListener('click', addSpace);
}

window.addEventListener('load', init);
