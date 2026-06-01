/**
 * app.js — 수화 번역기 (한글 자모 조합 포함)
 */

// ─── 상태 ─────────────────────────────────────────────────────────
let W = null;
let LABELS = [];
let prevSign = '';
let stableCount = 0;
let justCommitted = false;
const STABLE = 35;

// ─── 한글 조합 ────────────────────────────────────────────────────
// 초성 19개
const CHOSEONG  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'];
// 중성 21개
const JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'];
// 종성 28개 (0번은 없음)
const JONGSEONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'];

// 종성 → 다음 초성으로 분리될 때 쓰는 매핑
const JONGSEONG_TO_CHOSEONG = {
  'ㄱ':'ㄱ','ㄲ':'ㄲ','ㄴ':'ㄴ','ㄷ':'ㄷ','ㄹ':'ㄹ',
  'ㅁ':'ㅁ','ㅂ':'ㅂ','ㅅ':'ㅅ','ㅆ':'ㅆ','ㅇ':'ㅇ',
  'ㅈ':'ㅈ','ㅊ':'ㅊ','ㅋ':'ㅋ','ㅌ':'ㅌ','ㅍ':'ㅍ','ㅎ':'ㅎ'
};

// 복합 종성 분리
const DOUBLE_JONGSEONG = {
  'ㄳ':['ㄱ','ㅅ'],'ㄵ':['ㄴ','ㅈ'],'ㄶ':['ㄴ','ㅎ'],
  'ㄺ':['ㄹ','ㄱ'],'ㄻ':['ㄹ','ㅁ'],'ㄼ':['ㄹ','ㅂ'],
  'ㄽ':['ㄹ','ㅅ'],'ㄾ':['ㄹ','ㅌ'],'ㄿ':['ㄹ','ㅍ'],
  'ㅀ':['ㄹ','ㅎ'],'ㅄ':['ㅂ','ㅅ']
};

// 두 자음이 합쳐지는 복합 종성
const COMBINE_JONGSEONG = {
  'ㄱ+ㅅ':'ㄳ','ㄴ+ㅈ':'ㄵ','ㄴ+ㅎ':'ㄶ',
  'ㄹ+ㄱ':'ㄺ','ㄹ+ㅁ':'ㄻ','ㄹ+ㅂ':'ㄼ',
  'ㄹ+ㅅ':'ㄽ','ㄹ+ㅌ':'ㄾ','ㄹ+ㅍ':'ㄿ',
  'ㄹ+ㅎ':'ㅀ','ㅂ+ㅅ':'ㅄ'
};

// 모음 복합 조합
const COMBINE_VOWEL = {
  'ㅗ+ㅏ':'ㅘ','ㅗ+ㅐ':'ㅙ','ㅗ+ㅣ':'ㅚ',
  'ㅜ+ㅓ':'ㅝ','ㅜ+ㅔ':'ㅞ','ㅜ+ㅣ':'ㅟ',
  'ㅡ+ㅣ':'ㅢ'
};

function isVowel(c) { return JUNGSEONG.includes(c); }
function isCons(c)  { return CHOSEONG.includes(c) && !isVowel(c); }

// 자모 배열 → 완성형 문자열로 조합
function buildText(jamos) {
  let result = '';
  let i = 0;
  while (i < jamos.length) {
    const c = jamos[i];
    if (isVowel(c)) {
      result += c; i++; continue;
    }
    // 자음
    const cho = CHOSEONG.indexOf(c);
    if (cho === -1) { result += c; i++; continue; }

    // 다음이 모음?
    if (i + 1 < jamos.length && isVowel(jamos[i+1])) {
      let jung = jamos[i+1];
      // 복합 모음 체크
      if (i + 2 < jamos.length && isVowel(jamos[i+2])) {
        const combo = COMBINE_VOWEL[jung + '+' + jamos[i+2]];
        if (combo) { jung = combo; i++; }
      }
      const jungIdx = JUNGSEONG.indexOf(jung);

      // 다다음이 자음이고 그 다음이 모음이 아니면 → 종성
      let jongIdx = 0;
      let skip = 0;
      if (i + 2 < jamos.length && isCons(jamos[i+2])) {
        const nextCons = jamos[i+2];
        if (i + 3 < jamos.length && isVowel(jamos[i+3])) {
          // 종성 없음, 다음 글자의 초성으로
          jongIdx = 0;
        } else if (i + 3 < jamos.length && isCons(jamos[i+3])) {
          // 복합 종성 가능성
          const combo = COMBINE_JONGSEONG[nextCons + '+' + jamos[i+3]];
          if (combo && i + 4 < jamos.length && isVowel(jamos[i+4])) {
            jongIdx = 0;
          } else if (combo) {
            jongIdx = JONGSEONG.indexOf(combo);
            skip = 2;
          } else {
            jongIdx = JONGSEONG.indexOf(nextCons);
            if (jongIdx > 0) skip = 1;
          }
        } else {
          jongIdx = JONGSEONG.indexOf(nextCons);
          if (jongIdx > 0) skip = 1;
        }
      }

      const code = 0xAC00 + cho * 21 * 28 + jungIdx * 28 + jongIdx;
      result += String.fromCharCode(code);
      i += 2 + skip;
    } else {
      result += c; i++;
    }
  }
  return result;
}

// ─── 입력 버퍼 ────────────────────────────────────────────────────
let jamos = []; // 입력된 자모 배열

function appendJamo(sign) {
  jamos.push(sign);
  updateDisplay();
}

function updateDisplay() {
  const built = buildText(jamos);
  elText.textContent = built || '인식된 글자가 여기에 표시됩니다';
  if (built) elText.classList.remove('placeholder');
  else elText.classList.add('placeholder');
}

function reset() {
  jamos = []; prevSign = ''; stableCount = 0;
  elText.textContent = '인식된 글자가 여기에 표시됩니다';
  elText.classList.add('placeholder');
  elSign.textContent = '—';
  elBar.style.width = '0%';
}

function space() {
  jamos.push(' ');
  updateDisplay();
}

function backspace() {
  jamos.pop();
  updateDisplay();
}

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
  const s = e.reduce((a,b) => a+b, 0);
  return e.map(v => v/s);
}
function batchNorm(x, gamma, beta, mean, variance) {
  return x.map((v,i) => gamma[i]*(v-mean[i])/Math.sqrt(variance[i]+0.001)+beta[i]);
}
function forward(input) {
  let x = new Float32Array(input);
  x = relu(dot(x, W.dense[0], W.dense[1]));
  x = batchNorm(x, W.batch_normalization[0], W.batch_normalization[1],
                   W.batch_normalization[2], W.batch_normalization[3]);
  x = relu(dot(x, W.dense_1[0], W.dense_1[1]));
  x = batchNorm(x, W.batch_normalization_1[0], W.batch_normalization_1[1],
                   W.batch_normalization_1[2], W.batch_normalization_1[3]);
  x = relu(dot(x, W.dense_2[0], W.dense_2[1]));
  x = softmax(dot(x, W.dense_3[0], W.dense_3[1]));
  return x;
}

function normalize(lms) {
  const wx=lms[0].x, wy=lms[0].y, wz=lms[0].z;
  let rel = lms.map(p => ({x:p.x-wx, y:p.y-wy, z:p.z-wz}));
  const scale = Math.sqrt(rel[9].x**2+rel[9].y**2+rel[9].z**2)+1e-8;
  rel = rel.map(p => ({x:p.x/scale, y:p.y/scale, z:p.z/scale}));
  return rel.flatMap(p => [p.x,p.y,p.z]);
}

function classify(lms) {
  if (!W || !LABELS.length) return '?';
  const probs = forward(normalize(lms));
  let maxIdx = 0;
  for (let i=1; i<probs.length; i++) if (probs[i]>probs[maxIdx]) maxIdx=i;
  return probs[maxIdx] >= 0.65 ? LABELS[maxIdx] : '?';
}

function draw(lms) {
  const C=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
           [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
           [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
  ctx.strokeStyle='#4F6EF7'; ctx.lineWidth=2;
  C.forEach(([a,b])=>{
    ctx.beginPath();
    ctx.moveTo(lms[a].x*canvas.width, lms[a].y*canvas.height);
    ctx.lineTo(lms[b].x*canvas.width, lms[b].y*canvas.height);
    ctx.stroke();
  });
  lms.forEach((p,i)=>{
    ctx.beginPath();
    ctx.arc(p.x*canvas.width, p.y*canvas.height, i===0?7:4, 0, Math.PI*2);
    ctx.fillStyle = i===0?'#00BFA5':'#4F6EF7';
    ctx.fill();
  });
}

// ─── MediaPipe ────────────────────────────────────────────────────
function initMP() {
  const hands = new Hands({
    locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
  });
  hands.setOptions({
    maxNumHands:1, modelComplexity:1,
    minDetectionConfidence:0.7, minTrackingConfidence:0.5
  });

  hands.onResults(res => {
    ctx.save();
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.translate(canvas.width,0); ctx.scale(-1,1);
    ctx.drawImage(res.image,0,0,canvas.width,canvas.height);
    ctx.restore();

    if (res.multiHandLandmarks && res.multiHandLandmarks.length > 0) {
      const lms = res.multiHandLandmarks[0].map(p=>({...p, x:1-p.x}));
      draw(lms);

      const sign = classify(lms);
      elSign.textContent = sign !== '?' ? sign : '—';

      if (justCommitted) {
        elBar.style.width = '0%';
      } else if (sign !== '?') {
        if (sign === prevSign) {
          stableCount++;
          elBar.style.width = `${(stableCount/STABLE)*100}%`;
          if (stableCount >= STABLE) {
            appendJamo(sign);
            stableCount = 0; prevSign = '';
            justCommitted = true;
            setTimeout(()=>{ justCommitted=false; }, 300);
            elBar.style.width = '0%';
          }
        } else {
          stableCount = 1; prevSign = sign; elBar.style.width = '0%';
        }
      } else {
        stableCount = 0; prevSign = ''; elBar.style.width = '0%';
      }

      elStatus.textContent = '손 감지됨 — 지문자를 보여주세요';
      elStatus.classList.add('active');
    } else {
      elSign.textContent = '—'; stableCount=0; prevSign='';
      elBar.style.width='0%';
      elStatus.textContent='손을 카메라에 보여주세요';
      elStatus.classList.remove('active');
    }
  });

  const cam = new Camera(video,{
    onFrame: async()=>hands.send({image:video}),
    width:640, height:480
  });
  cam.start();
}

// ─── 초기화 ───────────────────────────────────────────────────────
async function init() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  } catch(e) { elStatus.textContent='웹캠 오류: '+e.message; return; }

  elStatus.textContent='모델 로딩 중...';
  try {
    const [wRes,lRes] = await Promise.all([
      fetch('./model/weights.json'),
      fetch('./model/labels.json')
    ]);
    W = await wRes.json();
    LABELS = await lRes.json();
    elStatus.textContent=`✅ 모델 로드 완료 — ${LABELS.length}개 지문자`;
    elStatus.classList.add('active');
  } catch(e) { elStatus.textContent='모델 로드 실패: '+e.message; return; }

  initMP();
  document.getElementById('btn-reset').addEventListener('click', reset);
  document.getElementById('btn-space').addEventListener('click', space);
  document.getElementById('btn-backspace').addEventListener('click', backspace);
}

window.addEventListener('load', init);
