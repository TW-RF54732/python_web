/* =====================================================
   audio.js  ── 音訊系統
   
   請將音效檔放到 static/audio/ 資料夾：
     bgm.mp3          背景音樂（循環）
     correct.mp3      答對音效
     wrong.mp3        答錯音效
     timeout.mp3      超時音效
     tick.mp3         倒數警示音（最後5秒）
     start.mp3        遊戲開始音效
     finish.mp3       遊戲結束音效
   
   若沒有對應檔案，該音效會靜默略過（不報錯）
   ===================================================== */

const Audio = (() => {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();

  // ── 音量節點 ──────────────────────────────────────
  const masterGain = ctx.createGain();   // 主音量
  const bgmGain    = ctx.createGain();   // BGM 獨立音量
  const sfxGain    = ctx.createGain();   // 音效獨立音量
  masterGain.connect(ctx.destination);
  bgmGain.connect(masterGain);
  sfxGain.connect(masterGain);

  masterGain.gain.value = 0.8;
  bgmGain.gain.value    = 0.5;   // BGM 預設較低
  sfxGain.gain.value    = 1.0;

  // ── 緩衝快取 ──────────────────────────────────────
  const buffers = {};

  async function load(key, url) {
    try {
      const res  = await fetch(url);
      if (!res.ok) return;
      const data = await res.arrayBuffer();
      buffers[key] = await ctx.decodeAudioData(data);
    } catch (e) {
      console.warn(`[Audio] 無法載入 ${url}:`, e.message);
    }
  }

  // ── 預載所有音效 ──────────────────────────────────
  async function init() {
    await Promise.all([
      load('bgm',     '/static/audio/bgm.mp3'),
      load('correct', '/static/audio/correct.mp3'),
      load('wrong',   '/static/audio/wrong.mp3'),
      load('timeout', '/static/audio/timeout.mp3'),
      load('tick',    '/static/audio/tick.mp3'),
      load('start',   '/static/audio/start.mp3'),
      load('finish',  '/static/audio/finish.mp3'),
    ]);
    console.log('[Audio] 載入完成，已載入：', Object.keys(buffers).join(', '));
  }

  // ── 播放音效（單次） ──────────────────────────────
  function play(key, gainNode = sfxGain, rate = 1.0) {
    if (!buffers[key]) return null;
    // 瀏覽器要求 resume（首次互動後才能播音）
    if (ctx.state === 'suspended') ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer         = buffers[key];
    src.playbackRate.value = rate;
    src.connect(gainNode);
    src.start(0);
    return src;
  }

  // ── BGM（循環） ───────────────────────────────────
  let bgmSrc    = null;
  let bgmPlaying = false;

  function playBGM() {
    if (bgmPlaying || !buffers['bgm']) return;
    if (ctx.state === 'suspended') ctx.resume();
    bgmSrc = ctx.createBufferSource();
    bgmSrc.buffer = buffers['bgm'];
    bgmSrc.loop   = true;
    bgmSrc.connect(bgmGain);
    bgmSrc.start(0);
    bgmPlaying = true;
  }

  function stopBGM(fadeTime = 1.5) {
    if (!bgmSrc || !bgmPlaying) return;
    const now = ctx.currentTime;
    bgmGain.gain.setValueAtTime(bgmGain.gain.value, now);
    bgmGain.gain.linearRampToValueAtTime(0, now + fadeTime);
    bgmSrc.stop(now + fadeTime);
    bgmSrc = null;
    bgmPlaying = false;
    // 淡出後還原音量
    setTimeout(() => { bgmGain.gain.value = _bgmVol; }, (fadeTime + 0.1) * 1000);
  }

  function pauseBGM() {
    if (ctx.state === 'running') ctx.suspend();
  }

  function resumeBGM() {
    if (ctx.state === 'suspended') ctx.resume();
  }

  // ── 倒數 tick 節流（避免重疊） ───────────────────
  let lastTickTime = 0;

  function playTick() {
    const now = Date.now();
    if (now - lastTickTime < 900) return;  // 最多 1 次/秒
    lastTickTime = now;
    play('tick', sfxGain, 1.2);
  }

  // ── 情境音效 API ──────────────────────────────────
  function onStart()   { play('start');   playBGM(); }
  function onCorrect() { play('correct'); }
  function onWrong()   { play('wrong');   }
  function onTimeout() { play('timeout'); }
  function onTick()    { playTick();      }
  function onFinish()  { stopBGM(0.8); setTimeout(() => play('finish'), 400); }

  // ── 音量控制 ──────────────────────────────────────
  let _masterVol = 0.8;
  let _bgmVol    = 0.5;
  let _sfxVol    = 1.0;

  function setMaster(v) {
    _masterVol = v;
    masterGain.gain.value = v;
  }
  function setBGM(v) {
    _bgmVol = v;
    bgmGain.gain.value = v;
  }
  function setSFX(v) {
    _sfxVol = v;
    sfxGain.gain.value = v;
  }

  function getMaster() { return _masterVol; }
  function getBGM()    { return _bgmVol; }
  function getSFX()    { return _sfxVol; }

  return {
    init,
    playBGM, stopBGM, pauseBGM, resumeBGM,
    onStart, onCorrect, onWrong, onTimeout, onTick, onFinish,
    setMaster, setBGM, setSFX,
    getMaster, getBGM, getSFX,
  };
})();
