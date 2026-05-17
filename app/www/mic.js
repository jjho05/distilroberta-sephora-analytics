// ── Variables globales ────────────────────────────────────────────────────────
let _micBtn = null;
let isRecording = false;
let isTranscribing = false;
let isStarting = false;
let mediaRecorder = null;
let audioChunks = [];
let silenceTimer = null;
let streamRef = null;
let streamSource = null;
let processor = null;
let globalAudioCtx = null;
let handlerDone = false;
let stopped = false;

const MIC_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`;
const STOP_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/></svg>`;
const LOAD_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>`;

const setIcon = (icon) => { if (_micBtn) _micBtn.innerHTML = icon; };
const resetBtn = () => {
    if (!_micBtn) return;
    _micBtn.classList.remove('recording', 'transcribing', 'mic-active');
    setIcon(MIC_ICON);
};

const getCtx = () => {
    if (!globalAudioCtx || globalAudioCtx.state === 'closed')
        globalAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return globalAudioCtx;
};

const cleanup = () => {
    stopped = true;
    if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
    if (processor) { try { processor.disconnect(); } catch (e) { } processor = null; }
    if (streamSource) { try { streamSource.disconnect(); } catch (e) { } streamSource = null; }
    if (streamRef) { streamRef.getTracks().forEach(t => t.stop()); streamRef = null; }
    mediaRecorder = null;
    audioChunks = [];
    isRecording = false;
    isTranscribing = false;
    isStarting = false;
};

const stopRecording = () => {
    if (stopped) return;
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    } else {
        cleanup();
        resetBtn();
    }
};

const startRecording = async () => {
    if (isStarting) return;
    try {
        isStarting = true;
        cleanup();
        stopped = false;
        resetBtn();

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef = stream;

        // ── Detección de silencio con ScriptProcessor ─────────────────────
        const ctx = getCtx();
        if (ctx.state === 'suspended') await ctx.resume();

        streamSource = ctx.createMediaStreamSource(stream);
        processor = ctx.createScriptProcessor(2048, 1, 1);
        streamSource.connect(processor);
        processor.connect(ctx.destination);

        // Esperar 1s antes de detectar silencio (mic necesita calentarse)
        let detectionActive = false;
        setTimeout(() => { detectionActive = true; }, 1000);

        processor.onaudioprocess = (e) => {
            if (stopped || !detectionActive) return;
            const input = e.inputBuffer.getChannelData(0);
            const avg = input.reduce((a, b) => a + Math.abs(b), 0) / input.length;

            if (avg > 0.01) {
                // Hay voz — cancelar timer de silencio
                if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
            } else {
                // Silencio — iniciar timer para cortar
                if (!silenceTimer) {
                    silenceTimer = setTimeout(() => stopRecording(), 1500);
                }
            }
        };

        // ── MediaRecorder ─────────────────────────────────────────────────
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data?.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stopped = true;
            if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
            if (processor) { try { processor.disconnect(); } catch (e) { } processor = null; }
            if (streamSource) { try { streamSource.disconnect(); } catch (e) { } streamSource = null; }
            if (streamRef) { streamRef.getTracks().forEach(t => t.stop()); streamRef = null; }
            isRecording = false;

            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];

            if (blob.size < 100) { resetBtn(); return; }

            isTranscribing = true;
            _micBtn?.classList.remove('recording', 'mic-active');
            _micBtn?.classList.add('transcribing');
            setIcon(LOAD_ICON);

            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = () => {
                if (window.Shiny)
                    Shiny.setInputValue('voice_audio_payload', reader.result, { priority: 'event' });
            };
        };

        mediaRecorder.start();
        isRecording = true;
        isStarting = false;
        _micBtn?.classList.add('recording', 'mic-active');
        setIcon(STOP_ICON);

        // Seguro máximo: cortar a los 30 segundos
        setTimeout(() => { if (isRecording) stopRecording(); }, 30000);

    } catch (err) {
        console.error('[Mic] Error:', err);
        cleanup();
        resetBtn();
    }
};

const onDone = () => {
    isTranscribing = false;
    isStarting = false;
    isRecording = false;
    stopped = false;
    mediaRecorder = null;
    streamRef = null;
    streamSource = null;
    processor = null;
    resetBtn();
};

const registerHandler = () => {
    if (handlerDone) return;
    if (window.Shiny?.addCustomMessageHandler) {
        try { Shiny.addCustomMessageHandler('transcription_done', () => onDone()); } catch (e) { }
        handlerDone = true;
    } else {
        setTimeout(registerHandler, 200);
    }
};

function initMic() {
    const btn = document.getElementById('floating-mic-btn');
    if (!btn || btn._micInit) return;
    btn._micInit = true;
    _micBtn = btn;

    btn.addEventListener('click', () => {
        if (isTranscribing || isStarting) return;
        isRecording ? stopRecording() : startRecording();
    });

    registerHandler();

    if (window.$) {
        $(document).on('shiny:idle', () => { if (isTranscribing) onDone(); });
    }

    console.log('[Olvera] Mic listo ✅');
}

function waitAndInit() {
    if (document.body) {
        initMic();
        const obs = new MutationObserver(() => initMic());
        obs.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => obs.disconnect(), 15000);
    } else {
        setTimeout(waitAndInit, 50);
    }
}

waitAndInit();