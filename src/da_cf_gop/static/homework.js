/* Recording, WAV encoding and audio preview adapted from code1/webapp.
   Results are reference positions, not estimated audio timestamps. */
"use strict";
const $ = (id) => document.getElementById(id);
let selectedFile = null, previewUrl = null, busy = false, recording = false;
let audioCtx = null, mediaStream = null, source = null, processor = null;
let pcmChunks = [], recordingTimer = null, recordingStarted = 0;
const MAX_RECORD_SECONDS = 20;

function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function updateControls() {
  $("reference").disabled = busy || recording;
  $("audio-file").disabled = busy || recording;
  $("feedback-mode").disabled = busy || recording;
  $("record").disabled = busy;
  $("assess").disabled = busy || recording || !selectedFile || !$("reference").value.trim();
  $("retry").disabled = busy || recording;
  $("record").textContent = recording ? "Stop recording" : "Start recording";
  $("record").classList.toggle("recording", recording);
}
function clearResult() {
  $("results").hidden = true;
  $("words").replaceChildren();
  $("feedback").textContent = "";
  $("details").textContent = "";
  error("");
}
function previewFile(file) {
  clearResult();
  selectedFile = file;
  $("player").pause();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  if (previewUrl) $("player").src = previewUrl;
  else $("player").removeAttribute("src");
  $("preview").hidden = !file;
  $("filename").textContent = file ? file.name : "";
  $("status").textContent = file ? "Ready. Replay your recording or select Review recording." : "Choose or record audio.";
  updateControls();
}
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}
async function releaseRecorder() {
  clearInterval(recordingTimer);
  if (processor) { processor.onaudioprocess = null; processor.disconnect(); }
  if (source) source.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioCtx && audioCtx.state !== "closed") await audioCtx.close();
  processor = source = mediaStream = audioCtx = null;
}
async function startRecording() {
  if (busy || recording) return;
  if (!$("reference").value.trim()) { error("Enter your assigned text first."); $("reference").focus(); return; }
  clearResult(); busy = true; updateControls();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone recording requires localhost or HTTPS.");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.resume();
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    pcmChunks = [];
    processor.onaudioprocess = (event) => {
      pcmChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      event.outputBuffer.getChannelData(0).fill(0);
    };
    source.connect(processor); processor.connect(audioCtx.destination);
    previewFile(null);
    $("audio-file").value = "";
    recording = true; recordingStarted = Date.now();
    $("status").textContent = "Recording… press Stop when you finish.";
    recordingTimer = setInterval(() => {
      const seconds = (Date.now() - recordingStarted) / 1000;
      $("status").textContent = `Recording… ${seconds.toFixed(0)} s (maximum ${MAX_RECORD_SECONDS} s)`;
      if (seconds >= MAX_RECORD_SECONDS) stopRecording();
    }, 250);
  } catch (exception) {
    await releaseRecorder();
    error("Could not start recording: " + exception.message);
  } finally { busy = false; updateControls(); }
}
async function stopRecording() {
  if (!recording) return;
  recording = false; busy = true; updateControls();
  const sampleRate = audioCtx.sampleRate;
  try {
    await releaseRecorder();
    const count = Math.min(pcmChunks.reduce((total, chunk) => total + chunk.length, 0),
      Math.floor(sampleRate * MAX_RECORD_SECONDS));
    if (!count) throw new Error("The recording contains no audio. Please try again.");
    const samples = new Float32Array(count);
    let offset = 0;
    for (const chunk of pcmChunks) {
      const part = chunk.subarray(0, count - offset);
      samples.set(part, offset); offset += part.length;
      if (offset === count) break;
    }
    // Preserve the actual hardware context rate. The server resamples to 16 kHz.
    previewFile(new File([encodeWAV(samples, sampleRate)], "recording.wav", { type: "audio/wav" }));
  } catch (exception) { error(exception.message); }
  finally { pcmChunks = []; busy = false; updateControls(); }
}
function addText(parent, tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  parent.appendChild(element);
  return element;
}
function renderResult(result) {
  $("results").hidden = false;
  const summary = result.summary;
  $("result-summary").textContent = `${summary.review_phones} of ${summary.assessed_phones} assessed sounds flagged for another listen. ${summary.unassessed_phones} sounds not assessed.`;
  const phones = new Map(result.phones.map((phone) => [phone.phone_index, phone]));
  const labels = { review: "Worth another listen", no_flag: "No flag", unassessed: "Not assessed" };
  for (const word of result.words) {
    const card = addText($("words"), "div", "", `word-card ${word.status}`);
    addText(card, "div", word.text, "word-title");
    addText(card, "span", labels[word.status] || "Partly assessed", "word-state");
    const row = addText(card, "div", "", "phone-row");
    for (const index of word.phones) {
      const phone = phones.get(index);
      const chip = addText(row, "span", phone.ipa || phone.canonical, `phone ${phone.status}`);
      const target = `Target /${phone.ipa || phone.canonical}/: ${labels[phone.status]}`;
      chip.title = target;
      chip.setAttribute("aria-label", target);
    }
  }
  $("feedback").textContent = result.feedback.text;
  const fallback = result.feedback.mode_requested === "qwen" && result.feedback.mode_used !== "qwen";
  $("feedback-note").hidden = !fallback;
  $("feedback-note").textContent = fallback ? "Standard feedback is shown because the optional language model was unavailable or did not return a supported response." : "";
  $("details").textContent = JSON.stringify({ provenance: result.provenance, timings: result.timings,
    feedback_mode: result.feedback.mode_used,
    note: "Scores concern target reference positions. Specific realized-phone diagnoses are withheld. No audio timestamps are estimated.",
    phones: result.phones }, null, 2);
}
async function assessRecording() {
  if (busy || recording || !selectedFile) return;
  const reference = $("reference").value.trim();
  if (!reference) { error("Enter your assigned text first."); return; }
  clearResult(); busy = true; updateControls();
  $("status").textContent = "Assessing your recording… The first assessment also loads the models.";
  const form = new FormData();
  form.append("audio", selectedFile, selectedFile.name);
  form.append("reference_text", reference);
  form.append("feedback_mode", $("feedback-mode").value);
  try {
    const response = await fetch("/api/assess", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) {
      const detail = result.detail;
      throw new Error(typeof detail === "string" ? detail : "Please check your text and audio file, then try again.");
    }
    renderResult(result);
    $("status").textContent = "Review ready. Listen again and practise the assigned text.";
  } catch (exception) { error("Assessment could not finish: " + exception.message); $("status").textContent = "Assessment not completed."; }
  finally { busy = false; updateControls(); }
}
$("record").addEventListener("click", () => recording ? stopRecording() : startRecording());
$("assess").addEventListener("click", assessRecording);
$("retry").addEventListener("click", startRecording);
$("audio-file").addEventListener("change", () => previewFile($("audio-file").files[0] || null));
$("reference").addEventListener("input", () => { clearResult(); updateControls(); $("status").textContent = "Text changed. Select Review recording to assess against this text."; });
$("feedback-mode").addEventListener("change", () => { clearResult(); $("status").textContent = "Select Review recording to use this feedback option."; });
$("replay").addEventListener("click", async () => {
  $("player").currentTime = 0;
  try { await $("player").play(); } catch (exception) { error("Playback unavailable: " + exception.message); }
});
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $("theme").textContent = dark ? "Light theme" : "Dark theme";
});
window.addEventListener("pagehide", () => {
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
updateControls();
