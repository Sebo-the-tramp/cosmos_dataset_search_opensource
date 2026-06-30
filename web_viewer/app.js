const API_HOST = ["127.0.0.1", "localhost"].includes(window.location.hostname) ? window.location.hostname : "127.0.0.1";
const API_ORIGIN = window.location.port === "5000" ? "" : `http://${API_HOST}:5000`;
const API_URL = `${API_ORIGIN}/search`;
const SEARCH_VIDEO_URL = `${API_ORIGIN}/search_video`;
const DOWNLOAD_URL = `${API_ORIGIN}/download`;
const HISTORY_URL = `${API_ORIGIN}/history`;
const FORM = document.querySelector("#search-form");
const VIDEO_FORM = document.querySelector("#video-search-form");
const QUERY = document.querySelector("#query");
const TOP_K = document.querySelector("#top-k");
const VIDEO_QUERY = document.querySelector("#video-query");
const SUMMARY = document.querySelector("#summary");
const RESULTS = document.querySelector("#results");
const TEMPLATE = document.querySelector("#result-card");
const DOWNLOAD_RESULTS = document.querySelector("#download-results");
const DOWNLOAD_PROGRESS = document.querySelector("#download-progress");
const DOWNLOAD_PROGRESS_BAR = document.querySelector("#download-progress-bar");
const DOWNLOAD_PROGRESS_TEXT = document.querySelector("#download-progress-text");
const HISTORY = document.querySelector("#history");
let LAST_PAYLOAD = null;

function assert(condition) {
  if (!condition) {
    throw new Error("Invalid input");
  }
}

function buildSearchUrl(word, quantity) {
  const url = new URL(API_URL);
  url.searchParams.set("word", word);
  url.searchParams.set("quantity", String(quantity));
  return url;
}

function parsePayload(text) {
  return JSON.parse(text.replace(/([:[,]\s*)(-?\d{16,})(?=[,\]}])/g, '$1"$2"'));
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  assert(response.ok);
  return parsePayload(await response.text());
}

async function search(word, quantity) {
  return requestJson(buildSearchUrl(word, quantity));
}

async function searchVideo(file, quantity) {
  const data = new FormData();
  data.append("video", file);
  data.append("quantity", String(quantity));
  return requestJson(SEARCH_VIDEO_URL, { method: "POST", body: data });
}

async function loadHistory() {
  renderHistory(await requestJson(HISTORY_URL));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hideProgress() {
  DOWNLOAD_PROGRESS.hidden = true;
  DOWNLOAD_PROGRESS_BAR.style.width = "0%";
  DOWNLOAD_PROGRESS_TEXT.textContent = "";
}

function clearResults() {
  RESULTS.querySelectorAll("video").forEach((video) => {
    video.pause();
    video.removeAttribute("src");
    video.load();
  });
  RESULTS.replaceChildren();
}

function renderProgress(job) {
  const total = Number(job.total || job.video_count || 0);
  const done = Number(job.done || (job.state === "done" ? total : 0));
  const percent = total > 0 ? Math.round((100 * done) / total) : 0;
  DOWNLOAD_PROGRESS.hidden = false;
  DOWNLOAD_PROGRESS_BAR.style.width = `${Math.min(percent, 100)}%`;
  DOWNLOAD_PROGRESS_TEXT.textContent = `${done}/${total} clips`;
}

async function pollDownload(job) {
  let current = job;
  while (current.state !== "done") {
    renderProgress(current);
    await sleep(800);
    current = await requestJson(`${DOWNLOAD_URL}/${current.job_id}`);
  }
  renderProgress(current);
  return current;
}

function buildVideoMap(videos) {
  const videoUrls = new Map();
  (videos || []).forEach((video) => {
    videoUrls.set(video.video_path, video);
    videoUrls.set(video.clip_id, video);
  });
  return videoUrls;
}

function stripVideoId(path) {
  return path.split("/").at(-1).split(".")[0].toLowerCase();
}

function renderVideo(card, url) {
  const thumbnail = card.querySelector(".thumbnail");
  const video = document.createElement("video");
  video.src = `${url}#t=0.1`;
  video.controls = true;
  video.preload = "auto";
  video.muted = true;
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  thumbnail.classList.add("video-frame");
  thumbnail.replaceChildren(video);
  video.load();
}

function renderHistory(payload) {
  HISTORY.replaceChildren();
  (payload.queries || []).forEach((item) => {
    const button = document.createElement("button");
    const name = document.createElement("span");
    const count = document.createElement("span");
    button.type = "button";
    button.className = "history-item";
    button.dataset.word = item.word;
    button.dataset.quantity = String(Math.max(1, item.video_count));
    name.textContent = item.query_name;
    count.textContent = `${item.video_count} clips`;
    button.append(name, count);
    HISTORY.append(button);
  });
}

function renderResults(payload, videos = []) {
  LAST_PAYLOAD = payload;
  DOWNLOAD_RESULTS.disabled = payload.results.length === 0;
  clearResults();
  const label = payload.mode === "video" ? `video ${payload.word}` : `"${payload.word}"`;
  SUMMARY.textContent = `Showing ${payload.results.length} of ${payload.quantity} results for ${label}.`;
  const videoUrls = buildVideoMap(videos.length > 0 ? videos : payload.videos);

  if (payload.results.length === 0) {
    const empty = document.createElement("p");
    empty.className = "summary";
    empty.textContent = "No results returned.";
    RESULTS.append(empty);
    return;
  }

  payload.results.forEach((result) => {
    const card = TEMPLATE.content.cloneNode(true);
    const videoInfo = videoUrls.get(result.video_path) || videoUrls.get(stripVideoId(result.video_path));
    const videoUrl = result.video_url || (videoInfo && videoInfo.url);
    if (videoUrl) {
      renderVideo(card, videoUrl);
    }
    card.querySelector(".chunk").textContent = result.chunk;
    card.querySelector(".video-path").textContent = result.local_video_path || (videoInfo && videoInfo.local_video_path) || result.video_path;
    card.querySelector(".result-id").textContent = `ID ${result.id}`;
    card.querySelector(".score").textContent = `Score ${Number(result.score).toFixed(4)}`;
    RESULTS.append(card);
  });
}

async function downloadResults() {
  assert(LAST_PAYLOAD && LAST_PAYLOAD.results);
  const word = LAST_PAYLOAD.word;
  const quantity = LAST_PAYLOAD.quantity;
  DOWNLOAD_RESULTS.disabled = true;
  SUMMARY.textContent = `Downloading ${LAST_PAYLOAD.results.length} videos for "${word}"...`;
  const job = await requestJson(DOWNLOAD_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(LAST_PAYLOAD),
  });
  const payload = await pollDownload(job);
  if (LAST_PAYLOAD.mode === "video") {
    renderResults({ ...LAST_PAYLOAD, videos: payload.videos }, payload.videos);
  } else {
    renderResults(await search(word, quantity), payload.videos);
  }
  renderProgress(payload);
  await loadHistory();
  SUMMARY.textContent = `Saved ${payload.video_count} videos to ${payload.output_dir}.`;
  DOWNLOAD_RESULTS.disabled = false;
}

FORM.addEventListener("submit", async (event) => {
  event.preventDefault();
  const word = QUERY.value.trim();
  const quantity = Number(TOP_K.value);

  assert(word.length > 0);
  assert(Number.isInteger(quantity) && quantity > 0);

  clearResults();
  LAST_PAYLOAD = null;
  DOWNLOAD_RESULTS.disabled = true;
  hideProgress();
  SUMMARY.textContent = `Searching "${word}"...`;
  renderResults(await search(word, quantity));
});

VIDEO_FORM.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = VIDEO_QUERY.files[0];
  const quantity = Number(TOP_K.value);

  assert(file);
  assert(Number.isInteger(quantity) && quantity > 0);

  clearResults();
  LAST_PAYLOAD = null;
  DOWNLOAD_RESULTS.disabled = true;
  hideProgress();
  SUMMARY.textContent = `Embedding "${file.name}"...`;
  renderResults(await searchVideo(file, quantity));
});

DOWNLOAD_RESULTS.addEventListener("click", downloadResults);

HISTORY.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-word]");
  if (!button) {
    return;
  }

  QUERY.value = button.dataset.word;
  TOP_K.value = button.dataset.quantity;
  FORM.requestSubmit();
});

RESULTS.addEventListener("click", (event) => {
  const button = event.target.closest(".rating button");
  if (!button) {
    return;
  }

  button.parentElement.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
});

loadHistory();
