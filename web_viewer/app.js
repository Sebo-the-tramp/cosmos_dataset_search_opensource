const API_URL = "/search";
const SEARCH_VIDEO_URL = "/search_video";
const DOWNLOAD_URL = "/download";
const HISTORY_URL = "/history";
const FORM = document.querySelector("#search-form");
const VIDEO_FORM = document.querySelector("#video-search-form");
const QUERY = document.querySelector("#query");
const TOP_K = document.querySelector("#top-k");
const METADATA_FILTER = document.querySelector("#metadata-filter");
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

function buildSearchUrl(word, quantity, metadataFilter) {
  const url = new URL(API_URL, window.location.origin);
  url.searchParams.set("word", word);
  url.searchParams.set("quantity", String(quantity));
  if (metadataFilter) {
    url.searchParams.set("metadata_filter", JSON.stringify(metadataFilter));
  }
  return url;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  assert(response.ok);
  return response.json();
}

function readMetadataFilter() {
  const text = METADATA_FILTER.value.trim();
  if (!text) {
    return null;
  }
  const metadataFilter = JSON.parse(text);
  assert(metadataFilter && typeof metadataFilter === "object" && !Array.isArray(metadataFilter));
  assert(typeof metadataFilter.field === "string" && metadataFilter.field.length > 0);
  assert(typeof metadataFilter.operator === "string" && metadataFilter.operator.length > 0);
  assert(Object.hasOwn(metadataFilter, "value"));
  return metadataFilter;
}

async function search(word, quantity, metadataFilter) {
  return requestJson(buildSearchUrl(word, quantity, metadataFilter));
}

async function searchVideo(file, quantity, metadataFilter) {
  const data = new FormData();
  data.append("video", file);
  data.append("quantity", String(quantity));
  if (metadataFilter) {
    data.append("metadata_filter", JSON.stringify(metadataFilter));
  }
  return requestJson(SEARCH_VIDEO_URL, { method: "POST", body: data });
}

async function loadHistory() {
  renderHistory(await requestJson(HISTORY_URL));
}

async function deleteHistory(queryName) {
  return requestJson(`${HISTORY_URL}/${encodeURIComponent(queryName)}`, { method: "DELETE" });
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
  const options = thumbnail.querySelector(".options-button");
  const video = document.createElement("video");
  video.src = `${url}#t=0.1`;
  video.controls = true;
  video.preload = "auto";
  video.muted = true;
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  thumbnail.classList.add("video-frame");
  thumbnail.replaceChildren(video, options);
  video.load();
}

async function copyPath(button) {
  await navigator.clipboard.writeText(button.dataset.path);
  button.textContent = "Copied";
  setTimeout(() => {
    button.textContent = "Copy path";
  }, 1200);
}

function metadataFromResult(result) {
  const metadata = {};
  Object.entries(result).forEach(([key, value]) => {
    if (!["score", "metadata", "video_url", "local_video_path", "video_downloaded"].includes(key)) {
      metadata[key] = value;
    }
  });
  if (result.metadata && typeof result.metadata === "object") {
    Object.assign(metadata, result.metadata);
  }
  return metadata;
}

function renderHistory(payload) {
  HISTORY.replaceChildren();
  (payload.queries || []).forEach((item) => {
    const row = document.createElement("div");
    const button = document.createElement("button");
    const deleteButton = document.createElement("button");
    const name = document.createElement("span");
    const count = document.createElement("span");
    row.className = "history-item";
    button.type = "button";
    button.className = "history-query";
    button.dataset.word = item.word;
    button.dataset.quantity = String(Math.max(1, item.video_count));
    deleteButton.type = "button";
    deleteButton.className = "history-delete";
    deleteButton.dataset.deleteQuery = item.query_name;
    deleteButton.setAttribute("aria-label", `Delete ${item.query_name}`);
    name.textContent = item.query_name;
    count.textContent = `${item.video_count} clips`;
    button.append(name, count);
    deleteButton.textContent = "Delete";
    row.append(button, deleteButton);
    HISTORY.append(row);
  });
}

function renderResults(payload, videos = []) {
  LAST_PAYLOAD = payload;
  DOWNLOAD_RESULTS.disabled = payload.results.length === 0;
  clearResults();
  const label = payload.mode === "video" ? `video ${payload.word}` : `"${payload.word}"`;
  const filterText = payload.metadata_filter
    ? ` Metadata filter ${payload.metadata_filter_applied ? "applied" : "recorded; not applied yet"}.`
    : "";
  SUMMARY.textContent = `Showing ${payload.results.length} of ${payload.quantity} results for ${label}.${filterText}`;
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
    const path = result.local_video_path || (videoInfo && videoInfo.local_video_path) || result.video_path;
    card.querySelector(".chunk").textContent = result.chunk;
    card.querySelector(".video-path").textContent = path;
    card.querySelector(".result-id").textContent = `ID ${result.id}`;
    card.querySelector(".metadata").textContent = JSON.stringify(metadataFromResult(result), null, 2);
    card.querySelector(".score").textContent = `Score ${Number(result.score).toFixed(4)}`;
    card.querySelector(".copy-path").dataset.path = path;
    RESULTS.append(card);
  });
}

async function downloadResults() {
  assert(LAST_PAYLOAD && LAST_PAYLOAD.results);
  const word = LAST_PAYLOAD.word;
  const quantity = LAST_PAYLOAD.quantity;
  const metadataFilter = LAST_PAYLOAD.metadata_filter;
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
    renderResults(await search(word, quantity, metadataFilter), payload.videos);
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
  const metadataFilter = readMetadataFilter();

  assert(word.length > 0);
  assert(Number.isInteger(quantity) && quantity > 0);

  clearResults();
  LAST_PAYLOAD = null;
  DOWNLOAD_RESULTS.disabled = true;
  hideProgress();
  SUMMARY.textContent = `Searching "${word}"...`;
  renderResults(await search(word, quantity, metadataFilter));
});

VIDEO_FORM.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = VIDEO_QUERY.files[0];
  const quantity = Number(TOP_K.value);
  const metadataFilter = readMetadataFilter();

  assert(file);
  assert(Number.isInteger(quantity) && quantity > 0);

  clearResults();
  LAST_PAYLOAD = null;
  DOWNLOAD_RESULTS.disabled = true;
  hideProgress();
  SUMMARY.textContent = `Embedding "${file.name}"...`;
  renderResults(await searchVideo(file, quantity, metadataFilter));
});

DOWNLOAD_RESULTS.addEventListener("click", downloadResults);

RESULTS.addEventListener("click", async (event) => {
  const copyButton = event.target.closest(".copy-path");
  if (copyButton) {
    await copyPath(copyButton);
    return;
  }

  const button = event.target.closest(".options-button");
  if (!button) {
    return;
  }

  const card = button.closest(".result-card");
  const metadata = card.querySelector(".metadata");
  metadata.hidden = !metadata.hidden;
  button.setAttribute("aria-expanded", String(!metadata.hidden));
});

HISTORY.addEventListener("click", (event) => {
  const deleteButton = event.target.closest("button[data-delete-query]");
  if (deleteButton) {
    if (!window.confirm(`Delete ${deleteButton.dataset.deleteQuery}?`)) {
      return;
    }
    deleteHistory(deleteButton.dataset.deleteQuery).then(loadHistory);
    return;
  }

  const button = event.target.closest("button[data-word]");
  if (!button) {
    return;
  }
  QUERY.value = button.dataset.word;
  TOP_K.value = button.dataset.quantity;
  FORM.requestSubmit();
});

loadHistory();
