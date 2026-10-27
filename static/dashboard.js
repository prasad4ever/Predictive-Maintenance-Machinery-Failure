const counts = document.querySelector("#counts");
const machines = document.querySelector("#machines");
const alarms = document.querySelector("#alarms");
const clock = document.querySelector("#clock");
const link = document.querySelector("#link");

const TONES = { P1: "var(--p1)", P2: "var(--p2)", P3: "var(--p3)", P4: "var(--p4)", OK: "var(--ok)" };

function tone(priority) {
  return TONES[priority] || "var(--muted)";
}

function render(payload) {
  clock.textContent = new Date(payload.timestamp).toLocaleString();
  counts.innerHTML = ["P1", "P2", "P3", "P4"].map((name) => `
    <article class="count" style="--tone:${tone(name)}">
      <strong>${payload.counts[name]}</strong>
      <span>${name}</span>
    </article>
  `).join("");

  machines.innerHTML = payload.machines.map((item) => {
    const open = item.priority !== "OK";
    const drivers = open
      ? `<ul class="drivers">${item.drivers.map((driver) => `<li>${driver.name} ${driver.z > 0 ? "+" : ""}${driver.z}σ</li>`).join("")}</ul>`
      : "";
    return `
      <article class="card" style="--tone:${tone(item.priority)}">
        <header>
          <strong>${item.machine_id}</strong>
          <span class="tag">${item.priority}</span>
        </header>
        <p class="area">${item.machine_name} · ${item.area}</p>
        <p class="component">${open ? item.component_name : "No component failure expected"}</p>
        <p class="${open ? "action" : "quiet"}">${item.action}</p>
        ${drivers}
      </article>
    `;
  }).join("");

  alarms.innerHTML = payload.alarms.length
    ? payload.alarms.map((item) => `
      <article class="alarm" style="--tone:${tone(item.priority)}">
        <header>
          <strong>${item.priority} · ${item.machine_id}</strong>
          <span>${Math.round(item.confidence * 100)}%</span>
        </header>
        <p class="component">${item.component_name}</p>
        <p class="action">${item.action}</p>
      </article>
    `).join("")
    : `<p class="quiet">No component is predicted to fail in the next 24 hours.</p>`;
}

fetch("/api/latest")
  .then((response) => response.json())
  .then((payload) => {
    if (payload.timestamp) render(payload);
  })
  .catch(() => {});

function connect() {
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  socket.onopen = () => {
    link.textContent = "live";
    link.classList.add("live");
  };
  socket.onmessage = (event) => render(JSON.parse(event.data));
  socket.onclose = () => {
    link.textContent = "reconnecting";
    link.classList.remove("live");
    setTimeout(connect, 1500);
  };
}

connect();
