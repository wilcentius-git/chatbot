const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const btnSend = document.getElementById("btnSend");
const btnClear = document.getElementById("btnClear");
const toastEl = document.getElementById("toast");

function toast(msg, ms=1800) {
  toastEl.textContent = msg;
  toastEl.classList.remove("hidden");
  setTimeout(() => toastEl.classList.add("hidden"), ms);
}

function addMessage(role, text, meta=null) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  if (role !== "user") {
    const av = document.createElement("div");
    av.className = "avatar";
    av.textContent = "KB";
    wrap.appendChild(av);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text || "";
  wrap.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function send() {
  const msg = (inputEl.value || "").trim();
  if (!msg) return;

  addMessage("user", msg);
  inputEl.value = "";
  inputEl.focus();

  // typing indicator
  const typingId = `typing-${Date.now()}`;
  addMessage("bot", "…");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg})
    });

    // remove last "…"
    chatEl.lastChild?.remove();

    if (!res.ok) {
      addMessage("bot", `Server error (${res.status}).`);
      return;
    }

    const data = await res.json();
    const meta = data.intent ? `Mode: ${data.intent}${data.matched ? ` • Match: ${data.matched}` : ""}${(data.score !== undefined && data.score !== null) ? ` • Score: ${Number(data.score).toFixed(2)}` : ""}` : null;

    addMessage("bot", data.reply || "(tidak ada respons)", meta);
  } catch (e) {
    chatEl.lastChild?.remove();
    addMessage("bot", "Gagal menghubungi server. Pastikan backend sedang berjalan.");
    console.error(e);
  }
}

document.querySelectorAll(".chip").forEach(btn => {
  btn.addEventListener("click", () => {
    inputEl.value = btn.dataset.msg || "";
    send();
  });
});

function clearChat() {
  chatEl.innerHTML = "";
  addMessage("bot",
    "Halo. Saya bisa bantu FAQ (akun, OTP, reset password) dan referensi KUHP (contoh: 'Pasal 362 KUHP')."
  );
}

btnSend.addEventListener("click", send);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});

btnClear.addEventListener("click", () => {
  clearChat();
  toast("Chat dibersihkan.");
});

// init
clearChat();
inputEl.focus();
