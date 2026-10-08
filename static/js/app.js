/**
 * AlphaBot - Frontend Client
 * CodeAlpha AI Internship Task 2: Chatbot for FAQs
 */

document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("userInput");
  const messagesArea = document.getElementById("messagesArea");
  const typingIndicator = document.getElementById("typingIndicator");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const openSidebarBtn = document.getElementById("openSidebarBtn");
  const closeSidebarBtn = document.getElementById("closeSidebarBtn");
  const sidebar = document.getElementById("sidebar");
  const initTime = document.getElementById("initTime");

  if (initTime) {
    initTime.textContent = getCurrentTime();
  }

  // Sidebar toggles
  if (openSidebarBtn && sidebar) {
    openSidebarBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  if (closeSidebarBtn && sidebar) {
    closeSidebarBtn.addEventListener("click", () => {
      sidebar.classList.remove("open");
    });
  }

  // Reset chat
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
      if (confirm("Reset the conversation?")) {
        window.location.reload();
      }
    });
  }

  // Focus input on load
  if (userInput) {
    userInput.focus();
  }
});

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  const messagesArea = document.getElementById("messagesArea");
  if (messagesArea) {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  }
}

function showTypingIndicator() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    indicator.classList.add("active");
    scrollToBottom();
  }
}

function hideTypingIndicator() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    indicator.classList.remove("active");
  }
}

function sendQuickPrompt(promptText) {
  const userInput = document.getElementById("userInput");
  if (userInput) {
    userInput.value = promptText;
    const form = document.getElementById("chatForm");
    if (form) {
      form.requestSubmit();
    }
  }

  // If on mobile and sidebar open, close it
  const sidebar = document.getElementById("sidebar");
  if (sidebar && sidebar.classList.contains("open")) {
    sidebar.classList.remove("open");
  }
}

async function handleUserSubmit(event) {
  event.preventDefault();
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const message = userInput.value.trim();

  if (!message) return;

  // Append user message
  appendUserMessage(message);
  userInput.value = "";
  userInput.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: message }),
    });

    const data = await response.json();
    hideTypingIndicator();

    if (data.success && data.response) {
      appendBotResponse(data.response);
    } else {
      appendBotResponse({
        answer: "Sorry, I encountered an issue processing your request. Please try again.",
        intent: "error",
      });
    }
  } catch (error) {
    console.error("Chat error:", error);
    hideTypingIndicator();
    appendBotResponse({
      answer: "Unable to connect to the chatbot server. Please ensure the backend is running.",
      intent: "error",
    });
  } finally {
    userInput.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    userInput.focus();
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const messagesArea = document.getElementById("messagesArea");
  const time = getCurrentTime();

  const msgDiv = document.createElement("div");
  msgDiv.className = "message user-message";
  msgDiv.innerHTML = `
    <div class="avatar-sm">
      <i class="fa-solid fa-user"></i>
    </div>
    <div class="message-content">
      <div class="bubble">
        <p>${escapeHtml(text)}</p>
      </div>
      <span class="message-time">${time}</span>
    </div>
  `;

  messagesArea.appendChild(msgDiv);
  scrollToBottom();
}

function appendBotResponse(respData) {
  const messagesArea = document.getElementById("messagesArea");
  const time = getCurrentTime();

  const msgDiv = document.createElement("div");
  msgDiv.className = "message bot-message";

  let metaHtml = "";
  if (respData.intent === "faq_match" && respData.confidence > 0) {
    const percent = Math.round(respData.confidence * 100);
    const confidenceClass = percent >= 50 ? "" : "moderate";
    metaHtml = `
      <div class="match-meta">
        ${respData.category ? `<span class="meta-pill pill-category"><i class="fa-solid fa-folder"></i> ${escapeHtml(respData.category)}</span>` : ""}
        <span class="meta-pill pill-confidence ${confidenceClass}">
          <i class="fa-solid fa-bolt"></i> ${percent}% match
        </span>
        ${respData.matched_question ? `<span class="pill-matched">Re: ${escapeHtml(respData.matched_question)}</span>` : ""}
      </div>
    `;
  }

  let suggestionsHtml = "";
  if (respData.suggestions && respData.suggestions.length > 0) {
    const chips = respData.suggestions
      .map(
        (sug) =>
          `<button class="prompt-chip" onclick="sendQuickPrompt('${escapeHtml(sug).replace(/'/g, "\\'")}')">${escapeHtml(sug)}</button>`
      )
      .join("");

    suggestionsHtml = `
      <div class="quick-prompts-wrapper">
        <span class="prompt-title">Related Questions:</span>
        <div class="quick-prompts">
          ${chips}
        </div>
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="avatar-sm">
      <i class="fa-solid fa-robot"></i>
    </div>
    <div class="message-content">
      <div class="bubble">
        <p>${escapeHtml(respData.answer)}</p>
        ${metaHtml}
      </div>
      ${suggestionsHtml}
      <span class="message-time">${time}</span>
    </div>
  `;

  messagesArea.appendChild(msgDiv);
  scrollToBottom();
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
