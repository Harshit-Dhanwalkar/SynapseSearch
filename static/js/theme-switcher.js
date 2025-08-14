// static/js/theme-switcher.js
document.addEventListener("DOMContentLoaded", function () {
  const themeToggle = document.getElementById("theme-toggle");
  const themeIcons = {
    dark: document.querySelector('[data-theme-icon="dark"]'),
    light: document.querySelector('[data-theme-icon="light"]'),
    cyberpunk: document.querySelector('[data-theme-icon="cyberpunk"]'),
  };
  const tooltip = document.querySelector(".theme-tooltip");
  const themes = ["dark", "light", "cyberpunk"];

  function getPreferredTheme() {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) return savedTheme;

    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);

    // Update icons
    Object.values(themeIcons).forEach((icon) => {
      icon.classList.add("hidden");
    });
    themeIcons[theme].classList.remove("hidden");

    // Update tooltip
    // if (tooltip) {
    //   tooltip.textContent = `Change theme (Current: ${
    //     theme.charAt(0).toUpperCase() + theme.slice(1)
    //   })`;
    // }
  }

  function cycleTheme() {
    const currentTheme =
      document.documentElement.getAttribute("data-theme") ||
      getPreferredTheme();
    const currentIndex = themes.indexOf(currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
  }

  // Initialize
  setTheme(getPreferredTheme());

  // Event listeners
  if (themeToggle) {
    themeToggle.addEventListener("click", cycleTheme);
    themeToggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        cycleTheme();
      }
    });
  }

  // Watch for system theme changes
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", (e) => {
      if (!localStorage.getItem("theme")) {
        setTheme(e.matches ? "dark" : "light");
      }
    });
});
