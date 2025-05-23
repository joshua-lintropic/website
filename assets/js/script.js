// --------------------------------------
//  Joshua Lin – Portfolio / Blog
//  Unified site script (sidebar + SPA nav)
//  May 2025
// --------------------------------------

(function () {
  "use strict";

  /* ---------------------------------------------------------------------
   * 1. SIDEBAR TOGGLE (works on all pages that include the sidebar)
   * -------------------------------------------------------------------*/
  const sidebar = document.querySelector("[data-sidebar]");
  const sidebarBtn = document.querySelector("[data-sidebar-btn]");

  sidebarBtn?.addEventListener("click", () => {
    sidebar.classList.toggle("active");
  });

  /* ---------------------------------------------------------------------
   * 2. SINGLE‑PAGE NAVIGATION (index.html only)
   * -------------------------------------------------------------------*/
  const navLinks = document.querySelectorAll("[data-nav-link]");   // buttons in the navbar
  const pages    = document.querySelectorAll("[data-page]");      // <article data-page="about"> …

  // helper → switch visible section & active link
  const activatePage = (pageName = "about") => {
    pages.forEach(page => {
      page.classList.toggle("active", page.dataset.page === pageName);
    });

    navLinks.forEach(link => {
      const name = link.textContent.trim().toLowerCase();
      link.classList.toggle("active", name === pageName);
    });
  };

  // 2.a – click on navbar button  →  change hash & activate
  navLinks.forEach(link => {
    link.addEventListener("click", () => {
      const pageName = link.textContent.trim().toLowerCase();

      // Skip if already on the same section to avoid double‑calling.
      if (window.location.hash.slice(1) !== pageName) {
        window.location.hash = pageName; // triggers hashchange event as well
      }

      activatePage(pageName);
    });
  });

  // 2.b – deep‑link support: on first load & when hash changes (Back/Forward)
  const handleHashChange = () => {
    const hashPage = window.location.hash.slice(1).toLowerCase(); // "resume", "blog", …
    if (hashPage) {
      activatePage(hashPage);
    }
  };

  window.addEventListener("hashchange", handleHashChange);
  document.addEventListener("DOMContentLoaded", handleHashChange);

  const modalContainer = document.querySelector('[data-modal-container]');
  const modalCloseBtn = document.querySelector('[data-modal-close-btn]');
  const testimonialsItems = document.querySelectorAll('[data-testimonials-item]');

  // add click event to all modal items
  for (let i = 0; i < testimonialsItems.length; i++) {
    testimonialsItems[i].addEventListener("click", function () {
      modalImg.src = this.querySelector("[data-testimonials-avatar]").src;
      modalImg.alt = this.querySelector("[data-testimonials-avatar]").alt;
      modalTitle.innerHTML = this.querySelector("[data-testimonials-title]").innerHTML;
      modalText.innerHTML = this.querySelector("[data-testimonials-text]").innerHTML;

      testimonialsModalFunc();
    });
  }
})();
