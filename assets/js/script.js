// --------------------------------------
//  Joshua Lin – Portfolio / Blog
//  Unified site script (sidebar + SPA nav + testimonials modal)
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

  /* ---------------------------------------------------------------------
   * 3. TESTIMONIALS MODAL (index.html only)
   * -------------------------------------------------------------------*/
  const modalContainer = document.querySelector("[data-modal-container]");

  // Guard‑clause: only run if the testimonials modal exists on the page.
  if (modalContainer) {
    const modalImg   = modalContainer.querySelector("[data-modal-img]");
    const modalTitle = modalContainer.querySelector("[data-modal-title]");
    const modalText  = modalContainer.querySelector("[data-modal-text]");

    const modalCloseBtn = modalContainer.querySelector("[data-modal-close-btn]");
    const overlay       = document.querySelector("[data-overlay]");
    const testimonialItems = document.querySelectorAll("[data-testimonials-item]");

    // toggle helper – mirrors the behaviour in the legacy script
    const toggleModal = () => {
      modalContainer.classList.toggle("active");
      overlay?.classList.toggle("active");
    };

    // populate + open on card click
    testimonialItems.forEach(item => {
      item.addEventListener("click", () => {
        const avatar   = item.querySelector("[data-testimonials-avatar]");
        const titleEl  = item.querySelector("[data-testimonials-title]");
        const textEl   = item.querySelector("[data-testimonials-text]");

        if (avatar) {
          modalImg.src = avatar.src;
          modalImg.alt = avatar.alt;
        }
        modalTitle.innerHTML = titleEl?.innerHTML ?? "";
        modalText.innerHTML  = textEl?.innerHTML  ?? "";

        toggleModal();
      });
    });

    // close events – button & dimmed overlay
    modalCloseBtn?.addEventListener("click", toggleModal);
    overlay?.addEventListener("click", toggleModal);
  }
})();
