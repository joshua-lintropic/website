// --------------------------------------
//  Joshua Lin – Portfolio / Blog
//  Unified site script (sidebar + SPA nav + testimonials modal + blog tag filter)
//  May 2025
// --------------------------------------

(function () {
  "use strict";

  /* ---------------------------------------------------------------------
   *  Helpers
   * -------------------------------------------------------------------*/
  const toggleActive = el => el?.classList.toggle("active");

  /* ---------------------------------------------------------------------
   * 1. SIDEBAR TOGGLE (works on all pages that include the sidebar)
   * -------------------------------------------------------------------*/
  const sidebar    = document.querySelector("[data-sidebar]");
  const sidebarBtn = document.querySelector("[data-sidebar-btn]");

  sidebarBtn?.addEventListener("click", () => {
    toggleActive(sidebar);
  });

  /* ---------------------------------------------------------------------
   * 2. SINGLE-PAGE NAVIGATION (index.html only)
   * -------------------------------------------------------------------*/
  const navLinks = document.querySelectorAll("[data-nav-link]"); // buttons in the navbar
  const pages    = document.querySelectorAll("[data-page]");    // <article data-page="about"> …

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

      // Skip if already on the same section to avoid double-calling.
      if (window.location.hash.slice(1) !== pageName) {
        window.location.hash = pageName; // triggers hashchange as well
      }

      activatePage(pageName);
    });
  });

  // 2.b – deep-link support: on first load & when hash changes (Back/Forward)
  const handleHashChange = () => {
    const hashPage = window.location.hash.slice(1).toLowerCase();
    if (hashPage) activatePage(hashPage);
  };

  window.addEventListener("hashchange", handleHashChange);
  document.addEventListener("DOMContentLoaded", handleHashChange);

  /* ---------------------------------------------------------------------
   * 3. TESTIMONIALS MODAL (index.html only)
   * -------------------------------------------------------------------*/
  const modalContainer = document.querySelector("[data-modal-container]");

  if (modalContainer) {
    const modalImg   = modalContainer.querySelector("[data-modal-img]");
    const modalTitle = modalContainer.querySelector("[data-modal-title]");
    const modalText  = modalContainer.querySelector("[data-modal-text]");

    const modalCloseBtn = modalContainer.querySelector("[data-modal-close-btn]");
    const overlay       = document.querySelector("[data-overlay]");
    const testimonialItems = document.querySelectorAll("[data-testimonials-item]");

    const toggleModal = () => {
      toggleActive(modalContainer);
      toggleActive(overlay);
    };

    testimonialItems.forEach(item => {
      item.addEventListener("click", () => {
        const avatar  = item.querySelector("[data-testimonials-avatar]");
        const titleEl = item.querySelector("[data-testimonials-title]");
        const textEl  = item.querySelector("[data-testimonials-text]");

        if (avatar) {
          modalImg.src = avatar.src;
          modalImg.alt = avatar.alt;
        }
        modalTitle.innerHTML = titleEl?.innerHTML ?? "";
        modalText.innerHTML  = textEl?.innerHTML  ?? "";

        toggleModal();
      });
    });

    modalCloseBtn?.addEventListener("click", toggleModal);
    overlay?.addEventListener("click", toggleModal);
  }

  /* ---------------------------------------------------------------------
   * 4. BLOG TAG FILTER (index.html → Blog section)
   * -------------------------------------------------------------------*/
  const selectEl = document.querySelector("[data-select]");

  if (selectEl) {
    const selectValue  = document.querySelector("[data-selecct-value]"); // markup typo kept for compat
    const selectItems  = document.querySelectorAll("[data-select-item]");
    const filterBtns   = document.querySelectorAll("[data-filter-btn]");
    const filterItems  = document.querySelectorAll("[data-filter-item]");

    const applyFilter = value => {
      filterItems.forEach(item => {
        item.classList.toggle(
          "active",
          value === "all" || item.dataset.category === value
        );
      });
    };

    // 4.a – open / close dropdown (mobile)
    selectEl.addEventListener("click", () => toggleActive(selectEl));

    // 4.b – choose tag from dropdown
    selectItems.forEach(item => {
      item.addEventListener("click", () => {
        const value = item.textContent.trim().toLowerCase();
        selectValue.textContent = item.textContent.trim();
        toggleActive(selectEl);
        applyFilter(value);
      });
    });

    // 4.c – large‑screen pill buttons
    let lastClickedBtn = filterBtns[0];
    filterBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const value = btn.textContent.trim().toLowerCase();
        selectValue.textContent = btn.textContent.trim();
        applyFilter(value);

        lastClickedBtn?.classList.remove("active");
        btn.classList.add("active");
        lastClickedBtn = btn;
      });
    });
  }
})();
