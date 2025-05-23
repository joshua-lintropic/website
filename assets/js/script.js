// --------------------------------------
//  Joshua Lin – Portfolio / Blog
//  Unified site script (sidebar + SPA nav + testimonials modal + blog tag filter)
//  May 2025
// --------------------------------------

(function () {
  "use strict";

  /* ---------------------------------------------------------------------
   * Helpers
   * -------------------------------------------------------------------*/
  const toggleActive = el => el?.classList.toggle("active");

  /* ---------------------------------------------------------------------
   * 1. SIDEBAR TOGGLE (works on all pages that include the sidebar)
   * -------------------------------------------------------------------*/
  const sidebar    = document.querySelector("[data-sidebar]");
  const sidebarBtn = document.querySelector("[data-sidebar-btn]");
  sidebarBtn?.addEventListener("click", () => toggleActive(sidebar));

  /* ---------------------------------------------------------------------
   * 2. SINGLE‑PAGE NAVIGATION (index.html only)
   * -------------------------------------------------------------------*/
  const navLinks = document.querySelectorAll("[data-nav-link]");
  const pages    = document.querySelectorAll("[data-page]");

  const activatePage = (pageName = "about") => {
    pages.forEach(page => page.classList.toggle("active", page.dataset.page === pageName));
    navLinks.forEach(link => {
      const name = link.textContent.trim().toLowerCase();
      link.classList.toggle("active", name === pageName);
    });
  };

  navLinks.forEach(link => {
    link.addEventListener("click", () => {
      const pageName = link.textContent.trim().toLowerCase();
      if (window.location.hash.slice(1) !== pageName) {
        window.location.hash = pageName; // also triggers hashchange
      }
      activatePage(pageName);
    });
  });

  const handleHashChange = () => {
    const hash = window.location.hash.slice(1).toLowerCase();
    if (hash) activatePage(hash);
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
    const testimonialCards = document.querySelectorAll("[data-testimonials-item]");

    const toggleModal = () => {
      toggleActive(modalContainer);
      toggleActive(overlay);
    };

    testimonialCards.forEach(card => {
      card.addEventListener("click", () => {
        const avatar = card.querySelector("[data-testimonials-avatar]");
        const title  = card.querySelector("[data-testimonials-title]");
        const text   = card.querySelector("[data-testimonials-text]");

        if (avatar) {
          modalImg.src = avatar.src;
          modalImg.alt = avatar.alt;
        }
        modalTitle.innerHTML = title?.innerHTML ?? "";
        modalText.innerHTML  = text?.innerHTML  ?? "";

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
    // Some templates still ship with the historic misspelling “selecct”. Support both.
    const selectValue = document.querySelector("[data-select-value]") ||
                        document.querySelector("[data-selecct-value]");

    const dropdownItems = document.querySelectorAll("[data-select-item]");
    const filterBtns    = document.querySelectorAll("[data-filter-btn]");
    const posts         = document.querySelectorAll("[data-filter-item]");

    const applyFilter = tag => {
      posts.forEach(post => {
        const match = tag === "all" || post.dataset.category === tag;
        post.classList.toggle("active", match);
      });
    };

    // 4.a – dropdown open/close (touch + mobile)
    selectEl.addEventListener("click", () => toggleActive(selectEl));

    // 4.b – choose tag from dropdown list
    dropdownItems.forEach(item => {
      item.addEventListener("click", () => {
        const tag = item.textContent.trim().toLowerCase();
        if (selectValue) selectValue.textContent = item.textContent.trim();
        toggleActive(selectEl);
        applyFilter(tag);
      });
    });

    // 4.c – large‑screen pill buttons
    let lastActiveBtn = Array.from(filterBtns).find(btn => btn.classList.contains("active")) || filterBtns[0];

    filterBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const tag = btn.textContent.trim().toLowerCase();
        if (selectValue) selectValue.textContent = btn.textContent.trim();
        applyFilter(tag);

        lastActiveBtn?.classList.remove("active");
        btn.classList.add("active");
        lastActiveBtn = btn;
      });
    });

    // 4.d – First‑load: show posts according to any pre‑selected tag or default to “all”
    const initialTag = (lastActiveBtn?.textContent || "all").trim().toLowerCase();
    applyFilter(initialTag);
  }
})();
