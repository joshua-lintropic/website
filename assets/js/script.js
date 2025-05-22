'use strict';

document.addEventListener('DOMContentLoaded', () => {
  // ─────────────── URL‑based auto‑activation ───────────────
  const path = window.location.pathname;
  if (path.startsWith('/blog')) {
    // clear existing
    document.querySelectorAll('[data-nav-link].active, [data-page].active')
            .forEach(el => el.classList.remove('active'));

    // activate blog nav button
    const blogBtn = Array.from(document.querySelectorAll('[data-nav-link]'))
                         .find(btn => btn.textContent.trim() === 'Blog');
    if (blogBtn) blogBtn.classList.add('active');

    // activate blog section
    const blogSection = document.querySelector('[data-page="blog"]');
    if (blogSection) blogSection.classList.add('active');
  }

  // ─────────────── sidebar toggle ───────────────
  const sidebar        = document.querySelector('[data-sidebar]');
  const sidebarBtn     = document.querySelector('[data-sidebar-btn]');
  sidebarBtn.addEventListener('click', () => {
    sidebar.classList.toggle('active');
  });

  // ─────────────── testimonials modal ───────────────
  const items          = document.querySelectorAll('[data-testimonials-item]');
  const modalContainer = document.querySelector('[data-modal-container]');
  const overlay        = document.querySelector('[data-overlay]');
  const modalImg       = document.querySelector('[data-modal-img]');
  const modalTitle     = document.querySelector('[data-modal-title]');
  const modalText      = document.querySelector('[data-modal-text]');
  function toggleModal() {
    modalContainer.classList.toggle('active');
    overlay.classList.toggle('active');
  }
  items.forEach(item => item.addEventListener('click', () => {
    modalImg.src   = item.querySelector('[data-testimonials-avatar]').src;
    modalImg.alt   = item.querySelector('[data-testimonials-avatar]').alt;
    modalTitle.innerHTML = item.querySelector('[data-testimonials-title]').innerHTML;
    modalText.innerHTML  = item.querySelector('[data-testimonials-text]').innerHTML;
    toggleModal();
  }));
  document.querySelector('[data-modal-close-btn]').addEventListener('click', toggleModal);
  overlay.addEventListener('click', toggleModal);

  // ─────────────── custom select & filtering ───────────────
  const select       = document.querySelector('[data-select]');
  const selectItems  = document.querySelectorAll('[data-select-item]');
  const selectValue  = document.querySelector('[data-selecct-value]');
  const filterBtns   = document.querySelectorAll('[data-filter-btn]');
  const filterItems  = document.querySelectorAll('[data-filter-item]');

  select.addEventListener('click', () => select.classList.toggle('active'));

  function filterFunc(value) {
    filterItems.forEach(item => {
      item.classList.toggle('active',
        value === 'all' || item.dataset.category === value);
    });
  }
  selectItems.forEach(si => si.addEventListener('click', () => {
    const v = si.innerText.toLowerCase();
    selectValue.innerText = si.innerText;
    select.classList.remove('active');
    filterFunc(v);
  }));
  let lastBtn = filterBtns[0];
  filterBtns.forEach(btn => btn.addEventListener('click', () => {
    const v = btn.innerText.toLowerCase();
    selectValue.innerText = btn.innerText;
    filterFunc(v);
    lastBtn.classList.remove('active');
    btn.classList.add('active');
    lastBtn = btn;
  }));

  // ─────────────── contact form enable ───────────────
  const form      = document.querySelector('[data-form]');
  const inputs    = document.querySelectorAll('[data-form-input]');
  const submitBtn = document.querySelector('[data-form-btn]');
  inputs.forEach(i => i.addEventListener('input', () => {
    submitBtn.disabled = !form.checkValidity();
  }));

  // ─────────────── manual nav clicks ───────────────
  const navLinks = document.querySelectorAll('[data-nav-link]');
  const pages    = document.querySelectorAll('[data-page]');
  navLinks.forEach((link, idx) => link.addEventListener('click', () => {
    pages.forEach((page, pIdx) => {
      const isActive = link.textContent.trim().toLowerCase() === page.dataset.page;
      link.classList.toggle('active',   isActive);
      page.classList.toggle('active',   isActive);
    });
    window.scrollTo(0, 0);
  }));

});
