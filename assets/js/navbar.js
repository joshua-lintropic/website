const assetsPath = window.assetsPath || '.';
const isHomePage = window.location.pathname.endsWith('index.html') || window.location.pathname.endsWith('/');

// Helper to determine if a link should be active
const isActive = (linkHash) => {
  if (isHomePage) {
    // On home page, check if the hash matches
    return window.location.hash === linkHash || (linkHash === '#about' && !window.location.hash);
  }
  return false;
};

const getLinkHref = (hash) => {
  if (isHomePage) {
    return hash;
  }
  return `${assetsPath}/index.html${hash}`;
};

const navbarHTML = `
<nav class="navbar">
  <ul class="navbar-list">
    <li class="navbar-item">
      <a href="${getLinkHref('#about')}" class="navbar-link ${isActive('#about') ? 'active' : ''}" data-nav-link>About</a>
    </li>
    <li class="navbar-item">
      <a href="${getLinkHref('#resume')}" class="navbar-link ${isActive('#resume') ? 'active' : ''}" data-nav-link>Resume</a>
    </li>
    <li class="navbar-item">
      <a href="${getLinkHref('#research')}" class="navbar-link ${isActive('#research') ? 'active' : ''}" data-nav-link>Research</a>
    </li>
    <li class="navbar-item">
      <a href="${getLinkHref('#blog')}" class="navbar-link ${isActive('#blog') ? 'active' : ''}" data-nav-link>Blog</a>
    </li>
  </ul>
</nav>
`;

// Inject navbar into the main content area
const mainContent = document.querySelector('.main-content');
if (mainContent) {
  mainContent.insertAdjacentHTML('afterbegin', navbarHTML);
} else {
  console.error('Navbar could not be injected: .main-content not found.');
}
