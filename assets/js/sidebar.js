document.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded event fired.');
  console.log('sidebar.js is loading...');
  console.log('window.assetsPath:', window.assetsPath);

  const assetsPath = window.assetsPath || '.';
  console.log('Using assetsPath:', assetsPath);

  const sidebarHTML = `
<aside class="sidebar" data-sidebar>

  <div class="sidebar-info">

    <figure class="avatar-box">
      <img src="${assetsPath}/assets/images/my-pfp.webp" alt="Joshua Lin" width="80">
    </figure>

    <div class="info-content">
      <h1 class="name" title="Joshua Lin">Joshua Lin</h1>

      <p class="title">Math @ Princeton</p>

      <button class="sidebar-collapse-btn" title="Collapse sidebar">
        <ion-icon name="chevron-back-outline"></ion-icon>
      </button>
    </div>

    <button class="info_more-btn" data-sidebar-btn>
      <span>Show Contacts</span>

      <ion-icon name="chevron-down"></ion-icon>
    </button>

  </div>

  <div class="sidebar-info_more">

    <div class="separator"></div>

    <ul class="contacts-list">

      <li class="contact-item">

        <div class="icon-box">
          <ion-icon name="mail-outline"></ion-icon>
        </div>

        <div class="contact-info">
          <p class="contact-title">Email</p>

          <a href="mailto:joshua.lin@princeton.edu" class="contact-link">joshua.lin@princeton.edu</a>
        </div>

      </li>

      <li class="contact-item">

        <div class="icon-box">
          <ion-icon name="phone-portrait-outline"></ion-icon>
        </div>

        <div class="contact-info">
          <p class="contact-title">Phone</p>

          <a href="tel:+15596910093" class="contact-link">+1 (559) 691-0093</a>
        </div>

      </li>

      <li class="contact-item">

        <div class="icon-box">
          <ion-icon name="school-outline"></ion-icon>
        </div>

        <div class="contact-info">
          <p class="contact-title">Education</p>

            <a href="https://www.princeton.edu/" class="contact-link">Princeton University</a>
        </div>

      </li>

      <li class="contact-item">

        <div class="icon-box">
          <ion-icon name="location-outline"></ion-icon>
        </div>

        <div class="contact-info">
          <p class="contact-title">Location</p>

          <address>New Jersey, USA</address>
        </div>

      </li>

    </ul>

    <div class="separator"></div>

    <ul class="social-list">

      <li class="social-item">
        <a href="https://github.com/joshua-lintropic" class="social-link">
          <ion-icon name="logo-github"></ion-icon>
        </a>
      </li>

      <li class="social-item">
        <a href="https://www.linkedin.com/in/joshua-linsanity/" class="social-link">
          <ion-icon name="logo-linkedin"></ion-icon>
        </a>
      </li>

      <li class="social-item">
        <a href="https://x.com/lintropicjoshua" class="social-link">
          <ion-icon name="logo-twitter"></ion-icon>
        </a>
      </li>

    </ul>

  </div>

</aside>
`;

  console.log('Sidebar HTML length:', sidebarHTML.length);
  console.log('Looking for main element...');

  const main = document.querySelector('main');
  console.log('Main element found:', main);

  if (main) {
    console.log('Injecting sidebar...');
    main.insertAdjacentHTML('afterbegin', sidebarHTML);
    console.log('Sidebar injected successfully');

    // Verify injection
    const sidebar = document.querySelector('.sidebar');
    console.log('Sidebar element after injection:', sidebar);

    // Collapse functionality
    if (sidebar) {
      // Restore collapsed state from localStorage
      const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
      if (isCollapsed) {
        sidebar.classList.add('collapsed');
      }

      // Add collapse button event listener
      const collapseBtn = sidebar.querySelector('.sidebar-collapse-btn');
      if (collapseBtn) {
        collapseBtn.addEventListener('click', () => {
          sidebar.classList.toggle('collapsed');
          const collapsed = sidebar.classList.contains('collapsed');
          localStorage.setItem('sidebarCollapsed', collapsed);
          console.log('Sidebar collapsed:', collapsed);
        });
      }
    }
  } else {
    console.error('Sidebar injection failed: main element not found');
  }
});
