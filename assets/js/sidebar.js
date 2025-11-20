const assetsPath = window.assetsPath || '.';

const sidebarHTML = `
<aside class="sidebar" data-sidebar>

  <div class="sidebar-info">

    <figure class="avatar-box">
      <img src="${assetsPath}/assets/images/my-pfp.webp" alt="Joshua Lin" width="80">
    </figure>

    <div class="info-content">
      <h1 class="name" title="Joshua Lin">Joshua Lin</h1>

      <p class="title">math + cs</p>
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

          <address>Princeton, NJ, USA</address>
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

document.querySelector('main').insertAdjacentHTML('afterbegin', sidebarHTML);
