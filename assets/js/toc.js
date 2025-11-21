document.addEventListener('DOMContentLoaded', () => {
    console.log('TOC: Script loaded');

    // Only run on research pages
    const article = document.querySelector('article[data-page="blog-post"]');
    if (!article) {
        console.log('TOC: Not a blog post page, skipping');
        return;
    }

    // Check if TOC is enabled (defaults to true)
    const tocEnabled = article.getAttribute('data-toc-enabled');
    if (tocEnabled === 'false') {
        console.log('TOC: Disabled via data-toc-enabled attribute');
        return;
    }

    console.log('TOC: Generating table of contents');

    // Find all h3 and h4 headers within the article
    const headers = article.querySelectorAll('.about-text h3, .about-text h4');

    if (headers.length === 0) {
        console.log('TOC: No headers found');
        return;
    }

    // Generate unique IDs for headers that don't have them
    headers.forEach((header, index) => {
        if (!header.id) {
            const text = header.textContent.trim();
            const id = text
                .toLowerCase()
                .replace(/[^\w\s-]/g, '')
                .replace(/\s+/g, '-')
                .replace(/-+/g, '-')
                .substring(0, 50);
            header.id = id || `header-${index}`;
        }
    });

    // Build TOC list
    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';

    let currentH3Item = null;
    let currentH4List = null;

    headers.forEach((header) => {
        const li = document.createElement('li');
        li.className = header.tagName === 'H3' ? 'toc-item-h3' : 'toc-item-h4';

        const link = document.createElement('a');
        link.href = `#${header.id}`;
        link.className = 'toc-link';
        link.textContent = header.textContent.trim();
        link.setAttribute('data-target', header.id);

        // Smooth scroll to section
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.getElementById(header.id);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Update URL without jumping
                history.pushState(null, null, `#${header.id}`);
            }
        });

        li.appendChild(link);

        if (header.tagName === 'H3') {
            tocList.appendChild(li);
            currentH3Item = li;
            currentH4List = null;
        } else if (header.tagName === 'H4') {
            // Create nested list for h4 items under h3
            if (currentH3Item) {
                if (!currentH4List) {
                    currentH4List = document.createElement('ul');
                    currentH4List.className = 'toc-list-nested';
                    currentH3Item.appendChild(currentH4List);
                }
                currentH4List.appendChild(li);
            } else {
                // No parent h3, add directly to main list
                tocList.appendChild(li);
            }
        }
    });

    // Create TOC sidebar container
    const tocSidebar = document.createElement('aside');
    tocSidebar.className = 'toc-sidebar';
    tocSidebar.setAttribute('data-toc', '');

    const tocHeader = document.createElement('div');
    tocHeader.className = 'toc-header';

    const tocTitle = document.createElement('h3');
    tocTitle.className = 'toc-title';
    tocTitle.textContent = 'Table of Contents';

    const collapseBtn = document.createElement('button');
    collapseBtn.className = 'toc-collapse-btn';
    collapseBtn.title = 'Collapse table of contents';
    collapseBtn.innerHTML = '<ion-icon name="chevron-forward-outline"></ion-icon>';

    tocHeader.appendChild(tocTitle);
    tocHeader.appendChild(collapseBtn);
    tocSidebar.appendChild(tocHeader);

    const tocContent = document.createElement('div');
    tocContent.className = 'toc-content';
    tocContent.appendChild(tocList);
    tocSidebar.appendChild(tocContent);

    // Insert TOC into the DOM
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.appendChild(tocSidebar);
        console.log('TOC: Injected successfully');

        // Restore collapsed state from localStorage
        const isTocCollapsed = localStorage.getItem('tocCollapsed') === 'true';
        if (isTocCollapsed) {
            tocSidebar.classList.add('collapsed');
        }

        // Add collapse button event listener
        collapseBtn.addEventListener('click', () => {
            tocSidebar.classList.toggle('collapsed');
            const collapsed = tocSidebar.classList.contains('collapsed');
            localStorage.setItem('tocCollapsed', collapsed);
            console.log('TOC collapsed:', collapsed);
        });
    } else {
        console.error('TOC: Could not find .main-content element');
        return;
    }

    // Highlight active section on scroll
    let ticking = false;
    const tocLinks = tocSidebar.querySelectorAll('.toc-link');

    function updateActiveSection() {
        const scrollPosition = window.scrollY + 100; // Offset for better UX

        let activeHeader = null;
        headers.forEach((header) => {
            const headerTop = header.offsetTop;
            if (scrollPosition >= headerTop) {
                activeHeader = header;
            }
        });

        // Update active class
        tocLinks.forEach((link) => {
            const targetId = link.getAttribute('data-target');
            if (activeHeader && targetId === activeHeader.id) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    window.addEventListener('scroll', () => {
        if (!ticking) {
            window.requestAnimationFrame(() => {
                updateActiveSection();
                ticking = false;
            });
            ticking = true;
        }
    });

    // Initial update
    updateActiveSection();
});
