function generateCarousel(ids_to_label, containerId) {
  const carouselContainer = document.getElementById(containerId);
  const splideHTML = `
      <div id="${containerId}-glide">
        <div data-glide-el="track" class="glide__track">
          <ul class="glide__slides">
            ${ids_to_label[containerId]
              .map(
                (imageSrc) => `
                      <li class="glide__slide">
                        <img class="gimg" src="https://lh3.googleusercontent.com/d/${imageSrc}=s220?authuser=0" alt="${containerId} image" loading="lazy" />
                      </li>
                  `,
              )
              .join("")}
          </ul>
        </div>
        <div class="glide__arrows" data-glide-el="controls">
          <button class="glide__arrow glide__arrow--left" data-glide-dir="<" aria-label="previous">
            <svg fill="none" viewBox="0 0 24 24"><path d="m15 18-6-6 6-6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>
          </button>
          <button class="glide__arrow glide__arrow--right" data-glide-dir=">" aria-label="next">
            <svg fill="none" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>
          </button>
        </div>
      </div>
  `;
  carouselContainer.innerHTML = splideHTML;
}

for (const key in fileIdsByLabel) {
  fileIdsByLabel[key] = fileIdsByLabel[key]
    .map((value) => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value);

  generateCarousel(fileIdsByLabel, key);
  var glide = new Glide(`#${key}-glide`, {
    type: "carousel",
    perView: 5,
    breakpoints: {
      600: {
        perView: 3,
      },
    },
  });
  glide.mount();
}
