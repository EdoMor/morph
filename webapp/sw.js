/* Service worker (R-603).
 *
 * App shell is cache-first so the installed app opens instantly and survives a
 * dropped connection. API calls are always network-only — a cached agent reply
 * would be a lie.
 */

const VERSION = "morph-v1";
const SHELL = [
  "/",
  "/index.html",
  "/style.css",
  "/app.js",
  "/manifest.webmanifest",
  "/icon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(VERSION)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  if (event.request.method !== "GET" || url.pathname.startsWith("/api/")) {
    return; // never cache the agent
  }

  event.respondWith(
    caches.match(event.request).then((hit) => {
      if (hit) {
        // Stale-while-revalidate: serve the cache, refresh in the background.
        event.waitUntil(
          fetch(event.request)
            .then((fresh) => caches.open(VERSION).then((c) => c.put(event.request, fresh.clone())))
            .catch(() => undefined)
        );
        return hit;
      }
      return fetch(event.request)
        .then((response) => {
          if (response.ok && url.origin === self.location.origin) {
            const copy = response.clone();
            caches.open(VERSION).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => caches.match("/index.html"));
    })
  );
});
