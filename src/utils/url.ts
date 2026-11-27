const base = import.meta.env.BASE_URL.replace(/\/$/, '');

/** Build a site-internal URL that respects the configured `base` path. */
export const url = (path = '') => `${base}/${path.replace(/^\//, '')}`;
