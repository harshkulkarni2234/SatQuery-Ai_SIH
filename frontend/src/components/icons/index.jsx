const ic = (paths) =>
  function Icon({ size = 16 }) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {paths}
      </svg>
    );
  };

export const IconSatellite = ic(
  <>
    <rect x="8" y="8" width="8" height="8" rx="1.5" />
    <path d="M4 8h4M16 8h4M4 16h4M16 16h4" />
    <path d="M8 8 12 4h3" />
  </>
);
export const IconUpload = ic(
  <>
    <path d="M12 15V4" />
    <path d="m7 9 5-5 5 5" />
    <path d="M4 18v1a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1" />
  </>
);
export const IconTrash = ic(
  <>
    <path d="M4 7h16" />
    <path d="M10 11v6M14 11v6" />
    <path d="M6 7l1 14h10l1-14" />
    <path d="M9 7V4h6v3" />
  </>
);
export const IconCheck = ic(<path d="m4 12 5 5L20 6" />);
export const IconPlay = ic(<path d="m7 5 13 7-13 7V5Z" />);
export const IconArrow = ic(
  <>
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </>
);
export const IconChevron = ic(<path d="m6 9 6 6 6-6" />);
export const IconEye = ic(
  <>
    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z" />
    <circle cx="12" cy="12" r="3" />
  </>
);
export const IconPin = ic(
  <>
    <path d="M12 17v5" />
    <path d="M9 2h6l-1 7H10Z" />
    <circle cx="12" cy="9" r="5" />
  </>
);
export const IconCompare = ic(
  <>
    <rect x="2" y="4" width="8" height="16" rx="1" />
    <rect x="14" y="4" width="8" height="16" rx="1" />
    <path d="M10 10h4M10 14h4" />
  </>
);
export const IconLayers = ic(
  <>
    <path d="m12 3 9 5-9 5-9-5 9-5Z" />
    <path d="m3 13 9 5 9-5" />
  </>
);
export const IconImage = ic(
  <>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="9" cy="9" r="2" />
    <path d="m21 15-4.5-4.5L6 21" />
  </>
);
export const IconLayer2 = ic(
  <>
    <rect x="3" y="3" width="7" height="7" rx="1" />
    <path d="m14 3 7 7" />
    <path d="M3 14l7 7" />
    <rect x="14" y="14" width="7" height="7" rx="1" />
  </>
);
export const IconPlus = ic(
  <>
    <path d="M12 5v14M5 12h14" />
  </>
);
