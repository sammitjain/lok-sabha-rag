# Frontend Reference

All frontend code is in `frontend/`. No build step — vanilla HTML/CSS/JS served as static files.
Cache busters: `?v=16` on both `styles.css` and `app.js` links in `index.html`.

---

## HTML Structure (`index.html`)

```
body
└── .container                          # max-width: 860px (1220px with stats)
    ├── header.header
    │   ├── h1                          # "Lok Sabha Questions & Answers"
    │   └── p.subtitle
    ├── main.main
    │   ├── form#search-form
    │   │   ├── .search-box
    │   │   │   ├── .input-wrapper
    │   │   │   │   ├── input#query-input
    │   │   │   │   └── div#mp-dropdown.mp-dropdown
    │   │   │   └── button#submit-btn  (span.btn-text + span.btn-loading>span.spinner)
    │   │   └── .options
    │   │       ├── .mode-toggle       # checkbox#ai-mode (AI synthesis on/off)
    │   │       └── .retrieval-controls
    │   │           ├── input#topk-input   (M: default 50)
    │   │           ├── input#topn-input   (N: default 10)
    │   │           ├── input#cpq-input    (C: default 1)
    │   │           └── input#topq-input   (Q: default 15)
    │   ├── div#active-filters.active-filters   # shows "Filtering by: MP: Name [x]"
    │   ├── div#results-layout.results-layout   # CSS grid: 1fr or 1fr+320px
    │   │   ├── section#results.results
    │   │   │   ├── div#answer-section.answer-section
    │   │   │   │   ├── h2 "Answer"
    │   │   │   │   └── div#answer-content
    │   │   │   └── div.evidence-section
    │   │   │       ├── h2 > span#evidence-title + span#evidence-count
    │   │   │       └── div#evidence-cards.evidence-cards
    │   │   └── aside#stats-panel.stats-panel   # right sidebar, hidden by default
    │   │       ├── .stats-panel-header > h2 "MP Activity Summary"
    │   │       └── div#stats-content.stats-content
    │   └── section#error.error
    │       └── .error-content > p#error-message
    ├── footer.footer
    └── div#modal.modal
        ├── .modal-backdrop
        └── .modal-content
            ├── button#modal-close
            ├── .modal-header > h3#modal-title + .modal-chips#modal-chips
            ├── .modal-body#modal-body
            └── .modal-footer#modal-footer
```

External scripts: `marked.js` (CDN), `app.js` (?v=16)

---

## JavaScript (`app.js`)

### Global State

```javascript
const API_BASE = '/api';
const LOK_NO = 18;                    // hardcoded Lok Sabha number
const CLEAN_CARD_TEXT = true;          // feature flag: lazy-fetch absolute leading chunks on expand
const SYNTH_STAGES = [                // phased synthesizing indicator labels
    'Collecting questions', 'Analysing sources',
    'Collating answers', 'Drafting response'
];
let currentEvidence = [];              // current search results (for modal)
const _questionTextCache = new Map();  // cache for lazy-fetched question text (key: "lok-sess-type-qno-cN")
let _synthTimer = null;                // timer state for synthesizing indicator
let mpList = [];                       // full MP name list from API
let dropdownIndex = -1;                // keyboard nav index in autocomplete
let mpFilter = null;                   // currently selected MP name string
```

### DOM Element References (`elements` object)

```
form, queryInput, submitBtn, aiMode,
topkInput, topnInput, cpqInput, topqInput,
results, answerSection, answerContent,
evidenceTitle, evidenceCount, evidenceCards,
error, errorMessage,
modal, modalClose, modalTitle, modalChips, modalBody, modalFooter,
mpDropdown, activeFilters,
resultsLayout, statsPanel, statsContent
```

### Functions by Section

**MP Autocomplete (lines 43-222):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `loadMpList` | `async ()` | Fetch `/api/members/18`, populate `mpList` |
| `getMpTrigger` | `()` | Detect `mp:` prefix at cursor, return `{search, tokenStart, tokenEnd}` or null |
| `fuzzyMatch` | `(name, search)` | Case-insensitive substring check |
| `highlightMatch` | `(name, search)` | Wrap match in `<span class="mp-match">` |
| `showDropdown` | `(trigger)` | Show up to 8 matching MPs in dropdown |
| `hideDropdown` | `()` | Clear and hide dropdown |
| `selectMp` | `(name, trigger)` | Replace `mp:...` token with name, set `mpFilter`, render filters |
| `renderActiveFilters` | `()` | Show/hide "Filtering by: MP: Name [x]" bar |
| `clearMpFilter` | `()` | Reset `mpFilter` to null |
| `getVisibleDropdownItems` | `()` | Query all `.mp-dropdown-item[data-name]` |
| `handleDropdownKeyboard` | `(e)` | Arrow/Tab/Enter/Escape handling, returns true if consumed |
| `updateDropdownActive` | `(items)` | Toggle `.active` class on dropdown items |

**Shared Helpers (lines 224-289):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `setLoading` | `(loading)` | Toggle button disabled + `.loading` class |
| `showError` | `(message)` | Show error banner, hide results |
| `hideError` | `()` | Hide error banner |
| `formatCitations` | `(text)` | Regex replace `[Q#]` → `<span class="citation" data-index="#">` |
| `formatAskedBy` | `(askedBy)` | Parse MP names (handles JSON arrays, comma-separated, strings) |
| `renderAnswer` | `(answer)` | `marked.parse()` → `formatCitations()` → inject HTML, attach citation click → `scrollToEvidence()` |

**Synthesizing Indicator:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `showSynthesizingIndicator` | `()` | Insert `.synth-indicator` div, start two `setInterval` timers: stage advances every 2s (stops at last), dots cycle 1→2→3→1 every 500ms |
| `hideSynthesizingIndicator` | `()` | Clear both timers, remove `.synth-indicator` element |

**Stats Panel:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `showStatsPanel` | `()` | Add `.has-stats` to container + layout, `.visible` to panel |
| `hideStatsPanel` | `()` | Remove classes, clear `statsContent` innerHTML |
| `renderStatsProgressive` | `(mpStats)` | Build 3 phased DOM sections (MP name + header → ministry bars → recent questions). Shows `by_lok` chips (not session). Each has `.stats-phase` class for staggered CSS animation. Bar widths animated via `data-width` + `requestAnimationFrame`. |

**Evidence Cards (lines 376-597):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `scrollToEvidence` | `(index)` | Scroll to card with `data-group-index`, add `.highlighted` for 2s |
| `groupByQuestionFlat` | `(evidence)` | Client-side grouping by `lok-session-type-ques` key for search mode |
| `buildChips` | `(group)` | Build `[{label, type}]` for Lok/Session/Q# |
| `buildMetaText` | `(group)` | Build `["Ministry: X", "Asked by: Y"]` |
| `buildSecondaryChips` | `(group)` | Build match %, part count, "+N more in PDF" chips |
| `renderChipsHTML` | `(chips)` | Render `<span class="chip ...">` HTML |
| `getCleanCardText` | `(chunks, maxC)` | Sort by chunk_index, slice to maxC, strip header from all but first chunk, join with `\n\n` |
| `getCombinedText` | `(chunks)` | Dispatches to `getCleanCardText` when `CLEAN_CARD_TEXT` is true; otherwise sort + join with `\n\n---\n\n` |
| `fetchQuestionText` | `async (group, c)` | POST `/api/question-text` with `{lok_no, session_no, ques_no, type, c}`; uses `_questionTextCache` keyed by `lok-sess-type-qno-cN` |
| `renderEvidenceCards` | `(data, isGrouped)` | Dispatch to grouped or flat renderer |
| `renderEvidenceCardsGrouped` | `(evidenceGroups)` | Render cards. When `CLEAN_CARD_TEXT`: card-text starts empty, lazy-fetches on first expand with "Loading…" state; `data-loaded`/`data-loading` guards prevent duplicate fetches |
| `renderEvidenceCardsFlat` | `(evidence)` | Group client-side then render. Same lazy-fetch pattern as grouped |

Card features: click main area → open modal (async), "Show full text" toggle (lazy-fetch on first expand), "View PDF" link.

**Modal (lines 599-634):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `openModal` | `async (group)` | Open modal immediately, show "Loading…" while lazy-fetching via `fetchQuestionText()`, populate with rendered markdown on resolve. Cache shared with card expand |
| `closeModal` | `()` | Hide modal, restore body scroll |

**Submit Handler (lines 636-733):**

`handleSubmit(e)` — main flow:
1. Get query, M/N/C/Q params, AI mode, MP filter
2. Build payload (N, C, Q only sent in AI mode)
3. Set loading, hide results + stats
4. **If AI mode**: `showSynthesizingIndicator()` — phased progress display
5. **If MP filter active** (regardless of AI mode): fire `POST /api/mp-stats` in parallel (non-blocking `.then()` renders stats panel immediately on resolve)
6. `await fetch(POST /api/synthesize` or `/api/search`)`
7. **AI mode**: `hideSynthesizingIndicator()` → `renderAnswer()` + `renderEvidenceCards(groups, true)` + fallback stats from synthesize response if parallel fetch wasn't used
8. **Search mode**: hide answer, `renderEvidenceCards(results, false)`
9. Error handling (`hideSynthesizingIndicator()` in catch), finally setLoading(false)

**Event Listeners (lines 735-778):**
- Form submit → dropdown intercept or `handleSubmit`
- Input → detect MP trigger, show/hide dropdown
- Keydown → dropdown keyboard handling
- Blur → delayed dropdown hide (150ms)
- Modal close button + backdrop click + Escape key

**Init (lines 780-783):**
- Focus query input
- `loadMpList()`

---

## CSS (`styles.css`)

### Design Tokens (CSS Variables)

```css
--color-bg: #f8f9fa                    /* page background */
--color-surface: #ffffff               /* cards, inputs */
--color-surface-elevated: #f1f3f5      /* expanded card bg, bar track */
--color-border: #dee2e6                /* primary borders */
--color-border-light: #e9ecef          /* subtle dividers */
--color-text: #212529                  /* body text */
--color-text-muted: #6c757d            /* secondary text */
--color-accent: rgb(77, 162, 118)      /* green - buttons, highlights, citations */
--color-accent-hover: rgb(65, 140, 100)
--color-accent-subtle: rgba(77, 162, 118, 0.1)  /* accent background tint */
--color-link: #0969da                  /* blue links */
--color-error: #dc3545                 /* red errors */
--font-display: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif
--font-body: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif
--radius-sm: 4px
--radius-md: 8px
--radius-lg: 12px
--shadow-sm: 0 1px 3px rgba(0,0,0,0.08)
--shadow-md: 0 4px 12px rgba(0,0,0,0.1)
```

### Layout

- `.container` — max-width 860px, flexbox column, min-height 100vh
- `.container.has-stats` — max-width 1220px (transition 0.3s)
- `.results-layout` — CSS grid, default `1fr`
- `.results-layout.has-stats` — `grid-template-columns: 1fr 320px`

### Major Sections & Key Classes

| Section | Key Classes | Notes |
|---------|-------------|-------|
| Search box | `.search-box`, `.input-wrapper` | Flex row, focus ring on wrapper |
| MP dropdown | `.mp-dropdown`, `.mp-dropdown-item`, `.mp-match` | Absolute positioned, z-index 50, max-height 200px |
| Submit button | `.btn-text`, `.btn-loading`, `.spinner` | Loading state toggles display |
| Toggle | `.toggle`, `.toggle-slider` | 40x22px custom checkbox switch |
| Controls | `.retrieval-controls`, `.topk-control` | Flex row, input width 52px |
| Active filters | `.active-filters`, `.filter-tag`, `.filter-tag-remove` | Hidden by default, uses `.visible` |
| Answer | `.answer-section`, `.answer-content` | Green left border accent, hidden by default |
| Citations | `.citation` | Inline-flex, accent bg, clickable → scrollToEvidence |
| Evidence | `.evidence-cards`, `.evidence-card` | Card with hover border, `.highlighted` for scroll target |
| Card parts | `.card-main`, `.card-header`, `.card-title`, `.card-chips`, `.card-meta`, `.card-footer`, `.card-secondary` | |
| Card expand | `.card-expand`, `.card-expand-toggle`, `.card-text` | Toggle `.visible` for text, `.expanded` rotates arrow. `data-loaded`/`data-loading` attrs for lazy-fetch state |
| Card loading | `.card-text-loading` | Muted italic text shown while lazy-fetching |
| Synth indicator | `.synth-indicator` | Flex row with pulsing green dot `::before` and stage text |
| Chips | `.chip`, `.chip.primary`, `.chip.secondary` | Three variants: default (gray), primary (accent fill), secondary (accent tint) |
| Stats panel | `.stats-panel`, `.stats-panel-header`, `.stats-content` | Hidden by default, `.visible` shows with slideIn animation |
| Stats header | `.stats-header`, `.stats-total`, `.stats-mp-name`, `.stats-big-number`, `.stats-label`, `.stats-breakdown` | Flex layout, MP name + big accent number |
| Stats bars | `.stats-ministries`, `.stats-bar`, `.stats-bar-label` (140px), `.stats-bar-track`, `.stats-bar-fill`, `.stats-bar-count` | `data-width` + rAF animation pattern |
| Stats recent | `.stats-recent`, `.stats-question`, `.stats-q-subject`, `.stats-q-meta` | Ordered list |
| Error | `.error`, `.error-content` | Red border, hidden by default |
| Modal | `.modal`, `.modal-backdrop`, `.modal-content` | Fixed overlay, z-index 100, max-width 700px |
| Modal parts | `.modal-close`, `.modal-header`, `.modal-chips`, `.chip-divider`, `.modal-body`, `.modal-footer` | |

### Animations

| Keyframe | Usage | Effect |
|----------|-------|--------|
| `fadeIn` | Results appear | opacity 0→1, translateY 8→0, 0.3s |
| `spin` | Submit button spinner | rotate 360deg, 0.7s linear infinite |
| `statsPanelIn` | Stats sidebar | opacity 0→1, translateX 20→0, 0.35s |
| `statsPhaseIn` | Stats sections | opacity 0→1, translateY 8→0, 0.4s (staggered: 0s, 0.15s, 0.3s) |
| `modalIn` | Modal open | opacity 0→1, scale 0.96→1, 0.2s |
| `synthPulse` | Synth indicator green dot | opacity 0.25↔1, scale 0.75↔1.2, 1s ease-in-out infinite |

`.stats-bar-fill` uses CSS transition: `width 0.6s ease-out` (not keyframes).

### Responsive Breakpoints

| Breakpoint | Changes |
|------------|---------|
| ≤1024px | `.results-layout.has-stats` → single column; stats panel gets bottom margin |
| ≤600px | Container padding reduced, max-width 100%; search box stacks vertically; options stack; retrieval controls wrap |

---

## UI Feature Matrix

| Feature | AI Mode ON | AI Mode OFF (Search) |
|---------|-----------|---------------------|
| Answer section | Shown (markdown + citations) | Hidden |
| Stats panel | Shown if MP filter active | Shown if MP filter active |
| Evidence cards | Grouped by question (from server) | Grouped client-side |
| Evidence title | "Sources" | "Results" |
| Citations `[Q#]` | Clickable, scroll to card | N/A |
| M/N/C/Q params | All sent to API | Only M sent |
| Parallel stats fetch | Yes (if MP filter) | Yes (if MP filter) |
